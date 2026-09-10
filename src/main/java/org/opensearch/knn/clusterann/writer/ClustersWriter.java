package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.clusterann.reader.Centroid;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.writer.rotation.RotationWriter;
import org.opensearch.knn.clusterann.writer.rotation.RotationWriterFactory;

import org.opensearch.common.Nullable;
import java.io.IOException;

/**
 * Writes one field's clusters across {@code .clac}, {@code .clap}, {@code .clar} and {@code .clam} — the write-side
 * counterpart of {@code Clusters}, and the orchestrator the rest of this package hangs off.
 *
 * <p><b>It runs bottom-up, and that is forced rather than chosen.</b> An offset cannot be recorded until the bytes it
 * points at exist, so the field's {@code .clam} entry is written last, once every other file has taken its share.
 * That makes this the only object that knows where everything landed — {@code .clam} is exactly that knowledge,
 * serialized. The read side runs the other way, meta first, which is why the two meet at
 * {@link ClusterANNFieldMeta} and nowhere else.
 *
 * <p>The stages, in order, each depending only on the ones above it:
 *
 * <ol>
 *   <li><b>cluster</b> — a black box; nothing downstream depends on how it decided
 *   <li><b>rotate</b> — chosen and persisted here, applied to everything stored from here on
 *   <li><b>arrange</b> — group into postings and order each by {@code ‖c−v‖}; the read side's pruning rests on this
 *   <li><b>centroids</b> — {@code .clac}, in both spaces when the field is rotated
 *   <li><b>postings</b> — {@code .clap}, recording each cluster's offset and length as it goes
 *   <li><b>entry</b> — {@code .clam}, now that every offset exists
 * </ol>
 *
 * <p>Flush and merge differ in exactly one thing: where the vectors come from. Both arrive through
 * {@link #write}, so nothing below this line can drift between them.
 *
 * <p>One instance per segment; not thread-safe. Lucene drives a segment's writer from a single thread.
 */
public final class ClustersWriter {

    private final IndexOutput meta;
    private final IndexOutput centroids;
    private final IndexOutput postings;
    private final IndexOutput rotationOut;
    private final ClusterANNWriteParams params;
    private final Clustering clustering;

    /**
     * @param meta the {@code .clam} output, positioned past its header and the shared block size
     * @param centroids the {@code .clac} output
     * @param postings the {@code .clap} output
     * @param rotationOut the {@code .clar} output, or {@code null} when no field is rotated
     * @param clustering how vectors are partitioned; a black box to everything here
     */
    public ClustersWriter(
        IndexOutput meta,
        IndexOutput centroids,
        IndexOutput postings,
        @Nullable IndexOutput rotationOut,
        ClusterANNWriteParams params,
        Clustering clustering
    ) {
        this.meta = meta;
        this.centroids = centroids;
        this.postings = postings;
        this.rotationOut = rotationOut;
        this.params = params;
        this.clustering = clustering;
    }

    /**
     * Clusters, encodes and writes one field, then appends its {@code .clam} entry.
     *
     * @param vectors random access by ordinal, because a posting is written in distance order while the vectors
     *     arrive in document order
     * @param maxDoc the segment's document count, which is what distinguishes a dense field — every document has a
     *     vector, so {@code ord == doc} and no mapping is stored — from a sparse one
     */
    public void write(FieldInfo fieldInfo, VectorSource vectors, int maxDoc) throws IOException {
        meta.writeInt(fieldInfo.number);
        if (vectors.size() == 0) {
            // Still writes the ord -> doc entry, even though there is no mapping: the reader reads its sentinel
            // unconditionally, so an entry that omits it is nineteen bytes short and everything after it misparses.
            emptyEntry(fieldInfo).write(meta, writeOrdToDoc(vectors, maxDoc));
            return;
        }

        VectorSimilarityFunction similarity = fieldInfo.getVectorSimilarityFunction();
        int dimension = vectors.dimension();

        // 1. Cluster. Everything below depends only on the result, never on how it was reached.
        ClusteringResult result = clustering.cluster(vectors, similarity, params);

        // 2. Rotate. The rotation applied here and the one written to .clar must be the same object — a mismatch is
        // undetectable at read time, since scores in the wrong space are still plausible numbers.
        RotationWriter rotation = RotationWriterFactory.create(params.rotationId(), dimension, params.seed());
        VectorSource stored = rotated(vectors, rotation);
        float[][] storedCentroids = rotation.rotationId() == ClusterANNFieldMeta.ROTATION_NONE
            ? result.centroids()
            : CentroidWriter.rotate(result.centroids(), rotation);

        // 3. Arrange. Distances are measured to the centroids in the *stored* space, because that is the space the
        // codes and every bound derived from them live in.
        Posting[] arranged = PostingArranger.arrange(result, stored, storedCentroids, similarity);

        // 4. Rotation matrix and centroids.
        long clarOffset = ClusterANNFieldMeta.NO_ROTATION;
        long clarLength = ClusterANNFieldMeta.NO_ROTATION;
        if (rotation.rotationId() != ClusterANNFieldMeta.ROTATION_NONE) {
            clarOffset = rotationOut.getFilePointer();
            clarLength = rotation.write(rotationOut);
        }

        long clacOffset = centroids.getFilePointer();
        CentroidWriter.Offsets clac = CentroidWriter.write(centroids, clacOffset, result.centroids(), rotation);

        // 5. Postings. Everything in .clap is addressed relative to this field's region, because the reader slices
        // the file at clapOffset before using any of it.
        long clapOffset = postings.getFilePointer();
        byte[] ordToDocMeta = writeOrdToDoc(vectors, maxDoc);

        ClusterWriter clusterWriter = ClusterWriterFactory.create(params, dimension, similarity);
        long[] centroidOffsets = new long[arranged.length];
        int[] centroidLengths = new int[arranged.length];
        int[] clusterSizes = new int[arranged.length];
        for (int c = 0; c < arranged.length; c++) {
            centroidOffsets[c] = postings.getFilePointer() - clapOffset;
            clusterWriter.write(postings, arranged[c], reference(storedCentroids[c]), stored);
            centroidLengths[c] = Math.toIntExact(postings.getFilePointer() - clapOffset - centroidOffsets[c]);
            clusterSizes[c] = arranged[c].size();
        }
        long clapLength = postings.getFilePointer() - clapOffset;

        // 6. The entry, last.
        new ClusterANNFieldMeta(
            params.blockSize(),
            dimension,
            vectors.size(),
            arranged.length,
            similarity,
            params.docBits(),
            rotation.rotationId(),
            params.quantizerId(),
            buildParams(),
            clacOffset,
            clac.length(),
            clac.rawOffset(),
            clac.rotatedOffset(),
            clapOffset,
            clapLength,
            centroidOffsets,
            centroidLengths,
            clusterSizes,
            clarOffset,
            clarLength,
            null
        ).write(meta, ordToDocMeta);
    }

    /**
     * Writes the {@code ord → doc} mapping into this field's {@code .clap} region and returns its metadata.
     *
     * <p>Both halves go through buffers, for two separate reasons. The metadata has to land <em>last</em> in the
     * {@code .clam} entry, while {@code writeStoredMeta} emits it interleaved with the data as it goes. And the data's
     * recorded offsets have to be relative to this field's region, which they are only if the output's pointer starts
     * at zero — the reader slices {@code .clap} at {@code clapOffset} and applies the offsets to the slice.
     *
     * <p>Writing this as the first thing in the region is therefore not cosmetic: it is what makes the buffered
     * pointer and the region-relative offsets the same number.
     */
    private byte[] writeOrdToDoc(VectorSource vectors, int maxDoc) throws IOException {
        DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        for (int ord = 0; ord < vectors.size(); ord++) {
            docsWithField.add(vectors.docId(ord));
        }

        ByteBuffersDataOutput metaBytes = new ByteBuffersDataOutput();
        ByteBuffersDataOutput dataBytes = new ByteBuffersDataOutput();
        try (
            ByteBuffersIndexOutput metaBuffer = new ByteBuffersIndexOutput(metaBytes, "ordToDocMeta", "ordToDocMeta");
            ByteBuffersIndexOutput dataBuffer = new ByteBuffersIndexOutput(dataBytes, "ordToDocData", "ordToDocData")
        ) {
            OrdToDocDISIReaderConfiguration.writeStoredMeta(
                params.monotonicBlockShift(),
                metaBuffer,
                dataBuffer,
                vectors.size(),
                maxDoc,
                docsWithField
            );
        }
        dataBytes.copyTo(postings);
        return metaBytes.toArrayCopy();
    }

    /**
     * A view that rotates on the way out, rather than a rotated copy of every vector.
     *
     * <p>A vector is read once per posting it appears in — twice when SOAR gave it a spill — so rotating on access
     * costs at most one extra transform per spill, against materialising the whole field.
     */
    private static VectorSource rotated(VectorSource source, RotationWriter rotation) {
        if (rotation.rotationId() == ClusterANNFieldMeta.ROTATION_NONE) {
            return source;
        }
        return new VectorSource() {
            private final float[] destination = new float[source.dimension()];

            @Override
            public int size() {
                return source.size();
            }

            @Override
            public int dimension() {
                return source.dimension();
            }

            @Override
            public float[] vector(int ord) throws IOException {
                rotation.rotate(source.vector(ord), destination);
                return destination;
            }

            @Override
            public int docId(int ord) {
                return source.docId(ord);
            }
        };
    }

    /** A centroid paired with {@code ‖c‖²}, which the quantizer's corrective terms are expressed against. */
    private static Centroid reference(float[] centroid) {
        float normSq = 0f;
        for (float value : centroid) {
            normSq += value * value;
        }
        return new Centroid(centroid, normSq);
    }

    /**
     * The entry for a field with no vectors.
     *
     * <p>Full shape, zeros where there is nothing to point at, so the reader parses one layout unconditionally. The
     * dimension and the similarity are the field's real ones rather than placeholders: both are known even with no
     * vectors, and the reader rejects a non-positive dimension.
     */
    private ClusterANNFieldMeta emptyEntry(FieldInfo fieldInfo) {
        return new ClusterANNFieldMeta(
            params.blockSize(),
            fieldInfo.getVectorDimension(),
            0,
            0,
            fieldInfo.getVectorSimilarityFunction(),
            params.docBits(),
            ClusterANNFieldMeta.ROTATION_NONE,
            params.quantizerId(),
            new byte[0],
            0L,
            0L,
            0L,
            ClusterANNFieldMeta.NO_ROTATION,
            0L,
            0L,
            new long[0],
            new int[0],
            new int[0],
            ClusterANNFieldMeta.NO_ROTATION,
            ClusterANNFieldMeta.NO_ROTATION,
            null
        );
    }

    /**
     * The build parameters the reader does not need but nobody can reconstruct later: the target cluster size, the
     * SOAR weight and the seed. Carried in {@code quantizerParams} so a segment can be explained after the fact —
     * {@code centroidCount} says how many clusters there are, never why.
     */
    private byte[] buildParams() throws IOException {
        ByteBuffersDataOutput out = new ByteBuffersDataOutput();
        out.writeVInt(params.targetClusterSize());
        out.writeInt(Float.floatToIntBits(params.soarLambda()));
        out.writeLong(params.seed());
        out.writeVInt(params.monotonicBlockShift());
        return out.toArrayCopy();
    }

    /** Releases the outputs this writer was given, in the order Lucene expects on failure. */
    public void closeOutputs() throws IOException {
        IOUtils.close(meta, centroids, postings, rotationOut);
    }
}
