/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.read.orchestration.CentroidPlanner;
import org.opensearch.knn.clusterann.read.orchestration.ClusterSearcher;
import org.opensearch.knn.clusterann.read.orchestration.PlanParams;

import java.io.IOException;
import java.util.Arrays;

/**
 * Per-segment {@link KnnVectorsReader} for {@link KNN1030ClusterANNVectorsFormat}.
 *
 * <p>In this initial implementation the reader exposes the raw, full-precision
 * vectors and answers nearest-neighbor queries by delegating to the wrapped
 * {@link FlatVectorsReader}, which performs an exact (brute-force) scan. Cluster-based
 * ANN traversal would replace or augment that delegated search in a later iteration.
 *
 * @see KNN1030ClusterANNVectorsFormat
 * @see KNN1030ClusterANNVectorsWriter
 */
@Log4j2
public class KNN1030ClusterANNVectorsReader extends KnnVectorsReader {

    private final SegmentReadState segmentReadState;
    private final FlatVectorsReader rawFlatVectorsReader;
    private final IntObjectHashMap<ClusterANNFieldMeta> fields;
    private final FieldInfos fieldInfos;

    /**
     * One {@link Clusters} per vector field, by field number — the query-independent handle a search runs over, so it
     * is built once here and shared by every search of this segment rather than per query.
     */
    private final IntObjectHashMap<Clusters> clusters;

    /**
     * The data files, opened only when a field actually has clusters. Held for the segment's lifetime because every
     * {@link Clusters} reads through clones of them; closed with this reader.
     *
     * <p>{@code rotation} is {@code null} unless some field in this segment is rotated: {@code .clar} holds nothing
     * for an unrotated field, so it is never written and there is no file to open.
     */
    private final IndexInput postings;
    private final IndexInput centroids;
    private final IndexInput rotation;

    /**
     * Creates a reader for a single segment.
     *
     * @param state    state describing the segment being read
     * @param rawFlatVectorsReader delegate reader that provides access to the raw vectors
     */
    public KNN1030ClusterANNVectorsReader(final SegmentReadState state, final FlatVectorsReader rawFlatVectorsReader) throws IOException {
        this.rawFlatVectorsReader = rawFlatVectorsReader;
        this.segmentReadState = state;
        this.fields = new IntObjectHashMap<>();
        this.clusters = new IntObjectHashMap<>();
        this.fieldInfos = state.fieldInfos;
        boolean success = false;

        final String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN1030ClusterANNVectorsFormat.META_EXTENSION
        );
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                CodecUtil.checkIndexHeader(
                    meta,
                    KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                    KNN1030ClusterANNVectorsFormat.VERSION_START,
                    KNN1030ClusterANNVectorsFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                int blockSize = meta.readVInt();
                readFieldMetadata(meta, blockSize);
            } catch (Throwable exception) {
                priorE = exception;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }

        // A segment with no ClusterANN field has no postings or centroids to open — and opening them anyway would
        // fail on a segment this format never wrote data for.
        if (fields.isEmpty()) {
            this.postings = null;
            this.centroids = null;
            this.rotation = null;
            return;
        }

        success = false;
        try {
            this.postings = openInput(state, KNN1030ClusterANNVectorsFormat.POSTINGS_EXTENSION);
            this.centroids = openInput(state, KNN1030ClusterANNVectorsFormat.CENTROIDS_EXTENSION);
            this.rotation = hasRotatedField() ? openInput(state, KNN1030ClusterANNVectorsFormat.ROTATION_EXTENSION) : null;
            for (IntObjectHashMap.IntObjectCursor<ClusterANNFieldMeta> field : fields) {
                final int fieldNumber = field.key;
                final ClusterANNFieldMeta fieldMeta = field.value;
                final String suffix = String.join("-", String.valueOf(fieldNumber), segmentReadState.segmentSuffix);
                clusters.put(
                    field.key,
                    new Clusters(
                        postings.slice("clap_" + suffix, fieldMeta.clapOffset(), fieldMeta.clapLength()),
                        centroids.slice("centroids_" + suffix, fieldMeta.clacOffset(), fieldMeta.clacLength()),
                        rotationSlice(fieldMeta, suffix),
                        field.value
                    )
                );
            }
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    /**
     * Whether any field in this segment carries a rotation, and so whether {@code .clar} was written at all. A
     * segment of unrotated fields has no such file, so opening one would fail on an index that is perfectly valid.
     */
    private boolean hasRotatedField() {
        for (IntObjectHashMap.IntObjectCursor<ClusterANNFieldMeta> field : fields) {
            if (field.value.hasRotation()) {
                return true;
            }
        }
        return false;
    }

    /**
     * The field's region of {@code .clar}, or {@code null} when it stores no rotation — an unrotated field carries
     * {@link org.opensearch.knn.clusterann.format.ClusterANNFormatConstants#NO_ROTATION} for both
     * the offset and the length, which is no region at all.
     */
    private IndexInput rotationSlice(ClusterANNFieldMeta fieldMeta, String suffix) throws IOException {
        if (!fieldMeta.hasRotation()) {
            return null;
        }
        return rotation.slice("rotation_" + suffix, fieldMeta.clarOffset(), fieldMeta.clarLength());
    }

    private static IndexInput openInput(SegmentReadState state, String extension) throws IOException {
        String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        return state.directory.openInput(name, state.context);
    }

    /**
     * Verifies the integrity of the underlying vector files, typically by validating their checksums.
     *
     * @throws IOException if an integrity check fails or the files cannot be read
     */
    @Override
    public void checkIntegrity() throws IOException {
        rawFlatVectorsReader.checkIntegrity();
        for (IndexInput input : Arrays.asList(postings, centroids, rotation)) {
            if (input != null) {
                CodecUtil.checksumEntireFile(input);
            }
        }
    }

    /**
     * Returns the float vectors stored for the given field.
     *
     * @param fieldName the name of the vector field
     * @return the field's {@link FloatVectorValues}, or {@code null} if the field has no float vectors
     * @throws IOException if the vectors cannot be read
     */
    @Override
    public FloatVectorValues getFloatVectorValues(final String fieldName) throws IOException {
        return rawFlatVectorsReader.getFloatVectorValues(fieldName);
    }

    /**
     * Returns the byte vectors stored for the given field.
     *
     * @param fieldName the name of the vector field
     * @return the field's {@link ByteVectorValues}, or {@code null} if the field has no byte vectors
     * @throws IOException if the vectors cannot be read
     */
    @Override
    public ByteVectorValues getByteVectorValues(final String fieldName) throws IOException {
        return rawFlatVectorsReader.getByteVectorValues(fieldName);
    }

    /**
     * Finds the nearest neighbors of a float query vector, feeding matches to the collector.
     *
     * @param fieldName    the name of the vector field to search
     * @param query        the float query vector
     * @param knnCollector collector that gathers the closest matching documents
     * @param acceptDocs    the set of documents that are allowed to match
     * @throws IOException if the search cannot be performed
     */
    @Override
    public void search(String fieldName, float[] query, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        if (fieldName == null || query == null || knnCollector == null || acceptDocs == null) {
            throw new IllegalArgumentException("fieldName, query, knn collector and  vector cannot be null");
        }

        Clusters fieldClusters = clusters(fieldName);
        if (fieldClusters == null) {
            return;
        }

        final LongValues ordToDoc = fieldClusters.ordToDoc();
        final OrdinalTranslatedKnnCollector translatedKnnCollector = new OrdinalTranslatedKnnCollector(
            knnCollector,
            (ord) -> Math.toIntExact(ordToDoc.get(ord))
        );
        Bits acceptedOrds = buildAcceptedOrds(acceptDocs, ordToDoc, fieldClusters.numVectors());

        int[] probes = CentroidPlanner.plan(fieldClusters, query, PlanParams.of(fieldClusters.numClusters()));

        float[] scanQuery = new float[query.length];
        fieldClusters.rotation().rotate(query, scanQuery);

        ClusterSearcher.search(fieldClusters, probes, ScanParams.of(scanQuery), translatedKnnCollector, acceptedOrds);
    }

    private Bits buildAcceptedOrds(AcceptDocs acceptDocs, LongValues ordToDoc, int numVectors) throws IOException {
        if (acceptDocs == null) return null;

        Bits docBits = acceptDocs.bits();
        if (docBits == null) return null; // match-all

        return new Bits() {
            @Override
            public boolean get(int ord) {
                return docBits.get((int) ordToDoc.get(ord));
            }

            @Override
            public int length() {
                return numVectors;
            }
        };
    }

    private Clusters clusters(String fieldName) {
        FieldInfo info = fieldInfos.fieldInfo(fieldName);
        return info == null ? null : clusters.get(info.number);
    }

    /**
     * Finds the nearest neighbors of a byte query vector, feeding matches to the collector.
     *
     * @param fieldName    the name of the vector field to search
     * @param query        the byte query vector
     * @param knnCollector collector that gathers the closest matching documents
     * @param acceptDocs    the set of documents that are allowed to match
     */
    @Override
    public void search(String fieldName, byte[] query, KnnCollector knnCollector, AcceptDocs acceptDocs) {
        throw new UnsupportedOperationException("Search on byte vector not supported with cluster ann");
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     *
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        IOUtils.close(rawFlatVectorsReader, postings, centroids, rotation);
    }

    private void readFieldMetadata(ChecksumIndexInput meta, int blockSize) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != KNN1030ClusterANNVectorsFormat.NO_MORE_FIELDS; fieldNumber = meta.readInt()) {
            FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }
            final ClusterANNFieldMeta field = ClusterANNFieldMeta.read(meta, blockSize);
            validateFieldEntry(info, field, segmentReadState.segmentInfo.maxDoc(), meta);
            if (!field.isEmpty()) {
                fields.put(fieldNumber, field);
            }
        }
    }

    private static void validateFieldEntry(FieldInfo info, ClusterANNFieldMeta field, int maxDoc, ChecksumIndexInput meta)
        throws IOException {
        if (!info.hasVectorValues()) {
            throw new CorruptIndexException("Field \"" + info.name + "\" has a ClusterANN entry but carries no vector values", meta);
        }
        if (info.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new CorruptIndexException(
                "Field \""
                    + info.name
                    + "\" is encoded as "
                    + info.getVectorEncoding()
                    + ", but ClusterANN stores only "
                    + VectorEncoding.FLOAT32,
                meta
            );
        }
        if (info.getVectorDimension() != field.dimension()) {
            throw new CorruptIndexException(
                "Field \""
                    + info.name
                    + "\" has dimension "
                    + info.getVectorDimension()
                    + ", but its ClusterANN entry says "
                    + field.dimension(),
                meta
            );
        }
        if (info.getVectorSimilarityFunction() != field.similarityFunction()) {
            throw new CorruptIndexException(
                "Field \""
                    + info.name
                    + "\" uses "
                    + info.getVectorSimilarityFunction()
                    + ", but its ClusterANN entry says "
                    + field.similarityFunction(),
                meta
            );
        }
        // A vector field is single-valued, so a segment cannot hold more vectors than documents. SOAR copies do
        // not lift this: they are extra postings counted in clusterSizes, not extra vectors.
        if (field.vectorCount() > maxDoc) {
            throw new CorruptIndexException(
                "Field \"" + info.name + "\" has " + field.vectorCount() + " vectors, but the segment holds only " + maxDoc + " documents",
                meta
            );
        }
    }
}
