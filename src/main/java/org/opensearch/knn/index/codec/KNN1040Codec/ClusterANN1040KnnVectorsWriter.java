/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.index.clusterann.ClusterANNVectorValues;
import org.opensearch.knn.index.clusterann.ClusteringResult;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.algorithm.IVFIndexBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.*;
import org.opensearch.knn.index.clusterann.codec.*;
import org.opensearch.knn.index.clusterann.algorithm.RandomRotation;

/**
 * Writer for ClusterANN IVF format.
 *
 * <p>Four files (plus the flat {@code .vec}/{@code .vem} for full-precision vectors):
 * <ul>
 *   <li>{@code .clam} — per-field metadata: field header, centroid stats (docCounts, norms,
 *       posting sizes), the {@code .clap} centroid offset table, the two {@code .clac} offsets
 *       (centroid block, region 1), and the {@code .clar} rotation offset. Read once at open,
 *       held in heap.</li>
 *   <li>{@code .clac} — centroid block (raw + transformed centroids) followed by region 1
 *       (per-field ordToDoc/doc→ord {@code OrdToDocDISIReaderConfiguration} plus the
 *       ordToCentroid array). mmap'd.</li>
 *   <li>{@code .clar} — serialized random rotation per rotated field (L2 only); the {@code .clam}
 *       rotation offset is -1 for fields with no rotation. mmap'd.</li>
 *   <li>{@code .clap} — one posting per centroid, columnar:
 *       {@code [ordinals | soarBitset | sortedDistances | quantized blocks]}. Primary + SOAR
 *       are merged into a single run sorted by centroid-to-vector distance ‖c−v‖ (descending for
 *       IP, ascending for L2/cosine). docIds are NOT stored here — they come from ordToDoc.</li>
 * </ul>
 *
 * <p>Centroids are written in spatial order (sorted by first principal component) for locality.
 */
@Log4j2
public class ClusterANN1040KnnVectorsWriter extends KnnVectorsWriter {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(ClusterANN1040KnnVectorsWriter.class);

    private final SegmentWriteState state;
    private final FlatVectorsWriter flatVectorsWriter;
    private final byte docBits;
    private final List<FieldWriterInfo> fields = new ArrayList<>();

    private final IndexOutput metaOutput;
    private final IndexOutput postingsOutput;
    private final IndexOutput centroidsOutput;
    private final IndexOutput rotationOutput;

    private static class FieldWriterInfo {
        final FieldInfo fieldInfo;
        final FlatFieldVectorsWriter<float[]> flatFieldWriter;

        FieldWriterInfo(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> flatFieldWriter) {
            this.fieldInfo = fieldInfo;
            this.flatFieldWriter = flatFieldWriter;
        }
    }

    public ClusterANN1040KnnVectorsWriter(SegmentWriteState state, FlatVectorsWriter flatVectorsWriter, int docBits) throws IOException {
        this.state = state;
        this.flatVectorsWriter = flatVectorsWriter;
        this.docBits = validateDocBits(docBits);

        boolean success = false;
        try {
            metaOutput = createOutput(META_EXTENSION);
            postingsOutput = createOutput(POSTINGS_EXTENSION);
            centroidsOutput = createOutput(CENTROIDS_EXTENSION);
            rotationOutput = createOutput(ROTATION_EXTENSION);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        @SuppressWarnings("unchecked")
        FlatFieldVectorsWriter<float[]> flatFieldWriter = (FlatFieldVectorsWriter<float[]>) flatVectorsWriter.addField(fieldInfo);
        fields.add(new FieldWriterInfo(fieldInfo, flatFieldWriter));
        return flatFieldWriter;
    }

    // ========== Flush ==========

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);
        for (FieldWriterInfo field : fields) {
            List<float[]> vectors = field.flatFieldWriter.getVectors();
            if (vectors.isEmpty()) {
                writeEmptyMeta(field.fieldInfo);
                continue;
            }
            int dimension = field.fieldInfo.getVectorDimension();
            int[] docIds = collectDocIds(field.flatFieldWriter, vectors.size());
            ClusterANNVectorValues vectorValues = ClusterANNVectorValues.fromList(vectors, docIds, dimension);
            writeIVF(field.fieldInfo, vectorValues, null);
        }
    }

    // ========== Merge ==========

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        float[][] reservoir = new float[4096][];
        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromMergeState(mergeState, fieldInfo, reservoir);

        if (vectors.size() == 0) {
            writeEmptyMeta(fieldInfo);
            return;
        }

        int numCentroids = estimateCentroids(vectors.size());
        int actualReservoir = Math.min(vectors.size(), reservoir.length);
        float[][] initialCentroids = Arrays.copyOf(reservoir, Math.min(numCentroids, actualReservoir));

        writeIVF(fieldInfo, vectors, initialCentroids);
    }

    // ========== Shared write path ==========

    private void writeIVF(FieldInfo fieldInfo, ClusterANNVectorValues vectors, float[][] initialCentroids) throws IOException {
        int numVectors = vectors.size();
        int dimension = fieldInfo.getVectorDimension();
        DistanceMetric metric = toDistanceMetric(fieldInfo.getVectorSimilarityFunction());

        // 1. Cluster (on original vectors — clustering doesn't need randomRotation)
        ClusteringResult result = IVFIndexBuilder.build(vectors, TARGET_CLUSTER_SIZE, metric, SOAR_LAMBDA, initialCentroids, 42L, true);

        int numCentroids = result.numCentroids();
        float[][] centroids = result.centroids();

        // 1b. Create random rotation and transform centroids for quantization (L2 only). Spreading
        // variance across dimensions improves scalar-quantization fidelity for Euclidean.
        boolean useRotation = fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
        RandomRotation randomRotation = RandomRotation.create(dimension);
        float[][] transformedCentroids = new float[numCentroids][dimension];
        for (int c = 0; c < numCentroids; c++) {
            if (useRotation) {
                randomRotation.transform(centroids[c], transformedCentroids[c]);
            } else {
                System.arraycopy(centroids[c], 0, transformedCentroids[c], 0, dimension);
            }
        }

        // 2. Spatial sort centroids (by first principal component)
        int[] spatialOrder = spatialSort(centroids, dimension);

        // 3. Write posting lists to .clap (in spatial order, primary+soar adjacent)
        postingsOutput.alignFilePointer(SECTION_ALIGNMENT);
        long postingsFieldOffset = postingsOutput.getFilePointer();
        long[] centroidOffsets = new long[numCentroids];
        int[] postingSizes = new int[numCentroids]; // exact byte size per centroid

        int[][] primaryPostings = result.primaryPostingLists();
        int[][] soarPostings = result.soarPostingLists();

        // IP orders posting distances descending (highest ceiling first); L2/cosine ascending.
        boolean descending = metric == DistanceMetric.INNER_PRODUCT;
        try (QuantizedVectorWriter qWriter = new QuantizedVectorWriter(fieldInfo.getVectorSimilarityFunction(), dimension, docBits)) {
            for (int si = 0; si < numCentroids; si++) {
                int origIdx = spatialOrder[si];
                long startPos = postingsOutput.getFilePointer();
                centroidOffsets[origIdx] = startPos;

                // Single posting per centroid: primary + SOAR merged, sorted by ‖c−v‖,
                // with a position-based SOAR bitset to identify SOAR entries.
                writePostingList(
                    primaryPostings[origIdx],
                    soarPostings[origIdx],
                    vectors,
                    centroids[origIdx],
                    transformedCentroids[origIdx],
                    qWriter,
                    useRotation ? randomRotation : null,
                    descending
                );

                postingSizes[origIdx] = (int) (postingsOutput.getFilePointer() - startPos);
            }
        }
        // Field posting region length (postings are contiguous from postingsFieldOffset), stored so the
        // reader can slice .clap to this field in O(1) instead of summing postingSizes.
        long postingsFieldLength = postingsOutput.getFilePointer() - postingsFieldOffset;

        // 4. Write .clam: meta + centroid stats + posting sizes + centroids + offset table
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(numVectors);
        metaOutput.writeInt(dimension);
        metaOutput.writeInt(numCentroids);
        metaOutput.writeString(metric.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeByte(QuantizerType.SCALAR.id()); // quantizer family (scalar-only today)
        metaOutput.writeLong(postingsFieldOffset);
        metaOutput.writeLong(postingsFieldLength);

        // Centroid doc counts (combined primary + SOAR — the posting's actual size)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeInt(primaryPostings[c].length + soarPostings[c].length);
        }

        // Centroid norms (‖c‖²) now live in .clac alongside the centroid vectors — not in .clam.

        // Posting sizes (exact bytes per centroid — for accurate prefetch)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeInt(postingSizes[c]);
        }

        // Offset table (indexed by original centroid index).
        // Centroids, rotation, and transformed centroids live in .clac (below) — not duplicated here.
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeLong(centroidOffsets[c]);
        }

        // Write centroids to .clac (off-heap, mmap'd at search time). Two per-field .clac offsets
        // are recorded in .clam: the centroid block start and region 1 start; the rotation offset
        // (into .clar) is recorded separately.
        long clacCentroidOffset = centroidsOutput.getFilePointer();
        // Raw region carries the norm only for L2 (ranking norm-trick); transformed region always does.
        boolean rawHasNorm = metric == DistanceMetric.L2;
        CentroidVectorValues.write(centroidsOutput, centroids, transformedCentroids, numCentroids, dimension, rawHasNorm);

        // Rotation is written to its own file .clar (L2 only). rotationOffset == -1 signals
        // "no rotation" (e.g. inner product); otherwise it points at the serialized matrix in .clar.
        long rotationOffset = -1L;
        if (useRotation) {
            rotationOffset = rotationOutput.getFilePointer();
            randomRotation.write(rotationOutput);
        }

        // 5. Write .clac region 1 (appended after centroids): the per-field ordToDoc/doc→ord
        // mapping (OrdToDocDISIReaderConfiguration) plus the ordToCentroid array (folded in from
        // the old .claf). Self-contained/heap-loaded for POC.
        long clacRegion1Offset = writeClacRegion1(vectors, numVectors, numCentroids, primaryPostings);
        metaOutput.writeLong(clacCentroidOffset);
        metaOutput.writeLong(rotationOffset);
        metaOutput.writeLong(clacRegion1Offset);

        log.info(
            "[ClusterANN-WRITE] field={} vectors={} centroids={} dim={} clapSize={}",
            fieldInfo.name,
            numVectors,
            numCentroids,
            dimension,
            postingsOutput.getFilePointer()
        );
    }

    /**
     * Write one posting list per centroid: primary + SOAR ordinals merged into a single run,
     * sorted by centroid-to-vector distance ‖c−v‖ — descending for IP so the highest-ceiling
     * vectors come first, ascending for L2/cosine.
     *
     * <p>Columnar layout: [docIds | ordinals | soarBitset | sortedDistances | quantized blocks].
     * {@code soarBitset} is position-based (bit i set ⟺ ordinals[i] is a SOAR assignment).
     * The reader scans in sort order and uses the stored ‖c−v‖ for early termination; no
     * calibrated λ is stored — the bound is the lossless triangle/Cauchy-Schwarz bound (λ=1).
     */
    private void writePostingList(
        int[] primaryOrdinals,
        int[] soarOrdinals,
        ClusterANNVectorValues vectors,
        float[] originalCentroid,
        float[] transformedCentroid,
        QuantizedVectorWriter qWriter,
        RandomRotation randomRotation,
        boolean descending
    ) throws IOException {
        int pCount = primaryOrdinals.length;
        int sCount = soarOrdinals.length;
        int count = pCount + sCount;

        // Merge primary + SOAR; compute distances and SOAR flags per position.
        // docIds are NOT stored per posting — they come from the per-field ordToDoc mapping.
        // PRODUCTIONIZING TODO: each vector is fetched via vectorValue(ord) twice — here for the
        // distance sort key, and again in writeBlocked for quantization. On the off-heap merge
        // path that doubles vector I/O. Left as-is to keep writer RAM flat (caching would buffer
        // ~clusterSize × dim floats per posting); optimize later if merge I/O becomes a bottleneck.
        int[] ordinals = new int[count];
        float[] dists = new float[count];
        FixedBitSet isSoar = new FixedBitSet(Math.max(count, 1));
        for (int i = 0; i < pCount; i++) {
            int ord = primaryOrdinals[i];
            ordinals[i] = ord;
            dists[i] = (float) Math.sqrt(squaredL2(originalCentroid, vectors.vectorValue(ord)));
        }
        for (int j = 0; j < sCount; j++) {
            int ord = soarOrdinals[j];
            int i = pCount + j;
            ordinals[i] = ord;
            dists[i] = (float) Math.sqrt(squaredL2(originalCentroid, vectors.vectorValue(ord)));
            isSoar.set(i);
        }

        // Sort the whole run by distance, keeping ordinals + SOAR flags in sync
        sortByDistance(dists, ordinals, isSoar, count, descending);

        // Ordinals: fixed-width bulk write. No count prefix — the reader takes the count from
        // centroidDocCounts in .clam (already in heap).
        for (int i = 0; i < count; i++) {
            postingsOutput.writeInt(ordinals[i]);
        }

        // SOAR bitset: position-based, written as raw longs (bit i set ⟺ ordinals[i] is SOAR)
        long[] soarWords = isSoar.getBits();
        postingsOutput.writeVInt(soarWords.length);
        for (long w : soarWords) {
            postingsOutput.writeLong(w);
        }

        // sortedDistances column (parallel to ordinals) — ‖c−v‖ in sort order.
        // NOTE (L2): this equals sqrt(block.add), since for L2 the block's additionalCorrection is
        // ‖v−c‖² and the orthonormal rotation preserves the residual length. The duplication is kept
        // intentionally so the block layout stays metric-uniform (for IP, add is ⟨v,c⟩, unrelated to
        // ‖c−v‖). See QuantizedVectorWriter's block-layout javadoc.
        for (int i = 0; i < count; i++) {
            postingsOutput.writeInt(Float.floatToIntBits(dists[i]));
        }

        // Quantized column — transform vectors before quantizing
        int dim = transformedCentroid.length;
        float[] transformedVec = new float[dim];
        qWriter.writeBlocked(ordinals, count, ord -> {
            float[] vec = vectors.vectorValue(ord);
            if (randomRotation != null) {
                randomRotation.transform(vec, transformedVec);
                return transformedVec;
            }
            return vec;
        }, transformedCentroid, postingsOutput);
    }

    // ========== Lifecycle ==========

    @Override
    public void finish() throws IOException {
        flatVectorsWriter.finish();
        metaOutput.writeInt(END_OF_FIELDS);
        CodecUtil.writeFooter(metaOutput);
        CodecUtil.writeFooter(postingsOutput);
        CodecUtil.writeFooter(centroidsOutput);
        CodecUtil.writeFooter(rotationOutput);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter, metaOutput, postingsOutput, centroidsOutput, rotationOutput);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed();
    }

    // ========== Helpers ==========

    /** Sort centroids by first principal component for spatial locality. */
    private static int[] spatialSort(float[][] centroids, int dimension) {
        int n = centroids.length;
        if (n <= 1) return new int[] { 0 };

        // Compute global mean
        float[] mean = new float[dimension];
        for (float[] c : centroids) {
            for (int d = 0; d < dimension; d++)
                mean[d] += c[d];
        }
        float invN = 1f / n;
        for (int d = 0; d < dimension; d++)
            mean[d] *= invN;

        // Find axis of max variance (farthest centroid from mean)
        float[] axis = new float[dimension];
        float maxDist = 0;
        for (float[] c : centroids) {
            float dist = 0;
            for (int d = 0; d < dimension; d++) {
                float diff = c[d] - mean[d];
                dist += diff * diff;
            }
            if (dist > maxDist) {
                maxDist = dist;
                for (int d = 0; d < dimension; d++)
                    axis[d] = c[d] - mean[d];
            }
        }

        // Project each centroid onto axis, sort by projection
        long[] packed = new long[n];
        for (int i = 0; i < n; i++) {
            float proj = 0;
            for (int d = 0; d < dimension; d++) {
                proj += centroids[i][d] * axis[d];
            }
            packed[i] = ((long) Float.floatToIntBits(proj) << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);

        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = (int) packed[i];
        }
        return order;
    }

    /**
     * Sort {@code dists} and the parallel {@code ordinals} and {@code isSoar} bitset in place by
     * distance. Ascending for L2/cosine, descending for IP ({@code descending == true}).
     * Distances are non-negative, so the raw IEEE-754 bits sort in value order; for descending we
     * invert the key so an ascending unsigned sort yields descending values.
     */
    private static void sortByDistance(float[] dists, int[] ordinals, FixedBitSet isSoar, int count, boolean descending) {
        if (count <= 1) return;
        long[] packed = new long[count];
        for (int i = 0; i < count; i++) {
            long key = Float.floatToIntBits(dists[i]) & 0xFFFFFFFFL;
            if (descending) key = (~key) & 0xFFFFFFFFL;
            packed[i] = (key << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);

        float[] td = new float[count];
        int[] tord = new int[count];
        FixedBitSet tsoar = new FixedBitSet(count);
        for (int i = 0; i < count; i++) {
            int o = (int) packed[i];
            td[i] = dists[o];
            tord[i] = ordinals[o];
            if (isSoar.get(o)) tsoar.set(i);
        }
        System.arraycopy(td, 0, dists, 0, count);
        System.arraycopy(tord, 0, ordinals, 0, count);
        // Overwrite isSoar's backing words with the reordered bitset (same length).
        System.arraycopy(tsoar.getBits(), 0, isSoar.getBits(), 0, isSoar.getBits().length);
    }

    /** Block shift for the ordToDoc DirectMonotonic encoding (64K-value blocks). */
    static final int ORD_TO_DOC_BLOCK_SHIFT = 16;

    /**
     * Write {@code .clac} region 1 (appended after centroid data) and return its start offset.
     * Layout: {@code [numValues vInt][disiMetaLen vInt][disiMeta][disiDataLen vInt][disiData]
     * [ordToCentroid: numValues × short]}.
     *
     * <p>The DISI meta/data are an {@link OrdToDocDISIReaderConfiguration} encoding that provides
     * both ord→doc ({@code getDirectMonotonicReader}) and doc→ord ({@code getIndexedDISI}).
     * {@code ordToCentroid[ord]} is the primary centroid of that ordinal (folded from the old
     * {@code .claf}). Self-contained and heap-loaded on read (POC).
     */
    private long writeClacRegion1(ClusterANNVectorValues vectors, int numVectors, int numCentroids, int[][] primaryPostings)
        throws IOException {
        long offset = centroidsOutput.getFilePointer();

        // doc set for OrdToDocDISI (docIds ascending because ordinals are in docId order)
        DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        for (int ord = 0; ord < numVectors; ord++) {
            docsWithField.add(vectors.ordToDoc(ord));
        }
        int maxDoc = state.segmentInfo.maxDoc();

        ByteBuffersDataOutput metaBuf = new ByteBuffersDataOutput();
        ByteBuffersDataOutput dataBuf = new ByteBuffersDataOutput();
        try (ByteBuffersIndexOutput metaIdx = new ByteBuffersIndexOutput(metaBuf, "clacR1", "meta");
             ByteBuffersIndexOutput dataIdx = new ByteBuffersIndexOutput(dataBuf, "clacR1", "data")) {
            OrdToDocDISIReaderConfiguration.writeStoredMeta(
                ORD_TO_DOC_BLOCK_SHIFT, metaIdx, dataIdx, numVectors, maxDoc, docsWithField);
        }
        byte[] metaBytes = metaBuf.toArrayCopy();
        byte[] dataBytes = dataBuf.toArrayCopy();

        // ordToCentroid (folded from .claf): primary centroid per ordinal
        short[] ordToCentroid = new short[numVectors];
        for (int c = 0; c < numCentroids; c++) {
            for (int ord : primaryPostings[c]) {
                ordToCentroid[ord] = (short) c;
            }
        }

        centroidsOutput.writeVInt(numVectors);
        centroidsOutput.writeVInt(metaBytes.length);
        centroidsOutput.writeBytes(metaBytes, metaBytes.length);
        centroidsOutput.writeVInt(dataBytes.length);
        centroidsOutput.writeBytes(dataBytes, dataBytes.length);
        for (int i = 0; i < numVectors; i++) {
            centroidsOutput.writeShort(ordToCentroid[i]);
        }
        return offset;
    }

    /** Squared L2 distance between two equal-length vectors. */
    private static float squaredL2(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    private static int estimateCentroids(int numVectors) {
        return Math.max(2, Math.min(4096, (numVectors + 256) / 512));
    }

    private int[] collectDocIds(FlatFieldVectorsWriter<float[]> writer, int numVectors) throws IOException {
        if (writer.getDocsWithFieldSet() == null) return null;
        DocIdSetIterator iter = writer.getDocsWithFieldSet().iterator();
        int[] docIds = new int[numVectors];
        for (int i = 0; i < numVectors; i++) {
            docIds[i] = iter.nextDoc();
        }
        return docIds;
    }

    private void writeEmptyMeta(FieldInfo fieldInfo) throws IOException {
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(0);
        metaOutput.writeInt(0);
        metaOutput.writeInt(0);
        metaOutput.writeString(DistanceMetric.L2.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeByte(QuantizerType.SCALAR.id()); // keep header shape identical to writeIVF
        metaOutput.writeLong(0); // postingsOffset
        metaOutput.writeLong(0); // postingsLength
    }

    private IndexOutput createOutput(String extension) throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        IndexOutput output = state.directory.createOutput(fileName, state.context);
        CodecUtil.writeIndexHeader(output, CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
        return output;
    }

    private static byte validateDocBits(int bits) {
        if (bits != 1 && bits != 2 && bits != 4) {
            throw new IllegalArgumentException("docBits must be 1, 2, or 4, got: " + bits);
        }
        return (byte) bits;
    }

    private static DistanceMetric toDistanceMetric(VectorSimilarityFunction simFunc) {
        return switch (simFunc) {
            case EUCLIDEAN -> DistanceMetric.L2;
            case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> DistanceMetric.INNER_PRODUCT;
            case COSINE -> DistanceMetric.COSINE;
        };
    }
}
