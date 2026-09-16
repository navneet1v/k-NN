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
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
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
import org.opensearch.knn.index.clusterann.algorithm.HadamardRotation;

/**
 * Writer for ClusterANN IVF format v2.
 *
 * <p>Two files:
 * <ul>
 *   <li>{@code .clam} — metadata + centroid stats + posting sizes + centroids + offset table</li>
 *   <li>{@code .clap} — per-centroid: [docIds | ordinals | quantized] columnar</li>
 * </ul>
 *
 * <p>Primary + SOAR posting lists are adjacent per centroid for sequential I/O.
 * Centroids written in spatial order (sorted by first principal component).
 */
@Log4j2
public class ClusterANN1040KnnVectorsWriter extends KnnVectorsWriter {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(ClusterANN1040KnnVectorsWriter.class);

    private final SegmentWriteState state;
    private final FlatVectorsWriter flatVectorsWriter;
    private final byte docBits;

    /**
     * Quantizer flow selector. Flow A (0) = per-centroid OSQ (default, unchanged). Flow B (1) =
     * IVFaster-style absolute (Hadamard rotation + Nitrox2 2-bit thermometer scan, existing exact
     * rescore). Overridable at index time via {@code -Dclusterann.quantizer=ivfaster} for A/B
     * benchmarking without a codec API change.
     */
    private final byte quantizerId = resolveQuantizer();

    private static byte resolveQuantizer() {
        String q = System.getProperty("clusterann.quantizer");
        if ("ivfaster".equalsIgnoreCase(q)) return ClusterANNFormatConstants.QUANTIZER_IVFASTER_ABSOLUTE;
        if ("scann".equalsIgnoreCase(q)) return ClusterANNFormatConstants.QUANTIZER_SCANN_RESIDUAL_PQ;
        return ClusterANNFormatConstants.QUANTIZER_PER_CENTROID_OSQ;
    }
    private final List<FieldWriterInfo> fields = new ArrayList<>();

    private final IndexOutput metaOutput;
    private final IndexOutput postingsOutput;
    private final IndexOutput filterOutput;
    private final IndexOutput centroidsOutput;
    private final IndexOutput clipOutput;

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
            filterOutput = createOutput(FILTER_EXTENSION);
            centroidsOutput = createOutput(CENTROIDS_EXTENSION);
            clipOutput = createOutput(ClipPruningData.EXTENSION);
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

        // 1b. Rotation + transformed centroids for quantization.
        // Flow A (per-centroid OSQ): random rotation, EUCLIDEAN only (existing behavior).
        // Flow B (IVFaster absolute): Hadamard FWHT, ALWAYS (the absolute thermometer/int8 scheme
        // requires the 1/sqrt(dim) rotated distribution for every similarity).
        final boolean flowB = quantizerId == QUANTIZER_IVFASTER_ABSOLUTE;
        final boolean flowC = quantizerId == QUANTIZER_SCANN_RESIDUAL_PQ;
        boolean useRotation = flowB || fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN;
        RandomRotation randomRotation = flowB ? null : RandomRotation.create(dimension);
        HadamardRotation hadamard = flowB ? HadamardRotation.create(dimension) : null;
        float[][] transformedCentroids = new float[numCentroids][dimension];
        for (int c = 0; c < numCentroids; c++) {
            if (flowB) {
                hadamard.transform(centroids[c], transformedCentroids[c]);
            } else if (useRotation) {
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

        // Flow C codebook location in the postings file (0/0 for flows A/B).
        long pqCodebookOffset = 0L;
        int pqCodebookLength = 0;

        if (flowB) {
            try (ThermometerVectorWriter tWriter = new ThermometerVectorWriter(dimension)) {
                for (int si = 0; si < numCentroids; si++) {
                    int origIdx = spatialOrder[si];
                    long startPos = postingsOutput.getFilePointer();
                    centroidOffsets[origIdx] = startPos;
                    writePostingListThermometer(primaryPostings[origIdx], vectors, tWriter, hadamard);
                    writePostingListThermometer(soarPostings[origIdx], vectors, tWriter, hadamard);
                    postingSizes[origIdx] = (int) (postingsOutput.getFilePointer() - startPos);
                }
            }
        } else if (flowC) {
            // Flow C = Google ScaNN residual Product Quantization with a SINGLE GLOBAL codebook.
            // Step 1: build residuals (v − its cell centroid) for every vector, mapped by cell.
            // Step 2: train the global PQ codebook on the residuals (ScaNN defaults: K=256, 2 dims/block).
            // Step 3: serialize the codebook into the postings region; record offset/length in meta.
            // Step 4: per posting, encode each vector's residual to PQ codes (numSubspaces bytes).
            int dim = dimension;
            // Collect residuals per cell so encode uses the SAME centroid the vector was assigned to.
            java.util.List<float[]> residualList = new java.util.ArrayList<>(numVectors);
            for (int c = 0; c < numCentroids; c++) {
                for (int pass = 0; pass < 2; pass++) {
                    int[] posting = pass == 0 ? primaryPostings[c] : soarPostings[c];
                    for (int ord : posting) {
                        float[] v = vectors.vectorValue(ord);
                        float[] res = new float[dim];
                        for (int d = 0; d < dim; d++) res[d] = v[d] - centroids[c][d];
                        residualList.add(res);
                    }
                }
            }
            float[][] residuals = residualList.toArray(new float[0][]);
            org.opensearch.knn.index.clusterann.codec.PQCodebook codebook =
                org.opensearch.knn.index.clusterann.codec.PQCodebook.trainDefault(residuals, dim, 42L);
            // Serialize the codebook at the start of the field's posting region.
            pqCodebookOffset = postingsOutput.getFilePointer();
            codebook.write(postingsOutput);
            pqCodebookLength = (int) (postingsOutput.getFilePointer() - pqCodebookOffset);
            // Now write each posting: [docIds | vInt count | ordinals | pqCodes(count·codeBytes)].
            for (int si = 0; si < numCentroids; si++) {
                int origIdx = spatialOrder[si];
                long startPos = postingsOutput.getFilePointer();
                centroidOffsets[origIdx] = startPos;
                writePostingListPQ(primaryPostings[origIdx], vectors, centroids[origIdx], codebook);
                writePostingListPQ(soarPostings[origIdx], vectors, centroids[origIdx], codebook);
                postingSizes[origIdx] = (int) (postingsOutput.getFilePointer() - startPos);
            }
        } else {
            try (QuantizedVectorWriter qWriter = new QuantizedVectorWriter(fieldInfo.getVectorSimilarityFunction(), dimension, docBits)) {
                for (int si = 0; si < numCentroids; si++) {
                    int origIdx = spatialOrder[si];
                    long startPos = postingsOutput.getFilePointer();
                    centroidOffsets[origIdx] = startPos;

                    // Primary posting list (quantize with transformed vectors + transformed centroids)
                    writePostingList(
                        primaryPostings[origIdx],
                        vectors,
                        transformedCentroids[origIdx],
                        qWriter,
                        useRotation ? randomRotation : null
                    );
                    // SOAR posting list (adjacent)
                    writePostingList(
                        soarPostings[origIdx],
                        vectors,
                        transformedCentroids[origIdx],
                        qWriter,
                        useRotation ? randomRotation : null
                    );

                    postingSizes[origIdx] = (int) (postingsOutput.getFilePointer() - startPos);
                }
            }
        }

        // 4. Write .clam: meta + centroid stats + posting sizes + centroids + offset table
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(numVectors);
        metaOutput.writeInt(dimension);
        metaOutput.writeInt(numCentroids);
        metaOutput.writeString(metric.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeByte(quantizerId);
        metaOutput.writeLong(pqCodebookOffset);   // Flow C: global PQ codebook location (0 otherwise)
        metaOutput.writeInt(pqCodebookLength);
        metaOutput.writeLong(postingsFieldOffset);

        // Centroid doc counts (primary posting list sizes)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeInt(primaryPostings[c].length);
        }

        // Centroid norms (||c||² for fast ADC correction)
        for (int c = 0; c < numCentroids; c++) {
            float norm = 0f;
            for (int d = 0; d < dimension; d++)
                norm += centroids[c][d] * centroids[c][d];
            metaOutput.writeInt(Float.floatToIntBits(norm));
        }

        // Posting sizes (exact bytes per centroid — for accurate prefetch)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeInt(postingSizes[c]);
        }

        // Centroids in original order
        for (int c = 0; c < numCentroids; c++) {
            for (int d = 0; d < dimension; d++) {
                metaOutput.writeInt(Float.floatToIntBits(centroids[c][d]));
            }
        }

        // Offset table (indexed by original centroid index)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeLong(centroidOffsets[c]);
        }

        // Rotation blob (for query-time transform). Flow A: the real RandomRotation. Flow B: the
        // query rotation is the Hadamard rebuilt reader-side from dim, so this blob is unused there —
        // write a placeholder RandomRotation only to keep the .clam/.clac byte layout intact.
        RandomRotation rotationBlob = flowB ? RandomRotation.create(dimension) : randomRotation;
        rotationBlob.write(metaOutput);

        // Transformed centroids (for ADC scoring — quantization is in transformed space)
        for (int c = 0; c < numCentroids; c++) {
            for (int d = 0; d < dimension; d++) {
                metaOutput.writeInt(Float.floatToIntBits(transformedCentroids[c][d]));
            }
        }

        // Write centroids to .clac (off-heap, mmap'd at search time)
        OffHeapCentroids.write(centroidsOutput, fieldInfo.number, centroids, transformedCentroids, numCentroids, dimension, (useRotation || flowB) ? rotationBlob : null);

        // 5. Write .claf: centroid assignment per ordinal (for filter-aware search)
        // Format: [fieldNumber:int][numVectors:int][numCentroids:int][assignments: numVectors × short]
        filterOutput.writeInt(fieldInfo.number);
        filterOutput.writeInt(numVectors);
        filterOutput.writeInt(numCentroids);
        short[] ordToCentroid = new short[numVectors];
        for (int c = 0; c < numCentroids; c++) {
            for (int ord : primaryPostings[c]) {
                ordToCentroid[ord] = (short) c;
            }
        }
        byte[] buf = new byte[numVectors * Short.BYTES];
        for (int i = 0; i < numVectors; i++) {
            buf[i * 2] = (byte) (ordToCentroid[i] >> 8);
            buf[i * 2 + 1] = (byte) ordToCentroid[i];
        }
        filterOutput.writeBytes(buf, buf.length);

        log.info(
            "[ClusterANN-WRITE] field={} vectors={} centroids={} dim={} clapSize={}",
            fieldInfo.name,
            numVectors,
            numCentroids,
            dimension,
            postingsOutput.getFilePointer()
        );

        // 6. CLIP pruning DISABLED. The reader's readClipData() returns an empty map and never opens
        // the .clid file, so calibration is pure wasted work — and ClipPruningData.calibrate is O(N^2)
        // at scale (per-cluster sample cap scales with totalVectors), which stalls flush at ~1M docs.
        // Skip calibrate() and leave .clid empty (created, no content). Re-enable by restoring the
        // calibrate+write block AND the real reader (readClipDataDISABLED).
        // ClipPruningData clipData = ClipPruningData.calibrate(centroids, primaryPostings, allVectors, dimension, 42L);
        // clipOutput.writeInt(fieldInfo.number);
        // clipData.write(clipOutput);
        log.info("[ClusterANN-WRITE] field={} CLIP calibration SKIPPED (disabled)", fieldInfo.name);
    }

    /**
     * Flow B posting writer: [docIds | ordinals | thermometer+int8 blocks]. Rotates each vector
     * with the Hadamard rotation (absolute — no centroid). UNVERIFIED (written blind).
     */
    private void writePostingListThermometer(
        int[] ordinals,
        ClusterANNVectorValues vectors,
        ThermometerVectorWriter tWriter,
        HadamardRotation hadamard
    ) throws IOException {
        int count = ordinals.length;
        int[] docIds = new int[count];
        for (int i = 0; i < count; i++) {
            docIds[i] = vectors.ordToDoc(ordinals[i]);
        }
        sortParallel(docIds, ordinals, count);
        PostingListCodec.write(docIds, postingsOutput);
        postingsOutput.writeVInt(count);
        for (int i = 0; i < count; i++) {
            postingsOutput.writeInt(ordinals[i]);
        }
        int dim = vectors.dimension();
        float[] rotated = new float[dim];
        tWriter.writeBlocked(ordinals, count, ord -> {
            float[] vec = vectors.vectorValue(ord);
            hadamard.transform(vec, rotated);
            return rotated;
        }, postingsOutput);
    }

    /**
     * Flow C posting writer: [docIds | vInt count | ordinals | pqCodes(count·codeBytes)].
     * Each vector's PQ code is over its RESIDUAL (v − this cell's centroid), using the global codebook.
     */
    private void writePostingListPQ(
        int[] ordinals,
        ClusterANNVectorValues vectors,
        float[] centroid,
        org.opensearch.knn.index.clusterann.codec.PQCodebook codebook
    ) throws IOException {
        int count = ordinals.length;
        int[] docIds = new int[count];
        for (int i = 0; i < count; i++) {
            docIds[i] = vectors.ordToDoc(ordinals[i]);
        }
        sortParallel(docIds, ordinals, count);
        PostingListCodec.write(docIds, postingsOutput);
        postingsOutput.writeVInt(count);
        for (int i = 0; i < count; i++) {
            postingsOutput.writeInt(ordinals[i]);
        }
        int dim = vectors.dimension();
        int codeBytes = codebook.codeBytes();
        float[] res = new float[dim];
        byte[] code = new byte[codeBytes];
        for (int i = 0; i < count; i++) {
            float[] vec = vectors.vectorValue(ordinals[i]);
            for (int d = 0; d < dim; d++) res[d] = vec[d] - centroid[d];
            codebook.encode(res, code, 0);
            postingsOutput.writeBytes(code, 0, codeBytes);
        }
    }

    /**
     * Write one posting list: [docIds | ordinals (fixed-width) | quantized blocks] columnar.
     */
    private void writePostingList(
        int[] ordinals,
        ClusterANNVectorValues vectors,
        float[] centroid,
        QuantizedVectorWriter qWriter,
        RandomRotation randomRotation
    ) throws IOException {
        int count = ordinals.length;

        // Convert ordinals to docIds
        int[] docIds = new int[count];
        for (int i = 0; i < count; i++) {
            docIds[i] = vectors.ordToDoc(ordinals[i]);
        }

        // Sort by docId, keep ordinals in sync
        sortParallel(docIds, ordinals, count);

        // Write columns
        PostingListCodec.write(docIds, postingsOutput);

        // Ordinals: fixed-width bulk write
        postingsOutput.writeVInt(count);
        for (int i = 0; i < count; i++) {
            postingsOutput.writeInt(ordinals[i]);
        }

        // Quantized column — transform vectors before quantizing
        int dim = centroid.length;
        float[] transformedVec = new float[dim];
        qWriter.writeBlocked(ordinals, count, ord -> {
            float[] vec = vectors.vectorValue(ord);
            if (randomRotation != null) {
                randomRotation.transform(vec, transformedVec);
                return transformedVec;
            }
            return vec;
        }, centroid, postingsOutput);
    }

    // ========== Lifecycle ==========

    @Override
    public void finish() throws IOException {
        flatVectorsWriter.finish();
        metaOutput.writeInt(END_OF_FIELDS);
        CodecUtil.writeFooter(metaOutput);
        CodecUtil.writeFooter(postingsOutput);
        CodecUtil.writeFooter(filterOutput);
        CodecUtil.writeFooter(centroidsOutput);
        CodecUtil.writeFooter(clipOutput);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter, metaOutput, postingsOutput, filterOutput, centroidsOutput, clipOutput);
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

    /** Sort two parallel arrays by the first (keys). */
    private static void sortParallel(int[] keys, int[] vals, int count) {
        if (count <= 1) return;
        long[] packed = new long[count];
        for (int i = 0; i < count; i++) {
            packed[i] = ((long) keys[i] << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        int[] tk = new int[count];
        int[] tv = new int[count];
        for (int i = 0; i < count; i++) {
            int origIdx = (int) packed[i];
            tk[i] = keys[origIdx];
            tv[i] = vals[origIdx];
        }
        System.arraycopy(tk, 0, keys, 0, count);
        System.arraycopy(tv, 0, vals, 0, count);
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
        metaOutput.writeByte(quantizerId);
        metaOutput.writeLong(0);   // pqCodebookOffset
        metaOutput.writeInt(0);    // pqCodebookLength
        metaOutput.writeLong(0);   // postingsFieldOffset
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
