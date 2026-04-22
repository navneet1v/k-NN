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
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.IVFIndex;
import org.opensearch.knn.index.clusterann.ClusterANNVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Writer for the ClusterANN IVF format. Builds a cluster-based inverted file index
 * with SOAR secondary assignments during segment flush and merge.
 *
 * <p>File layout:
 * <ul>
 *   <li>{@code .clam} — metadata: per-field config, centroid count, offsets</li>
 *   <li>{@code .clac} — centroids: flat float arrays</li>
 *   <li>{@code .clap} — posting lists: primary + SOAR, size-prefixed per centroid</li>
 * </ul>
 *
 * <p>Raw vectors are stored separately by the delegate {@link FlatVectorsWriter} for
 * exact rescoring during the second phase of search.
 */
@Log4j2
public class ClusterANN1040KnnVectorsWriter extends KnnVectorsWriter {

    static final String META_EXTENSION = "clam";
    static final String CENTROIDS_EXTENSION = "clac";
    static final String POSTINGS_EXTENSION = "clap";
    static final String QUANTIZED_EXTENSION = "claq";
    static final String CODEC_NAME = "ClusterANN1040";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;
    static final int END_OF_FIELDS = -1;

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(ClusterANN1040KnnVectorsWriter.class);

    private final SegmentWriteState state;
    private final FlatVectorsWriter flatVectorsWriter;
    private final byte docBits;
    private final List<FieldWriterInfo> fields = new ArrayList<>();

    private final IndexOutput metaOutput;
    private final IndexOutput centroidsOutput;
    private final IndexOutput postingsOutput;
    private final IndexOutput quantizedOutput;

    /**
     * Per-field state: holds reference to the flat writer for zero-copy vector access at flush time.
     */
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
            centroidsOutput = createOutput(CENTROIDS_EXTENSION);
            postingsOutput = createOutput(POSTINGS_EXTENSION);
            quantizedOutput = createOutput(QUANTIZED_EXTENSION);
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

        FieldWriterInfo info = new FieldWriterInfo(fieldInfo, flatFieldWriter);
        fields.add(info);

        return flatFieldWriter;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        for (FieldWriterInfo info : fields) {
            writeField(info);
        }
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Delegate raw vector merge
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        // Collect merged vectors and rebuild IVF
        FloatVectorValues mergedValues = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        if (mergedValues == null) return;

        int dimension = fieldInfo.getVectorDimension();

        // Write vectors to temp file for off-heap access during clustering
        IndexOutput tempOut = state.directory.createTempOutput(state.segmentInfo.name, "clann_merge", state.context);
        List<Integer> docIdList = new ArrayList<>();
        int count = 0;
        try {
            var iterator = mergedValues.iterator();
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                float[] vec = mergedValues.vectorValue(iterator.index());
                for (int d = 0; d < dimension; d++) {
                    tempOut.writeInt(Float.floatToIntBits(vec[d]));
                }
                docIdList.add(doc);
                count++;
            }
        } finally {
            tempOut.close();
        }

        if (count > 0) {
            IndexInput tempIn = state.directory.openInput(tempOut.getName(), state.context);
            try {
                int[] docIds = docIdList.stream().mapToInt(Integer::intValue).toArray();
                // Off-heap: cluster directly from disk
                ClusterANNVectorValues vectorValues = ClusterANNVectorValues.fromIndexInput(tempIn, docIds, count, dimension);
                writeMergedField(fieldInfo, vectorValues, count, dimension);
            } finally {
                tempIn.close();
                state.directory.deleteFile(tempOut.getName());
            }
        }
    }

    @Override
    public void finish() throws IOException {
        flatVectorsWriter.finish();
        metaOutput.writeInt(END_OF_FIELDS);
        CodecUtil.writeFooter(metaOutput);
        CodecUtil.writeFooter(centroidsOutput);
        CodecUtil.writeFooter(postingsOutput);
        CodecUtil.writeFooter(quantizedOutput);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter, metaOutput, centroidsOutput, postingsOutput, quantizedOutput);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed();
    }

    // ========== Private: Index Building & Serialization ==========

    /** Below this threshold, store as flat (single centroid) — clustering overhead not worth it. */
    private static final int FLAT_VECTOR_THRESHOLD = 128;

    private void writeField(FieldWriterInfo info) throws IOException {
        List<float[]> vectors = info.flatFieldWriter.getVectors();
        int numVectors = vectors.size();
        int dimension = info.fieldInfo.getVectorDimension();

        if (numVectors == 0) {
            writeEmptyFieldMeta(info.fieldInfo);
            return;
        }

        // Build doc ID mapping if there are deleted docs
        int[] docIds = null;
        if (info.flatFieldWriter.getDocsWithFieldSet() != null) {
            DocIdSetIterator iter = info.flatFieldWriter.getDocsWithFieldSet().iterator();
            docIds = new int[numVectors];
            for (int i = 0; i < numVectors; i++) {
                docIds[i] = iter.nextDoc();
            }
        }

        ClusterANNVectorValues vectorValues = ClusterANNVectorValues.fromList(vectors, docIds, dimension);
        DistanceMetric metric = toDistanceMetric(info.fieldInfo.getVectorSimilarityFunction());

        // Skip clustering for tiny segments — single centroid, all vectors in one posting list
        int numCentroids = numVectors <= FLAT_VECTOR_THRESHOLD ? 1 : estimateCentroids(numVectors);

        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(numCentroids)
            .targetClusterSize(512)
            .metric(metric)
            .soarLambda(1.0f)
            .seed(42L)
            .parallel(true)
            .build();

        IVFIndex index = IVFIndex.build(vectorValues, config);

        // Serialize
        long centroidsOffset = centroidsOutput.getFilePointer();
        long postingsOffset = postingsOutput.getFilePointer();
        long quantizedOffset = quantizedOutput.getFilePointer();

        writeCentroids(index);
        writePostings(index);
        writeQuantizedVectors(info.fieldInfo, i -> info.flatFieldWriter.getVectors().get(i).clone(), vectors.size(), index, dimension);
        writeFieldMeta(
            info.fieldInfo,
            numVectors,
            dimension,
            index.numCentroids(),
            metric,
            centroidsOffset,
            postingsOffset,
            quantizedOffset
        );

        log.info(
            "[ClusterANN] wrote field={} vectors={} centroids={} dim={}",
            info.fieldInfo.name,
            numVectors,
            index.numCentroids(),
            dimension
        );
    }

    /**
     * Write a merged field using off-heap ClusterANNVectorValues (from IndexInput).
     * For merge-time, vectors need to be read into memory for quantization.
     */
    private void writeMergedField(FieldInfo fieldInfo, ClusterANNVectorValues vectorValues, int numVectors, int dimension)
        throws IOException {
        if (numVectors == 0) {
            writeEmptyFieldMeta(fieldInfo);
            return;
        }

        DistanceMetric metric = toDistanceMetric(fieldInfo.getVectorSimilarityFunction());

        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(estimateCentroids(numVectors))
            .targetClusterSize(512)
            .metric(metric)
            .soarLambda(1.0f)
            .seed(42L)
            .parallel(true)
            .build();

        IVFIndex index = IVFIndex.build(vectorValues, config);

        long centroidsOffset = centroidsOutput.getFilePointer();
        long postingsOffset = postingsOutput.getFilePointer();
        long quantizedOffset = quantizedOutput.getFilePointer();

        writeCentroids(index);
        writePostings(index);
        writeQuantizedVectors(fieldInfo, i -> vectorValues.vectorValueCopy(i), numVectors, index, dimension);
        writeFieldMeta(fieldInfo, numVectors, dimension, index.numCentroids(), metric, centroidsOffset, postingsOffset, quantizedOffset);

        log.info(
            "[ClusterANN] merged field={} vectors={} centroids={} dim={}",
            fieldInfo.name,
            numVectors,
            index.numCentroids(),
            dimension
        );
    }

    private void writeCentroids(IVFIndex index) throws IOException {
        float[][] centroids = index.centroids();
        for (float[] centroid : centroids) {
            for (float v : centroid) {
                centroidsOutput.writeInt(Float.floatToIntBits(v));
            }
        }
    }

    private void writePostings(IVFIndex index) throws IOException {
        int numCentroids = index.numCentroids();

        long[] primaryOffsets = new long[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            primaryOffsets[c] = postingsOutput.getFilePointer();
            PostingListCodec.write(index.primaryPostings()[c], postingsOutput);
        }

        long[] soarOffsets = new long[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            soarOffsets[c] = postingsOutput.getFilePointer();
            PostingListCodec.write(index.soarPostings()[c], postingsOutput);
        }

        // Write offset table at the end of postings (for direct seek + prefetch)
        long offsetTablePos = postingsOutput.getFilePointer();
        for (int c = 0; c < numCentroids; c++) {
            postingsOutput.writeLong(primaryOffsets[c]);
        }
        for (int c = 0; c < numCentroids; c++) {
            postingsOutput.writeLong(soarOffsets[c]);
        }
        // Write offset table position as the very last long (reader reads this first)
        postingsOutput.writeLong(offsetTablePos);
    }

    /**
     * Quantize each vector relative to its assigned centroid using Lucene's OptimizedScalarQuantizer.
     * Writes packed codes + 4 correction factors per vector in ordinal order.
     *
     * <p>Format per vector:
     * <ul>
     *   <li>packed codes: byte[packedBytesPerVector]</li>
     *   <li>lowerInterval: float (as int bits)</li>
     *   <li>upperInterval: float (as int bits)</li>
     *   <li>additionalCorrection: float (as int bits)</li>
     *   <li>quantizedComponentSum: int</li>
     * </ul>
     */
    @FunctionalInterface
    private interface VectorSupplier {
        float[] get(int ordinal) throws IOException;
    }

    private void writeQuantizedVectors(FieldInfo fieldInfo, VectorSupplier vectors, int numVectors, IVFIndex index, int dimension)
        throws IOException {
        VectorSimilarityFunction simFunc = fieldInfo.getVectorSimilarityFunction();
        OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(simFunc);

        int[] assignments = new int[numVectors];
        int numCentroids = index.numCentroids();
        for (int c = 0; c < numCentroids; c++) {
            for (int ord : index.primaryPostings()[c]) {
                assignments[ord] = c;
            }
        }

        float[][] centroids = index.centroids();
        byte[] scratch = new byte[dimension];
        byte[] bitsArray = new byte[] { docBits };

        // Pre-normalize centroids for cosine (once per centroid, not per vector)
        float[][] normalizedCentroids = centroids;
        if (simFunc == VectorSimilarityFunction.COSINE) {
            normalizedCentroids = new float[centroids.length][];
            for (int c = 0; c < centroids.length; c++) {
                normalizedCentroids[c] = centroids[c].clone();
                float cNorm = 0f;
                for (float v : normalizedCentroids[c])
                    cNorm += v * v;
                cNorm = (float) Math.sqrt(cNorm);
                if (cNorm > 0f) {
                    for (int d = 0; d < dimension; d++)
                        normalizedCentroids[c][d] /= cNorm;
                }
            }
        }

        for (int i = 0; i < numVectors; i++) {
            float[] vector = vectors.get(i);

            // For cosine, normalize the vector (Lucene OSQ expects normalized input for cosine)
            if (simFunc == VectorSimilarityFunction.COSINE) {
                float norm = 0f;
                for (float v : vector)
                    norm += v * v;
                norm = (float) Math.sqrt(norm);
                if (norm > 0f) {
                    for (int d = 0; d < dimension; d++)
                        vector[d] /= norm;
                }
            }

            float[] centroid = normalizedCentroids[assignments[i]];

            // Quantize relative to centroid
            byte[][] destinations = new byte[][] { scratch };
            OptimizedScalarQuantizer.QuantizationResult[] results = osq.multiScalarQuantize(vector, destinations, bitsArray, centroid);

            // Pack codes based on bit width
            int packedBytes = packedBytesPerVector(dimension, docBits);
            byte[] packed = new byte[packedBytes];
            packQuantizedCodes(scratch, packed, dimension, docBits);

            // Write packed codes + corrections
            quantizedOutput.writeBytes(packed, 0, packed.length);
            quantizedOutput.writeInt(Float.floatToIntBits(results[0].lowerInterval()));
            quantizedOutput.writeInt(Float.floatToIntBits(results[0].upperInterval()));
            quantizedOutput.writeInt(Float.floatToIntBits(results[0].additionalCorrection()));
            quantizedOutput.writeInt(results[0].quantizedComponentSum());
        }
    }

    /** Pack quantized codes based on bit width. */
    static void packQuantizedCodes(byte[] raw, byte[] packed, int dimension, byte bits) {
        if (bits == 1) {
            packAsBinary(raw, packed, dimension);
        } else if (bits == 2) {
            transposeDibit(raw, packed, dimension);
        } else {
            // 4-bit: transpose into 4 nibble stripes (same as transposeHalfByte)
            transposeHalfByte(raw, packed, dimension);
        }
    }

    /** Pack 1-bit values MSB-first into bytes. */
    static void packAsBinary(byte[] vector, byte[] packed, int dimension) {
        for (int i = 0; i < dimension;) {
            byte result = 0;
            for (int j = 7; j >= 0 && i < dimension; j--) {
                result |= (byte) ((vector[i] & 1) << j);
                ++i;
            }
            packed[((i + 7) / 8) - 1] = result;
        }
    }

    /** Transpose 2-bit values into 2 stripes (lower bits, upper bits) MSB-first. */
    static void transposeDibit(byte[] vector, byte[] packed, int dimension) {
        int stripeSize = packed.length / 2;
        int i = 0, index = 0;
        int limit = dimension - 7;
        for (; i < limit; i += 8, index++) {
            int lower = 0, upper = 0;
            for (int j = 7; j >= 0; j--) {
                lower |= (vector[i + (7 - j)] & 1) << j;
                upper |= ((vector[i + (7 - j)] >> 1) & 1) << j;
            }
            packed[index] = (byte) lower;
            packed[index + stripeSize] = (byte) upper;
        }
        if (i < dimension) {
            int lower = 0, upper = 0;
            for (int j = 7; i < dimension; j--, i++) {
                lower |= (vector[i] & 1) << j;
                upper |= ((vector[i] >> 1) & 1) << j;
            }
            packed[index] = (byte) lower;
            packed[index + stripeSize] = (byte) upper;
        }
    }

    /** Transpose 4-bit values into 4 nibble stripes for SIMD-friendly dot product. */
    static void transposeHalfByte(byte[] input, byte[] output, int dimension) {
        int stripeSize = (dimension + 7) / 8;
        for (int i = 0; i < dimension; i++) {
            int val = input[i] & 0x0F;
            int byteIdx = i / 8;
            int bitIdx = 7 - (i % 8);
            if ((val & 1) != 0) output[byteIdx] |= (byte) (1 << bitIdx);
            if ((val & 2) != 0) output[stripeSize + byteIdx] |= (byte) (1 << bitIdx);
            if ((val & 4) != 0) output[2 * stripeSize + byteIdx] |= (byte) (1 << bitIdx);
            if ((val & 8) != 0) output[3 * stripeSize + byteIdx] |= (byte) (1 << bitIdx);
        }
    }

    /** Compute packed bytes per vector for a given bit width. */
    static int packedBytesPerVector(int dimension, int bits) {
        if (bits == 1) {
            return (dimension + 7) / 8;
        } else if (bits == 2) {
            return ((dimension + 7) / 8) * 2;  // 2 stripes
        } else {
            // 4-bit: 4 stripes
            return ((dimension + 7) / 8) * 4;
        }
    }

    /** Validate docBits is a supported value. */
    private static byte validateDocBits(int bits) {
        if (bits != 1 && bits != 2 && bits != 4) {
            throw new IllegalArgumentException("docBits must be 1, 2, or 4, got: " + bits);
        }
        return (byte) bits;
    }

    private void writeFieldMeta(
        FieldInfo fieldInfo,
        int numVectors,
        int dimension,
        int numCentroids,
        DistanceMetric metric,
        long centroidsOffset,
        long postingsOffset,
        long quantizedOffset
    ) throws IOException {
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(numVectors);
        metaOutput.writeInt(dimension);
        metaOutput.writeInt(numCentroids);
        metaOutput.writeString(metric.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeLong(centroidsOffset);
        metaOutput.writeLong(postingsOffset);
        metaOutput.writeLong(quantizedOffset);
    }

    private void writeEmptyFieldMeta(FieldInfo fieldInfo) throws IOException {
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(0); // numVectors = 0 signals empty
        metaOutput.writeInt(0);
        metaOutput.writeInt(0);
        metaOutput.writeString(DistanceMetric.L2.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeLong(0);
        metaOutput.writeLong(0);
        metaOutput.writeLong(0);
    }

    // ========== Helpers ==========

    private IndexOutput createOutput(String extension) throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        IndexOutput output = state.directory.createOutput(fileName, state.context);
        CodecUtil.writeIndexHeader(output, CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
        return output;
    }

    /**
     * Estimate number of centroids based on dataset size.
     * Formula: clamp((n + 256) / 512, 2, 4096)
     */
    private static int estimateCentroids(int numVectors) {
        return Math.max(2, Math.min(4096, (numVectors + 256) / 512));
    }

    private static DistanceMetric toDistanceMetric(VectorSimilarityFunction simFunc) {
        switch (simFunc) {
            case EUCLIDEAN:
                return DistanceMetric.L2;
            case DOT_PRODUCT:
            case MAXIMUM_INNER_PRODUCT:
                return DistanceMetric.INNER_PRODUCT;
            case COSINE:
                return DistanceMetric.COSINE;
            default:
                return DistanceMetric.L2;
        }
    }
}
