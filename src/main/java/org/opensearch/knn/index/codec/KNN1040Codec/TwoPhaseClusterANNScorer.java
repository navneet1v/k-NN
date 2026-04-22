/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;
import java.util.Arrays;

/**
 * Two-phase scorer: ADC (Asymmetric Distance Computation) first pass using 1-bit quantized
 * vectors from .claq, then exact rescore of top candidates via Lucene's RandomVectorScorer.
 *
 * <p>Phase 1 (ADC): For each candidate, reads packed 1-bit codes + correction factors from
 * the quantized file and computes an approximate score using the BBQ formula. Keeps top
 * {@code k * oversampleFactor} candidates.
 *
 * <p>Phase 2 (Rescore): Scores the top candidates with full-precision vectors and collects
 * final results.
 *
 * <p>This matches the scoring approach from Lucene's BinaryQuantizedVectors but adapted
 * for IVF posting list access patterns.
 */
final class TwoPhaseClusterANNScorer {

    private static final float RESCORE_OVERSAMPLE = 2.0f;
    private static final int ADC_MULTIPLIER = 20;
    private static final byte QUERY_BITS = 4;
    private static final float FOUR_BIT_SCALE = 1.0f / ((1 << QUERY_BITS) - 1);

    private final RandomVectorScorer exactScorer;
    private final IndexInput quantizedInput;
    private final ClusterANNFieldState fieldState;
    private final VectorSimilarityFunction simFunc;
    private final float[] queryVector;
    private final int k;
    private final ScalarBitEncoding encoding;
    private final int recordSize;

    // Native SIMD: mmap address of .claq for C++ bulk scoring
    private final long[] quantizedAddressAndSize;
    private final boolean useNativeScoring;

    // Candidate buffer for ADC results — bounded min-heap (evicts worst when full)
    private int[] candidateOrdinals;
    private float[] candidateScores;
    private int candidateCount;
    private final int maxCandidates;

    // Temp buffer for native bulk scoring
    private int[] nativeBatchOrds;
    private float[] nativeBatchScores;

    // Reusable scratch for quantization and per-vector reads
    private final QueryQuantizationState qState;
    private final byte[] packedBuffer;
    private final byte[] transposedBuffer;

    /** Reusable scratch arrays for per-centroid query quantization. */
    private static final class QueryQuantizationState {
        final OptimizedScalarQuantizer osq;
        final byte[] scratch;
        final byte[][] destinations;
        final byte[] bitsArray;
        final float[] queryCopy;

        QueryQuantizationState(VectorSimilarityFunction simFunc, int dimension) {
            this.osq = new OptimizedScalarQuantizer(simFunc);
            this.scratch = new byte[dimension];
            this.destinations = new byte[][] { scratch };
            this.bitsArray = new byte[] { QUERY_BITS };
            this.queryCopy = new float[dimension];
        }
    }

    TwoPhaseClusterANNScorer(
        RandomVectorScorer exactScorer,
        IndexInput quantizedInput,
        ClusterANNFieldState fieldState,
        VectorSimilarityFunction simFunc,
        float[] queryVector,
        int k
    ) {
        this.exactScorer = exactScorer;
        this.quantizedInput = quantizedInput;
        this.fieldState = fieldState;
        this.simFunc = simFunc;
        this.queryVector = queryVector;
        this.k = k;
        this.encoding = ScalarBitEncoding.fromDocBits(fieldState.docBits);
        this.recordSize = encoding.recordBytes(fieldState.dimension);
        int maxCandidates = k * ADC_MULTIPLIER;
        this.maxCandidates = maxCandidates;
        this.candidateOrdinals = new int[maxCandidates];
        this.candidateScores = new float[maxCandidates];
        this.candidateCount = 0;
        this.nativeBatchOrds = new int[256];
        this.nativeBatchScores = new float[256];
        this.packedBuffer = new byte[encoding.docPackedBytes(fieldState.dimension)];
        this.transposedBuffer = new byte[((fieldState.dimension + 7) / 8) * 4];
        this.qState = new QueryQuantizationState(simFunc, fieldState.dimension);

        // Try to extract mmap address for native SIMD scoring
        long totalQuantizedBytes = (long) fieldState.numVectors * recordSize;
        this.quantizedAddressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
            quantizedInput,
            fieldState.quantizedOffset,
            totalQuantizedBytes
        );
        this.useNativeScoring = quantizedAddressAndSize != null && fieldState.docBits == 1;
    }

    public int ordToDoc(int ordinal) {
        return exactScorer.ordToDoc(ordinal);
    }

    /**
     * Score a candidate using ADC against a specific centroid's quantized query.
     *
     * @param ordinal          vector ordinal
     * @param queryTransposed  transposed 4-bit quantized query
     * @param queryLower               query lower interval
     * @param queryScale               query (upper - lower) * FOUR_BIT_SCALE
     * @param queryComponentSum query quantized component sum
     * @param queryAdditionalCorrection query additional correction
     */
    void scoreADC(
        int ordinal,
        byte[] queryTransposed,
        float queryLower,
        float queryScale,
        float queryComponentSum,
        float queryAdditionalCorrection
    ) throws IOException {
        long qOffset = fieldState.quantizedOffset + (long) ordinal * recordSize;
        quantizedInput.seek(qOffset);

        // #4 fix: reuse buffer
        quantizedInput.readBytes(packedBuffer, 0, encoding.docPackedBytes(fieldState.dimension));
        float docLower = Float.intBitsToFloat(quantizedInput.readInt());
        float docUpper = Float.intBitsToFloat(quantizedInput.readInt());
        float docAdditionalCorrection = Float.intBitsToFloat(quantizedInput.readInt());
        float docComponentSum = (float) quantizedInput.readInt();
        float docScale = (docUpper - docLower) * encoding.docBitScale();

        // Dot product based on doc bit width — delegates to Lucene SIMD (Panama on capable JVMs)
        long rawDot;
        if (fieldState.docBits == 1) {
            rawDot = VectorUtil.int4BitDotProduct(queryTransposed, packedBuffer);
        } else if (fieldState.docBits == 2) {
            rawDot = VectorUtil.int4DibitDotProduct(queryTransposed, packedBuffer);
        } else {
            rawDot = int4NibbleDotProduct(queryTransposed, packedBuffer);
        }

        // BBQ formula
        float score = docLower * queryLower * fieldState.dimension + queryLower * docScale * docComponentSum + docLower * queryScale
            * queryComponentSum + docScale * queryScale * (float) rawDot;

        float adcSimilarity;
        if (simFunc == VectorSimilarityFunction.EUCLIDEAN) {
            score = queryAdditionalCorrection + docAdditionalCorrection - 2 * score;
            adcSimilarity = Math.max(1.0f / (1.0f + score), 0);
        } else {
            // DOT_PRODUCT / COSINE / MAXIMUM_INNER_PRODUCT
            score += queryAdditionalCorrection + docAdditionalCorrection;
            adcSimilarity = Math.max((1.0f + score) / 2.0f, 0);
        }

        if (candidateCount < maxCandidates) {
            candidateOrdinals[candidateCount] = ordinal;
            candidateScores[candidateCount] = adcSimilarity;
            candidateCount++;
            if (candidateCount == maxCandidates) {
                buildMinHeap();
            }
        } else if (adcSimilarity > candidateScores[0]) {
            // Evict worst (heap root) and sift down
            candidateOrdinals[0] = ordinal;
            candidateScores[0] = adcSimilarity;
            siftDown(0);
        }
    }

    /**
     * Batch ADC scoring for a posting list. When native SIMD is available, uses
     * {@link SimdVectorComputeService} for AVX512/NEON-accelerated bulk scoring.
     * Falls back to per-ordinal Java scoring otherwise.
     *
     * @param ordinals     vector ordinals to score
     * @param count        number of valid ordinals
     * @param centroidIdx  centroid index for query quantization
     * @param centroid     centroid vector
     * @param centroidDp   dot product of query with centroid (for IP/cosine)
     */
    void scoreADCBatch(int[] ordinals, int count, int centroidIdx, float[] centroid, float centroidDp) throws IOException {
        if (count == 0) return;

        QueryQuantization qQuant = quantizeQuery(centroid);

        if (useNativeScoring) {
            // Native path: C++ AVX512/NEON bulk scoring via JNI
            if (nativeBatchOrds.length < count) {
                nativeBatchOrds = new int[count];
                nativeBatchScores = new float[count];
            }
            System.arraycopy(ordinals, 0, nativeBatchOrds, 0, count);

            SimdVectorComputeService.saveSQSearchContext(
                qQuant.transposed,
                qQuant.queryLower,
                qQuant.queryLower + qQuant.queryScale / FOUR_BIT_SCALE, // reconstruct upper
                qQuant.additionalCorrection,
                (int) qQuant.queryComponentSum,
                quantizedAddressAndSize,
                simFunc == VectorSimilarityFunction.EUCLIDEAN
                    ? SimdVectorComputeService.SimilarityFunctionType.SQ_L2.ordinal()
                    : SimdVectorComputeService.SimilarityFunctionType.SQ_IP.ordinal(),
                fieldState.dimension,
                centroidDp
            );

            SimdVectorComputeService.scoreSimilarityInBulk(nativeBatchOrds, nativeBatchScores, count);

            for (int i = 0; i < count; i++) {
                addCandidate(ordinals[i], nativeBatchScores[i]);
            }
        } else {
            // Java fallback: per-ordinal scoring
            for (int i = 0; i < count; i++) {
                scoreADC(
                    ordinals[i],
                    qQuant.transposed,
                    qQuant.queryLower,
                    qQuant.queryScale,
                    qQuant.queryComponentSum,
                    qQuant.additionalCorrection
                );
            }
        }
    }

    /** Similarity function used by this scorer. */
    VectorSimilarityFunction getSimFunc() {
        return simFunc;
    }

    /** Whether native SIMD scoring via mmap is active. */
    boolean isNativeScoring() {
        return useNativeScoring;
    }

    /**
     * Phase 2: rescore top ADC candidates with exact scorer and collect results.
     */
    public void finish(KnnCollector collector) throws IOException {
        int rescoreCount = Math.min((int) (k * RESCORE_OVERSAMPLE), candidateCount);
        if (rescoreCount == 0) return;

        // Partial select: find top rescoreCount by ADC score using min-heap selection
        // For small counts, just sort the indices directly with primitive array
        int[] idx = new int[candidateCount];
        for (int i = 0; i < candidateCount; i++)
            idx[i] = i;

        // Partial sort: move top rescoreCount to front via selection
        for (int i = 0; i < rescoreCount; i++) {
            int bestIdx = i;
            for (int j = i + 1; j < candidateCount; j++) {
                if (candidateScores[idx[j]] > candidateScores[idx[bestIdx]]) {
                    bestIdx = j;
                }
            }
            int tmp = idx[i];
            idx[i] = idx[bestIdx];
            idx[bestIdx] = tmp;
        }

        for (int i = 0; i < rescoreCount; i++) {
            int ord = candidateOrdinals[idx[i]];
            int doc = exactScorer.ordToDoc(ord);
            float exactScore = exactScorer.score(ord);
            collector.collect(doc, exactScore);
        }
    }

    /**
     * Quantize the query vector relative to a centroid for ADC scoring.
     * Returns the transposed query + correction factors.
     */
    QueryQuantization quantizeQuery(float[] centroid) {
        Arrays.fill(qState.scratch, (byte) 0);

        System.arraycopy(queryVector, 0, qState.queryCopy, 0, queryVector.length);
        float[] queryCopy = qState.queryCopy;
        OptimizedScalarQuantizer.QuantizationResult result = qState.osq.multiScalarQuantize(
            queryCopy,
            qState.destinations,
            qState.bitsArray,
            centroid
        )[0];

        Arrays.fill(transposedBuffer, (byte) 0);
        transposeHalfByte(qState.scratch, transposedBuffer);

        float queryLower = result.lowerInterval();
        float queryScale = (result.upperInterval() - queryLower) * FOUR_BIT_SCALE;

        return new QueryQuantization(
            transposedBuffer,
            queryLower,
            queryScale,
            (float) result.quantizedComponentSum(),
            result.additionalCorrection()
        );
    }

    /** Quantized query state for ADC scoring against a specific centroid. */
    static final class QueryQuantization {
        final byte[] transposed;
        final float queryLower;
        final float queryScale;
        final float queryComponentSum;
        final float additionalCorrection;

        QueryQuantization(byte[] transposed, float queryLower, float queryScale, float queryComponentSum, float additionalCorrection) {
            this.transposed = transposed;
            this.queryLower = queryLower;
            this.queryScale = queryScale;
            this.queryComponentSum = queryComponentSum;
            this.additionalCorrection = additionalCorrection;
        }
    }

    // ========== Min-Heap for Bounded Candidate Buffer ==========

    private void addCandidate(int ordinal, float score) {
        if (candidateCount < maxCandidates) {
            candidateOrdinals[candidateCount] = ordinal;
            candidateScores[candidateCount] = score;
            candidateCount++;
            if (candidateCount == maxCandidates) {
                buildMinHeap();
            }
        } else if (score > candidateScores[0]) {
            candidateOrdinals[0] = ordinal;
            candidateScores[0] = score;
            siftDown(0);
        }
    }

    private void buildMinHeap() {
        for (int i = candidateCount / 2 - 1; i >= 0; i--) {
            siftDown(i);
        }
    }

    private void siftDown(int i) {
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;
            if (left < candidateCount && candidateScores[left] < candidateScores[smallest]) smallest = left;
            if (right < candidateCount && candidateScores[right] < candidateScores[smallest]) smallest = right;
            if (smallest == i) break;
            swap(i, smallest);
            i = smallest;
        }
    }

    private void swap(int a, int b) {
        int tmpOrd = candidateOrdinals[a];
        candidateOrdinals[a] = candidateOrdinals[b];
        candidateOrdinals[b] = tmpOrd;
        float tmpScore = candidateScores[a];
        candidateScores[a] = candidateScores[b];
        candidateScores[b] = tmpScore;
    }

    // ========== Bit Manipulation ==========

    /** 4-bit × 4-bit transposed dot product (Lucene has no equivalent for transposed stripe format). */
    static long int4NibbleDotProduct(byte[] queryTransposed, byte[] docTransposed) {
        int stripeSize = docTransposed.length / 4;
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int d0 = docTransposed[i] & 0xFF;
            int d1 = docTransposed[i + stripeSize] & 0xFF;
            int d2 = docTransposed[i + stripeSize * 2] & 0xFF;
            int d3 = docTransposed[i + stripeSize * 3] & 0xFF;
            int q0 = queryTransposed[i] & 0xFF;
            int q1 = queryTransposed[i + stripeSize] & 0xFF;
            int q2 = queryTransposed[i + stripeSize * 2] & 0xFF;
            int q3 = queryTransposed[i + stripeSize * 3] & 0xFF;
            sum += Integer.bitCount(q0 & d0);
            sum += Integer.bitCount(q0 & d1) * 2L;
            sum += Integer.bitCount(q0 & d2) * 4L;
            sum += Integer.bitCount(q0 & d3) * 8L;
            sum += Integer.bitCount(q1 & d0) * 2L;
            sum += Integer.bitCount(q1 & d1) * 4L;
            sum += Integer.bitCount(q1 & d2) * 8L;
            sum += Integer.bitCount(q1 & d3) * 16L;
            sum += Integer.bitCount(q2 & d0) * 4L;
            sum += Integer.bitCount(q2 & d1) * 8L;
            sum += Integer.bitCount(q2 & d2) * 16L;
            sum += Integer.bitCount(q2 & d3) * 32L;
            sum += Integer.bitCount(q3 & d0) * 8L;
            sum += Integer.bitCount(q3 & d1) * 16L;
            sum += Integer.bitCount(q3 & d2) * 32L;
            sum += Integer.bitCount(q3 & d3) * 64L;
        }
        return sum;
    }

    /** Transpose 4-bit values into nibble stripes for SIMD-friendly dot product. */
    static void transposeHalfByte(byte[] input, byte[] output) {
        int stripeSize = (input.length + 7) / 8;
        for (int i = 0; i < input.length; i++) {
            int val = input[i] & 0x0F;
            int byteIdx = i / 8;
            int bitIdx = 7 - (i % 8);
            if ((val & 1) != 0) output[byteIdx] |= (byte) (1 << bitIdx);
            if ((val & 2) != 0) output[stripeSize + byteIdx] |= (byte) (1 << bitIdx);
            if ((val & 4) != 0) output[2 * stripeSize + byteIdx] |= (byte) (1 << bitIdx);
            if ((val & 8) != 0) output[3 * stripeSize + byteIdx] |= (byte) (1 << bitIdx);
        }
    }
}
