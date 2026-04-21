/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
final class TwoPhaseClusterANNScorer implements ClusterANNScorer {

    private static final float RESCORE_OVERSAMPLE = 2.0f;
    private static final int ADC_MULTIPLIER = 20;
    private static final byte QUERY_BITS = 4;
    private static final float FOUR_BIT_SCALE = 1.0f / 15f;

    private final RandomVectorScorer exactScorer;
    private final IndexInput quantizedInput;
    private final ClusterANNFieldState fieldState;
    private final VectorSimilarityFunction simFunc;
    private final OptimizedScalarQuantizer luceneOsq;
    private final float[] queryVector;
    private final int k;

    // Per-vector record size in .claq: packed codes + 4 correction ints
    private final int packedBytesPerVec;
    private final int recordSize;
    private final float docBitScale;

    // ADC candidate buffer: [ordinal, adcScore]
    private final List<float[]> adcCandidates = new ArrayList<>();
    private final int maxCandidates;

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
        this.luceneOsq = new OptimizedScalarQuantizer(simFunc);
        this.queryVector = queryVector;
        this.k = k;
        this.packedBytesPerVec = ClusterANN1040KnnVectorsWriter.packedBytesPerVector(fieldState.dimension, fieldState.docBits);
        this.recordSize = packedBytesPerVec + 16; // 4 ints = 16 bytes
        this.docBitScale = 1.0f / ((1 << fieldState.docBits) - 1);
        this.maxCandidates = k * ADC_MULTIPLIER;
    }

    @Override
    public void prefetch(int[] ordinals, int count) throws IOException {
        // Prefetch quantized data from .claq for upcoming ADC scoring
        org.opensearch.knn.index.codec.scorer.PrefetchHelper.prefetch(
            quantizedInput, fieldState.quantizedOffset, recordSize, ordinals, count
        );
    }

    /**
     * ADC score: reads quantized data and computes approximate similarity.
     * The query is quantized per-centroid at call time by the reader.
     */
    @Override
    public float score(int ordinal) throws IOException {
        // This is called per-ordinal during posting list scan.
        // We buffer candidates and defer exact scoring to finish().
        return Float.NaN; // Not used directly — see scoreADC()
    }

    /**
     * Score a candidate using ADC against a specific centroid's quantized query.
     *
     * @param ordinal          vector ordinal
     * @param queryTransposed  transposed 4-bit quantized query
     * @param ay               query lower interval
     * @param ly               query (upper - lower) * FOUR_BIT_SCALE
     * @param queryComponentSum query quantized component sum
     * @param queryAdditionalCorrection query additional correction
     */
    void scoreADC(int ordinal, byte[] queryTransposed, float ay, float ly,
                  float queryComponentSum, float queryAdditionalCorrection) throws IOException {
        long qOffset = fieldState.quantizedOffset + (long) ordinal * recordSize;
        quantizedInput.seek(qOffset);

        byte[] packed = new byte[packedBytesPerVec];
        quantizedInput.readBytes(packed, 0, packedBytesPerVec);
        float ax = Float.intBitsToFloat(quantizedInput.readInt());
        float docUpper = Float.intBitsToFloat(quantizedInput.readInt());
        float docAdditionalCorrection = Float.intBitsToFloat(quantizedInput.readInt());
        float docComponentSum = (float) quantizedInput.readInt();
        float lx = (docUpper - ax) * docBitScale;

        // Dot product based on doc bit width
        long rawDot;
        if (fieldState.docBits == 1) {
            rawDot = int4BitDotProduct(queryTransposed, packed);
        } else if (fieldState.docBits == 2) {
            rawDot = int4DibitDotProduct(queryTransposed, packed);
        } else {
            rawDot = int4NibbleDotProduct(queryTransposed, packed);
        }

        // BBQ formula
        float score = ax * ay * fieldState.dimension
            + ay * lx * docComponentSum
            + ax * ly * queryComponentSum
            + lx * ly * (float) rawDot;

        float adcSimilarity;
        if (simFunc == VectorSimilarityFunction.EUCLIDEAN) {
            score = queryAdditionalCorrection + docAdditionalCorrection - 2 * score;
            adcSimilarity = Math.max(1.0f / (1.0f + score), 0);
        } else {
            // DOT_PRODUCT / COSINE / MAXIMUM_INNER_PRODUCT
            score += queryAdditionalCorrection + docAdditionalCorrection;
            adcSimilarity = Math.max((1.0f + score) / 2.0f, 0);
        }

        adcCandidates.add(new float[]{ordinal, adcSimilarity});
        // Evict worst if over capacity (keep as simple list, sort at finish)
    }

    @Override
    public int ordToDoc(int ordinal) {
        return exactScorer.ordToDoc(ordinal);
    }

    /**
     * Phase 2: rescore top ADC candidates with exact scorer and collect results.
     */
    @Override
    public void finish(ResultCollector collector) throws IOException {
        // Sort by ADC score descending
        adcCandidates.sort((a, b) -> Float.compare(b[1], a[1]));

        int rescoreCount = Math.min((int) (k * RESCORE_OVERSAMPLE), adcCandidates.size());
        for (int i = 0; i < rescoreCount; i++) {
            int ord = (int) adcCandidates.get(i)[0];
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
        byte[] scratch = new byte[fieldState.dimension];
        byte[][] destinations = new byte[][]{scratch};
        byte[] bitsArray = new byte[]{QUERY_BITS};

        float[] queryCopy = queryVector.clone();
        OptimizedScalarQuantizer.QuantizationResult result =
            luceneOsq.multiScalarQuantize(queryCopy, destinations, bitsArray, centroid)[0];

        int stripeSize = (fieldState.dimension + 7) / 8;
        byte[] transposed = new byte[stripeSize * 4];
        transposeHalfByte(scratch, transposed);

        float ay = result.lowerInterval();
        float ly = (result.upperInterval() - ay) * FOUR_BIT_SCALE;

        return new QueryQuantization(transposed, ay, ly,
            (float) result.quantizedComponentSum(), result.additionalCorrection());
    }

    /** Quantized query state for ADC scoring against a specific centroid. */
    static final class QueryQuantization {
        final byte[] transposed;
        final float ay;
        final float ly;
        final float queryComponentSum;
        final float additionalCorrection;

        QueryQuantization(byte[] transposed, float ay, float ly, float queryComponentSum, float additionalCorrection) {
            this.transposed = transposed;
            this.ay = ay;
            this.ly = ly;
            this.queryComponentSum = queryComponentSum;
            this.additionalCorrection = additionalCorrection;
        }
    }

    // ========== Bit Manipulation ==========

    /** Compute dot product between 4-bit transposed query and 1-bit packed document. */
    static long int4BitDotProduct(byte[] queryTransposed, byte[] packed) {
        long sum = 0;
        int stripeSize = packed.length;
        for (int i = 0; i < stripeSize; i++) {
            int docByte = packed[i] & 0xFF;
            sum += Integer.bitCount(docByte & (queryTransposed[i] & 0xFF));
            sum += 2L * Integer.bitCount(docByte & (queryTransposed[stripeSize + i] & 0xFF));
            sum += 4L * Integer.bitCount(docByte & (queryTransposed[2 * stripeSize + i] & 0xFF));
            sum += 8L * Integer.bitCount(docByte & (queryTransposed[3 * stripeSize + i] & 0xFF));
        }
        return sum;
    }

    /** Compute dot product between 4-bit transposed query and 2-bit transposed document. */
    static long int4DibitDotProduct(byte[] queryTransposed, byte[] dibitPacked) {
        int stripeSize = dibitPacked.length / 2;
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int dLower = dibitPacked[i] & 0xFF;
            int dUpper = dibitPacked[i + stripeSize] & 0xFF;

            int q0 = queryTransposed[i] & 0xFF;
            int q1 = queryTransposed[i + stripeSize] & 0xFF;
            int q2 = queryTransposed[i + stripeSize * 2] & 0xFF;
            int q3 = queryTransposed[i + stripeSize * 3] & 0xFF;

            // Cross-product: int4 weights (1,2,4,8) × dibit weights (1,2)
            sum += Integer.bitCount(q0 & dLower);           // 1*1
            sum += Integer.bitCount(q0 & dUpper) * 2L;      // 1*2
            sum += Integer.bitCount(q1 & dLower) * 2L;      // 2*1
            sum += Integer.bitCount(q1 & dUpper) * 4L;      // 2*2
            sum += Integer.bitCount(q2 & dLower) * 4L;      // 4*1
            sum += Integer.bitCount(q2 & dUpper) * 8L;      // 4*2
            sum += Integer.bitCount(q3 & dLower) * 8L;      // 8*1
            sum += Integer.bitCount(q3 & dUpper) * 16L;     // 8*2
        }
        return sum;
    }

    /** Compute dot product between 4-bit transposed query and 4-bit transposed document. */
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

            // Full cross-product: int4 weights (1,2,4,8) × int4 weights (1,2,4,8)
            sum += Integer.bitCount(q0 & d0);            // 1*1
            sum += Integer.bitCount(q0 & d1) * 2L;       // 1*2
            sum += Integer.bitCount(q0 & d2) * 4L;       // 1*4
            sum += Integer.bitCount(q0 & d3) * 8L;       // 1*8
            sum += Integer.bitCount(q1 & d0) * 2L;       // 2*1
            sum += Integer.bitCount(q1 & d1) * 4L;       // 2*2
            sum += Integer.bitCount(q1 & d2) * 8L;       // 2*4
            sum += Integer.bitCount(q1 & d3) * 16L;      // 2*8
            sum += Integer.bitCount(q2 & d0) * 4L;       // 4*1
            sum += Integer.bitCount(q2 & d1) * 8L;       // 4*2
            sum += Integer.bitCount(q2 & d2) * 16L;      // 4*4
            sum += Integer.bitCount(q2 & d3) * 32L;      // 4*8
            sum += Integer.bitCount(q3 & d0) * 8L;       // 8*1
            sum += Integer.bitCount(q3 & d1) * 16L;      // 8*2
            sum += Integer.bitCount(q3 & d2) * 32L;      // 8*4
            sum += Integer.bitCount(q3 & d3) * 64L;      // 8*8
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
