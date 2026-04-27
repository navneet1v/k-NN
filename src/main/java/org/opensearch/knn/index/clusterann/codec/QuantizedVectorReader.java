/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;
import java.util.Arrays;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * Reads block-columnar quantized vectors from .clap and performs ADC scoring.
 *
 * <p>Block layout per BLOCK_SIZE vectors:
 * <pre>
 *   codes[0..BS-1]  ← contiguous for SIMD
 *   lower[0..BS-1]  ← bulk readInts
 *   upper[0..BS-1]  ← bulk readInts
 *   add[0..BS-1]    ← bulk readInts
 *   sum[0..BS-1]    ← bulk readInts
 * </pre>
 */
public final class QuantizedVectorReader {

    private static final float RESCORE_OVERSAMPLE = 2.0f;
    private static final int ADC_MULTIPLIER = 20;
    private static final byte QUERY_BITS = 4;
    private static final float FOUR_BIT_SCALE = 1.0f / ((1 << QUERY_BITS) - 1);

    private final RandomVectorScorer exactScorer;
    private final ClusterANNFieldState fieldState;
    private final VectorSimilarityFunction simFunc;
    private final float[] queryVector;
    private final ScalarBitEncoding encoding;
    private final int packedBytes;
    private final int k;
    private final CandidateCollector candidates;

    // Query quantization scratch
    private final OptimizedScalarQuantizer osq;
    private final byte[] scratch;
    private final byte[][] destinations;
    private final byte[] bitsArray;
    private final float[] queryCopy;
    private final byte[] transposedBuffer;

    // Block read buffers
    private final byte[] flatCodesBuf;
    private final int[] intBuf;
    private final float[] blockLower;
    private final float[] blockUpper;
    private final float[] blockAdd;
    private final int[] blockSum;
    private final float[] rawDotBuf;
    private final boolean nativeAvailable;

    // Cached query quantization state — avoid re-quantizing per block
    private float[] cachedCentroid;
    private byte[] currentTransposed;
    private float currentQueryLower;
    private float currentQueryScale;
    private float currentQueryComponentSum;
    private float currentQueryAdditionalCorrection;

    public QuantizedVectorReader(
        RandomVectorScorer exactScorer,
        IndexInput postingsInput,
        ClusterANNFieldState fieldState,
        VectorSimilarityFunction simFunc,
        float[] queryVector,
        int k
    ) {
        this.exactScorer = exactScorer;
        this.fieldState = fieldState;
        this.simFunc = simFunc;
        this.queryVector = queryVector;
        this.encoding = ScalarBitEncoding.fromDocBits(fieldState.docBits);
        this.packedBytes = encoding.docPackedBytes(fieldState.dimension);
        this.k = k;
        this.candidates = new CandidateCollector(k * ADC_MULTIPLIER);

        this.osq = new OptimizedScalarQuantizer(simFunc);
        this.scratch = new byte[fieldState.dimension];
        this.destinations = new byte[][] { scratch };
        this.bitsArray = new byte[] { QUERY_BITS };
        this.queryCopy = new float[fieldState.dimension];
        this.transposedBuffer = new byte[((fieldState.dimension + 7) / 8) * 4];

        this.flatCodesBuf = new byte[BLOCK_SIZE * packedBytes];
        this.intBuf = new int[BLOCK_SIZE];
        this.blockLower = new float[BLOCK_SIZE];
        this.blockUpper = new float[BLOCK_SIZE];
        this.blockAdd = new float[BLOCK_SIZE];
        this.blockSum = new int[BLOCK_SIZE];
        this.rawDotBuf = new float[BLOCK_SIZE];
        this.nativeAvailable = isNativeAvailable();
    }

    public VectorSimilarityFunction getSimFunc() {
        return simFunc;
    }

    /** Bytes for one block of given size. */
    public long blockBytes(int blockSize) {
        return (long) blockSize * packedBytes + (long) blockSize * Integer.BYTES * 4;
    }

    /**
     * Score a block of vectors from current input position.
     */
    public void scoreBlock(
        IndexInput input,
        int blockStart,
        int blockSize,
        int[] docIdBuf,
        int[] ordBuf,
        boolean[] validBuf,
        float[] centroid,
        float centroidDp
    ) throws IOException {
        // Count valid entries without separate loop — check while reading
        boolean anyValid = false;
        for (int j = 0; j < blockSize; j++) {
            if (validBuf[blockStart + j]) {
                anyValid = true;
                break;
            }
        }

        if (!anyValid) {
            input.skipBytes(blockBytes(blockSize));
            return;
        }

        ensureQueryQuantized(centroid);

        // Bulk read codes
        input.readBytes(flatCodesBuf, 0, blockSize * packedBytes);

        // Bulk read corrections (4 calls total)
        readFloatsFromInts(input, blockLower, blockSize);
        readFloatsFromInts(input, blockUpper, blockSize);
        readFloatsFromInts(input, blockAdd, blockSize);
        input.readInts(blockSum, 0, blockSize);

        // Bulk dot product
        if (nativeAvailable) {
            SimdVectorComputeService.bulkQuantizedDotProduct(
                currentTransposed,
                flatCodesBuf,
                rawDotBuf,
                packedBytes,
                blockSize,
                fieldState.docBits
            );
        } else {
            // Java fallback: offset-based dot product (no per-vector copy)
            for (int j = 0; j < blockSize; j++) {
                if (!validBuf[blockStart + j]) {
                    rawDotBuf[j] = 0;
                    continue;
                }
                int offset = j * packedBytes;
                if (fieldState.docBits == 1) {
                    rawDotBuf[j] = int4BitDotProductOffset(currentTransposed, flatCodesBuf, offset, packedBytes);
                } else if (fieldState.docBits == 2) {
                    rawDotBuf[j] = int4DibitDotProductOffset(currentTransposed, flatCodesBuf, offset, packedBytes);
                } else {
                    rawDotBuf[j] = int4NibbleDotProductOffset(currentTransposed, flatCodesBuf, offset, packedBytes);
                }
            }
        }

        // Apply corrections and collect with early rejection
        float docBitScale = encoding.docBitScale();
        int dim = fieldState.dimension;
        float candidateThreshold = candidates.threshold();
        for (int j = 0; j < blockSize; j++) {
            if (!validBuf[blockStart + j]) continue;

            float docScale = (blockUpper[j] - blockLower[j]) * docBitScale;
            float score = blockLower[j] * currentQueryLower * dim + currentQueryLower * docScale * blockSum[j] + blockLower[j]
                * currentQueryScale * currentQueryComponentSum + docScale * currentQueryScale * rawDotBuf[j];

            float adcSimilarity;
            if (simFunc == VectorSimilarityFunction.EUCLIDEAN) {
                score = currentQueryAdditionalCorrection + blockAdd[j] - 2 * score;
                adcSimilarity = Math.max(1.0f / (1.0f + score), 0);
            } else {
                score += currentQueryAdditionalCorrection + blockAdd[j];
                adcSimilarity = Math.max((1.0f + score) / 2.0f, 0);
                // Early reject: if score can't beat threshold, skip collection
                if (adcSimilarity <= candidateThreshold) continue;
            }

            candidates.add(ordBuf[blockStart + j], adcSimilarity);
        }
    }

    /**
     * Phase 2: rescore top ADC candidates with exact scorer.
     */
    public void finish(KnnCollector collector) throws IOException {
        int rescoreCount = Math.min((int) (k * RESCORE_OVERSAMPLE), candidates.count());
        if (rescoreCount == 0) return;

        int[] topIdx = candidates.topN(rescoreCount);
        for (int idx : topIdx) {
            int ord = candidates.ordinal(idx);
            int doc = exactScorer.ordToDoc(ord);
            float exactScore = exactScorer.score(ord);
            collector.collect(doc, exactScore);
        }
    }

    private void readFloatsFromInts(IndexInput input, float[] out, int count) throws IOException {
        input.readInts(intBuf, 0, count);
        for (int i = 0; i < count; i++)
            out[i] = Float.intBitsToFloat(intBuf[i]);
    }

    /** Cache query quantization per centroid — skip if same centroid reference. */
    private void ensureQueryQuantized(float[] centroid) {
        if (centroid == cachedCentroid) return;
        cachedCentroid = centroid;

        Arrays.fill(scratch, (byte) 0);
        System.arraycopy(queryVector, 0, queryCopy, 0, queryVector.length);
        OptimizedScalarQuantizer.QuantizationResult qResult = osq.multiScalarQuantize(queryCopy, destinations, bitsArray, centroid)[0];

        Arrays.fill(transposedBuffer, (byte) 0);
        OptimizedScalarQuantizer.transposeHalfByte(scratch, transposedBuffer);

        currentTransposed = transposedBuffer;
        currentQueryLower = qResult.lowerInterval();
        currentQueryScale = (qResult.upperInterval() - currentQueryLower) * FOUR_BIT_SCALE;
        currentQueryComponentSum = (float) qResult.quantizedComponentSum();
        currentQueryAdditionalCorrection = qResult.additionalCorrection();
    }

    private static boolean isNativeAvailable() {
        try {
            Class.forName("org.opensearch.knn.jni.SimdVectorComputeService");
            SimdVectorComputeService.bulkQuantizedDotProduct(new byte[0], new byte[0], new float[0], 0, 0, 1);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    // ===== Public static dot product for tests =====

    /** 4-bit × 4-bit transposed dot product (full array). */
    public static long int4NibbleDotProduct(byte[] queryTransposed, byte[] docTransposed) {
        return (long) int4NibbleDotProductOffset(queryTransposed, docTransposed, 0, docTransposed.length);
    }

    // ===== Offset-based dot products (no per-vector array copy) =====

    /** 1-bit doc × 4-bit query: VectorUtil delegates with offset. */
    private static float int4BitDotProductOffset(byte[] query, byte[] docs, int offset, int len) {
        long sum = 0;
        for (int i = 0; i < len; i++) {
            int q0 = query[i] & 0xFF, q1 = query[i + len] & 0xFF;
            int q2 = query[i + len * 2] & 0xFF, q3 = query[i + len * 3] & 0xFF;
            int d = docs[offset + i] & 0xFF;
            sum += Integer.bitCount(q0 & d) + Integer.bitCount(q1 & d) * 2L + Integer.bitCount(q2 & d) * 4L + Integer.bitCount(q3 & d) * 8L;
        }
        return sum;
    }

    /** 2-bit doc × 4-bit query with offset. */
    private static float int4DibitDotProductOffset(byte[] query, byte[] docs, int offset, int len) {
        int stripeSize = len / 2;
        int qStripe = stripeSize; // query always has 4 stripes of stripeSize
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int d0 = docs[offset + i] & 0xFF, d1 = docs[offset + i + stripeSize] & 0xFF;
            int q0 = query[i] & 0xFF, q1 = query[i + qStripe] & 0xFF;
            int q2 = query[i + qStripe * 2] & 0xFF, q3 = query[i + qStripe * 3] & 0xFF;
            sum += Integer.bitCount(q0 & d0) + Integer.bitCount(q0 & d1) * 2L + Integer.bitCount(q1 & d0) * 2L + Integer.bitCount(q1 & d1)
                * 4L + Integer.bitCount(q2 & d0) * 4L + Integer.bitCount(q2 & d1) * 8L + Integer.bitCount(q3 & d0) * 8L + Integer.bitCount(
                    q3 & d1
                ) * 16L;
        }
        return sum;
    }

    /** 4-bit doc × 4-bit query with offset. */
    private static float int4NibbleDotProductOffset(byte[] query, byte[] docs, int offset, int len) {
        int stripeSize = len / 4;
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int d0 = docs[offset + i] & 0xFF, d1 = docs[offset + i + stripeSize] & 0xFF;
            int d2 = docs[offset + i + stripeSize * 2] & 0xFF, d3 = docs[offset + i + stripeSize * 3] & 0xFF;
            int q0 = query[i] & 0xFF, q1 = query[i + stripeSize] & 0xFF;
            int q2 = query[i + stripeSize * 2] & 0xFF, q3 = query[i + stripeSize * 3] & 0xFF;
            sum += Integer.bitCount(q0 & d0) + Integer.bitCount(q0 & d1) * 2L + Integer.bitCount(q0 & d2) * 4L + Integer.bitCount(q0 & d3)
                * 8L + Integer.bitCount(q1 & d0) * 2L + Integer.bitCount(q1 & d1) * 4L + Integer.bitCount(q1 & d2) * 8L + Integer.bitCount(
                    q1 & d3
                ) * 16L + Integer.bitCount(q2 & d0) * 4L + Integer.bitCount(q2 & d1) * 8L + Integer.bitCount(q2 & d2) * 16L + Integer
                    .bitCount(q2 & d3) * 32L + Integer.bitCount(q3 & d0) * 8L + Integer.bitCount(q3 & d1) * 16L + Integer.bitCount(q3 & d2)
                        * 32L + Integer.bitCount(q3 & d3) * 64L;
        }
        return sum;
    }
}
