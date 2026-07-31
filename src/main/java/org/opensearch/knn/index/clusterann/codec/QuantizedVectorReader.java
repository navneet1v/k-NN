/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

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

    private static final byte QUERY_BITS = 4;
    private static final float FOUR_BIT_SCALE = 1.0f / ((1 << QUERY_BITS) - 1);

    private final RandomVectorScorer exactScorer;
    private final ClusterANNFieldState fieldState;
    private final VectorSimilarityFunction simFunc;
    private final float[] queryVector;
    private final ScalarBitEncoding encoding;
    private final int packedBytes;
    private final int k;
    private KnnCollector currentCollector;

    // Bulk collection buffers
    private final int[] bulkDocs;
    private final float[] bulkScores;
    private final int[] validOffsets;
    private int bulkCount;
    private float bulkMaxScore;
    private long bytesRead;

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

    // Cached query quantization state — avoid re-quantizing per block
    private float[] cachedCentroid;
    private byte[] currentTransposed;
    private float currentQueryLower;
    private float currentQueryScale;
    private float currentQueryComponentSum;
    private float currentQueryAdditionalCorrection;
    private float currentCentroidNormSq;

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
        this.bulkDocs = new int[BLOCK_SIZE];
        this.bulkScores = new float[BLOCK_SIZE];
        this.validOffsets = new int[BLOCK_SIZE];
        this.bulkCount = 0;
        this.bulkMaxScore = Float.NEGATIVE_INFINITY;
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

        // Read corrections FIRST (small — 16 bytes per vector)
        readFloatsFromInts(input, blockLower, blockSize);
        readFloatsFromInts(input, blockUpper, blockSize);
        readFloatsFromInts(input, blockAdd, blockSize);
        input.readInts(blockSum, 0, blockSize);
        long correctionsBytes = (long) blockSize * Integer.BYTES * 4;
        bytesRead += correctionsBytes;

        // Block-level early skip: compute upper bound from corrections alone
        float blockThreshold = currentCollector.minCompetitiveSimilarity();
        if (blockThreshold > Float.NEGATIVE_INFINITY && simFunc != VectorSimilarityFunction.EUCLIDEAN) {
            // Fast pre-check: use block-wide max/min of corrections for single-comparison skip
            float maxUpper = Float.NEGATIVE_INFINITY, minLower = Float.MAX_VALUE, maxAdd = Float.NEGATIVE_INFINITY;
            int maxAbsSum = 0;
            for (int j = 0; j < blockSize; j++) {
                if (!validBuf[blockStart + j]) continue;
                if (blockUpper[j] > maxUpper) maxUpper = blockUpper[j];
                if (blockLower[j] < minLower) minLower = blockLower[j];
                if (blockAdd[j] > maxAdd) maxAdd = blockAdd[j];
                int absSum = Math.abs(blockSum[j]);
                if (absSum > maxAbsSum) maxAbsSum = absSum;
            }
            if (maxUpper != Float.NEGATIVE_INFINITY) {
                float docBitScaleCheck = encoding.docBitScale();
                float maxDocScale = (maxUpper - minLower) * docBitScaleCheck;
                float fastUpperBound = minLower * currentQueryLower * fieldState.dimension
                    + Math.abs(currentQueryLower) * maxDocScale * maxAbsSum
                    + Math.abs(minLower) * Math.abs(currentQueryScale) * Math.abs(currentQueryComponentSum)
                    + maxDocScale * Math.abs(currentQueryScale) * packedBytes * 4f
                    + maxAdd + centroidDp - currentCentroidNormSq;
                float fastSimilarity = fastUpperBound >= 0 ? fastUpperBound + 1 : 1f / (1f - fastUpperBound);
                if (fastSimilarity <= blockThreshold) {
                    input.skipBytes((long) blockSize * packedBytes);
                    return;
                }
            }

            // Detailed per-vector check (only if fast check didn't skip)
            float maxUpperBound = Float.NEGATIVE_INFINITY;
            float docBitScaleCheck = encoding.docBitScale();
            for (int j = 0; j < blockSize; j++) {
                if (!validBuf[blockStart + j]) continue;
                float docScale = (blockUpper[j] - blockLower[j]) * docBitScaleCheck;
                float maxScore = blockLower[j] * currentQueryLower * fieldState.dimension + Math.abs(currentQueryLower) * docScale * Math
                    .abs(blockSum[j]) + Math.abs(blockLower[j]) * Math.abs(currentQueryScale) * Math.abs(currentQueryComponentSum)
                    + docScale * Math.abs(currentQueryScale) * packedBytes * 4f;
                float upperBound = maxScore + blockAdd[j] + centroidDp - currentCentroidNormSq;
                if (upperBound > maxUpperBound) maxUpperBound = upperBound;
            }
            float upperSimilarity = maxUpperBound >= 0 ? maxUpperBound + 1 : 1f / (1f - maxUpperBound);
            if (upperSimilarity <= blockThreshold) {
                input.skipBytes((long) blockSize * packedBytes);
                return;
            }
        }

        // Read codes (only if block is potentially competitive)
        input.readBytes(flatCodesBuf, 0, blockSize * packedBytes);
        bytesRead += (long) blockSize * packedBytes;

        // Compact valid entries — enables branchless dot product loop
        int validCount = 0;
        for (int j = 0; j < blockSize; j++) {
            if (validBuf[blockStart + j]) validOffsets[validCount++] = j;
        }
        // Dot product over valid entries only — batch 4 vectors for ILP
        if (fieldState.docBits == 1) {
            int v = 0;
            // Batch of 4: share query loads across 4 doc vectors
            for (; v + 3 < validCount; v += 4) {
                int j0 = validOffsets[v], j1 = validOffsets[v+1], j2 = validOffsets[v+2], j3 = validOffsets[v+3];
                rawDotBuf[j0] = int4BitDotProduct4(currentTransposed, flatCodesBuf, j0 * packedBytes, j1 * packedBytes, j2 * packedBytes, j3 * packedBytes, packedBytes, rawDotBuf, j1, j2, j3);
            }
            // Remainder
            for (; v < validCount; v++) {
                int j = validOffsets[v];
                rawDotBuf[j] = int4BitDotProductOffset(currentTransposed, flatCodesBuf, j * packedBytes, packedBytes);
            }
        } else if (fieldState.docBits == 2) {
            for (int v = 0; v < validCount; v++) {
                int j = validOffsets[v];
                rawDotBuf[j] = int4DibitDotProductOffset(currentTransposed, flatCodesBuf, j * packedBytes, packedBytes);
            }
        } else {
            for (int v = 0; v < validCount; v++) {
                int j = validOffsets[v];
                rawDotBuf[j] = int4NibbleDotProductOffset(currentTransposed, flatCodesBuf, j * packedBytes, packedBytes);
            }
        }

        // Bulk score and collect directly into KnnCollector
        float docBitScale = encoding.docBitScale();
        int dim = fieldState.dimension;
        // Precompute constants (invariant across vectors in this block)
        float qLowerDim = currentQueryLower * dim;
        float qScaleCompSum = currentQueryScale * currentQueryComponentSum;
        float dpMinusNorm = centroidDp - currentCentroidNormSq;
        bulkCount = 0;
        bulkMaxScore = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < blockSize; j++) {
            if (!validBuf[blockStart + j]) continue;

            float docScale = (blockUpper[j] - blockLower[j]) * docBitScale;
            float score = blockLower[j] * qLowerDim + currentQueryLower * docScale * blockSum[j] + blockLower[j] * qScaleCompSum + docScale
                * currentQueryScale * rawDotBuf[j];

            float adcSimilarity;
            if (simFunc == VectorSimilarityFunction.EUCLIDEAN) {
                score = currentQueryAdditionalCorrection + blockAdd[j] - 2 * score;
                adcSimilarity = 1.0f / (1.0f + Math.max(score, 0f));
            } else {
                float rawDot = score + blockAdd[j] + dpMinusNorm;
                if (simFunc == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
                    adcSimilarity = rawDot >= 0 ? rawDot + 1 : 1f / (1f - rawDot);
                } else {
                    adcSimilarity = Math.max((1.0f + rawDot) / 2.0f, 0f);
                }
            }

            bulkDocs[bulkCount] = exactScorer.ordToDoc(ordBuf[blockStart + j]);
            bulkScores[bulkCount] = adcSimilarity;
            if (adcSimilarity > bulkMaxScore) bulkMaxScore = adcSimilarity;
            bulkCount++;
        }

        // Bulk collect: one threshold check for entire block
        if (bulkCount > 0 && bulkMaxScore > currentCollector.minCompetitiveSimilarity()) {
            for (int j = 0; j < bulkCount; j++) {
                currentCollector.collect(bulkDocs[j], bulkScores[j]);
            }
        }
    }

    /** Set the collector for this search. Must be called before scoreBlock. */
    public void setCollector(KnnCollector collector) {
        this.currentCollector = collector;
        this.bytesRead = 0;
    }

    /** Actual bytes read during ADC scoring (excludes skipped blocks). */
    public long getBytesRead() {
        return bytesRead;
    }

    /**
     * No-op — collection happens directly during scoreBlock.
     */
    public void finish(KnnCollector collector) {
        // Direct collection eliminates the need for drain
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
        currentCentroidNormSq = VectorUtil.dotProduct(centroid, centroid);
    }


    // ===== Public static dot product for tests =====

    /** 4-bit × 4-bit transposed dot product (full array). */
    public static long int4NibbleDotProduct(byte[] queryTransposed, byte[] docTransposed) {
        return (long) int4NibbleDotProductOffset(queryTransposed, docTransposed, 0, docTransposed.length);
    }

    // ===== Offset-based dot products (no per-vector array copy) =====

    /** 1-bit doc × 4-bit query: VectorUtil delegates with offset. */
    private static final java.lang.invoke.VarHandle LONG_LE = java.lang.invoke.MethodHandles.byteArrayViewVarHandle(
        long[].class,
        java.nio.ByteOrder.LITTLE_ENDIAN
    );


    /** Batch 4 vectors: share query loads, exploit instruction-level parallelism. Returns score for first vector, writes others to rawDotBuf. */
    private static float int4BitDotProduct4(byte[] query, byte[] docs, int off0, int off1, int off2, int off3, int len, float[] outBuf, int idx1, int idx2, int idx3) {
        long s0_0 = 0, s0_1 = 0, s0_2 = 0, s0_3 = 0;
        long s1_0 = 0, s1_1 = 0, s1_2 = 0, s1_3 = 0;
        long s2_0 = 0, s2_1 = 0, s2_2 = 0, s2_3 = 0;
        long s3_0 = 0, s3_1 = 0, s3_2 = 0, s3_3 = 0;
        int r = 0;
        for (final int upperBound = len & -Long.BYTES; r < upperBound; r += Long.BYTES) {
            long q0 = (long) LONG_LE.get(query, r);
            long q1 = (long) LONG_LE.get(query, r + len);
            long q2 = (long) LONG_LE.get(query, r + len * 2);
            long q3 = (long) LONG_LE.get(query, r + len * 3);
            long d0 = (long) LONG_LE.get(docs, off0 + r);
            long d1 = (long) LONG_LE.get(docs, off1 + r);
            long d2 = (long) LONG_LE.get(docs, off2 + r);
            long d3 = (long) LONG_LE.get(docs, off3 + r);
            s0_0 += Long.bitCount(q0 & d0); s0_1 += Long.bitCount(q1 & d0); s0_2 += Long.bitCount(q2 & d0); s0_3 += Long.bitCount(q3 & d0);
            s1_0 += Long.bitCount(q0 & d1); s1_1 += Long.bitCount(q1 & d1); s1_2 += Long.bitCount(q2 & d1); s1_3 += Long.bitCount(q3 & d1);
            s2_0 += Long.bitCount(q0 & d2); s2_1 += Long.bitCount(q1 & d2); s2_2 += Long.bitCount(q2 & d2); s2_3 += Long.bitCount(q3 & d2);
            s3_0 += Long.bitCount(q0 & d3); s3_1 += Long.bitCount(q1 & d3); s3_2 += Long.bitCount(q2 & d3); s3_3 += Long.bitCount(q3 & d3);
        }
        outBuf[idx1] = s1_0 + s1_1 * 2L + s1_2 * 4L + s1_3 * 8L;
        outBuf[idx2] = s2_0 + s2_1 * 2L + s2_2 * 4L + s2_3 * 8L;
        outBuf[idx3] = s3_0 + s3_1 * 2L + s3_2 * 4L + s3_3 * 8L;
        return s0_0 + s0_1 * 2L + s0_2 * 4L + s0_3 * 8L;
    }

    private static float int4BitDotProductOffset(byte[] query, byte[] docs, int offset, int len) {
        long sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        int r = 0;
        for (final int upperBound = len & -Long.BYTES; r < upperBound; r += Long.BYTES) {
            long d = (long) LONG_LE.get(docs, offset + r);
            sum0 += Long.bitCount((long) LONG_LE.get(query, r) & d);
            sum1 += Long.bitCount((long) LONG_LE.get(query, r + len) & d);
            sum2 += Long.bitCount((long) LONG_LE.get(query, r + len * 2) & d);
            sum3 += Long.bitCount((long) LONG_LE.get(query, r + len * 3) & d);
        }
        for (; r < len; r++) {
            int d = docs[offset + r] & 0xFF;
            sum0 += Integer.bitCount((query[r] & d) & 0xFF);
            sum1 += Integer.bitCount((query[r + len] & d) & 0xFF);
            sum2 += Integer.bitCount((query[r + len * 2] & d) & 0xFF);
            sum3 += Integer.bitCount((query[r + len * 3] & d) & 0xFF);
        }
        return sum0 + sum1 * 2L + sum2 * 4L + sum3 * 8L;
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
