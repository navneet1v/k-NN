/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

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
@Log4j2
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

    public void scoreBlock(IndexInput input,
                      int blockStart,
                      int blockSize,
                      int[] docIdBuf,
                      int[] ordBuf,
                      boolean[] validBuf,
                      float centroidDp) throws IOException {
        scoreBlock(input, blockStart, blockSize, docIdBuf, ordBuf, validBuf, centroidDp, false);
    }

    /**
     * Score a block of vectors from current input position.
     * Caller must invoke {@link #ensureQueryQuantized(float[])} before the first call.
     */
    public void scoreBlock(
        IndexInput input,
        int blockStart,
        int blockSize,
        int[] docIdBuf,
        int[] ordBuf,
        boolean[] validBuf,
        float centroidDp,
        boolean useBulkSIMD
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
            float maxUpperBound = Float.NEGATIVE_INFINITY;
            float docBitScaleCheck = encoding.docBitScale();
            for (int j = 0; j < blockSize; j++) {
                if (!validBuf[blockStart + j]) continue;
                // Upper bound: assume max rawDot contribution (generous estimate)
                float docScale = (blockUpper[j] - blockLower[j]) * docBitScaleCheck;
                float maxScore = blockLower[j] * currentQueryLower * fieldState.dimension
                    + Math.abs(currentQueryLower) * docScale * Math.abs(blockSum[j])
                    + Math.abs(blockLower[j]) * Math.abs(currentQueryScale) * Math.abs(currentQueryComponentSum)
                    + docScale * Math.abs(currentQueryScale) * packedBytes * 4f;
                float upperBound = maxScore + blockAdd[j] + centroidDp - currentCentroidNormSq;
                if (upperBound > maxUpperBound) maxUpperBound = upperBound;
            }
            float upperSimilarity = maxUpperBound >= 0 ? maxUpperBound + 1 : 1f / (1f - maxUpperBound);
            if (upperSimilarity <= blockThreshold) {
                // Skip codes entirely — this block can't compete
                input.skipBytes((long) blockSize * packedBytes);
                return;
            }
        }

        bytesRead += (long) blockSize * packedBytes;
        if (useBulkSIMD) {
            // MemorySegment + bulk4: process 4 vectors at a time, zero-copy from mmap
            int validCount = 0;
            for (int j = 0; j < blockSize; j++) {
                if (validBuf[blockStart + j]) validOffsets[validCount++] = j;
            }
            final MemorySegment memorySegment = ((MemorySegmentAccessInput) input)
                .segmentSliceOrNull(input.getFilePointer(), (long) blockSize * packedBytes);
            if (memorySegment == null) {
                throw new IllegalStateException("MemorySegment unavailable — MMapDirectory required for bulk SIMD path");
            }
            if (fieldState.docBits == 1) {
                int v = 0;
                for (; v + 3 < validCount; v += 4) {
                    int j0 = validOffsets[v], j1 = validOffsets[v + 1], j2 = validOffsets[v + 2], j3 = validOffsets[v + 3];
                    int4BitDotProductBulk4(currentTransposed, memorySegment,
                        (long) j0 * packedBytes, (long) j1 * packedBytes,
                        (long) j2 * packedBytes, (long) j3 * packedBytes,
                        packedBytes, rawDotBuf, j0, j1, j2, j3);
                }
                for (; v < validCount; v++) {
                    int j = validOffsets[v];
                    rawDotBuf[j] = int4BitDotProductOffset(currentTransposed, memorySegment, (long) j * packedBytes, packedBytes);
                }
            }
            input.skipBytes((long) blockSize * packedBytes);
        } else {
            // Baseline: readBytes into byte[] + single-vector dot product
            input.readBytes(flatCodesBuf, 0, blockSize * packedBytes);
            int validCount = 0;
            for (int j = 0; j < blockSize; j++) {
                if (validBuf[blockStart + j]) validOffsets[validCount++] = j;
            }
            if (fieldState.docBits == 1) {
                for (int v = 0; v < validCount; v++) {
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
            float score = blockLower[j] * qLowerDim + currentQueryLower * docScale * blockSum[j] + blockLower[j]
                * qScaleCompSum + docScale * currentQueryScale * rawDotBuf[j];

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
    public long getBytesRead() { return bytesRead; }

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

    /** Quantize the query against a centroid. Cached — no-op if same centroid reference. */
    public void ensureQueryQuantized(float[] centroid) {
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
    private static final java.lang.invoke.VarHandle LONG_LE =
        java.lang.invoke.MethodHandles.byteArrayViewVarHandle(long[].class, java.nio.ByteOrder.LITTLE_ENDIAN);

    /**
     * Computes the 4-bit query × 1-bit doc asymmetric dot product from a heap byte[].
     *
     * <p>The query is transposed into 4 bit-plane stripes of length {@code len}, stored
     * consecutively: stripe0 at [0..len), stripe1 at [len..2*len), etc. Each stripe holds
     * one bit-plane of the 4-bit quantized query. The doc vector is a single 1-bit packed
     * array of length {@code len}.
     *
     * <p>For each byte position, we AND the doc byte with each query stripe byte and popcount
     * the result. The final score weights the 4 stripes by powers of 2 (1, 2, 4, 8) to
     * reconstruct the 4-bit dot product value.
     *
     * <p>The main loop processes 8 bytes at a time using VarHandle long reads for throughput.
     * A scalar tail handles the remaining 0-7 bytes.
     *
     * @param query  transposed query bit-planes: 4 stripes of {@code len} bytes each
     * @param docs   flat doc codes buffer
     * @param offset byte offset of this doc vector within {@code docs}
     * @param len    packed byte length per doc vector (e.g., 96 at dim=768, 1-bit)
     * @return weighted popcount sum: sum0 + sum1*2 + sum2*4 + sum3*8
     */
    private static float int4BitDotProductOffset(byte[] query, byte[] docs, int offset, int len) {
        long sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        int r = 0;
        // Main loop: process 8 bytes per iteration via VarHandle long reads
        for (final int upperBound = len & -Long.BYTES; r < upperBound; r += Long.BYTES) {
            long d = (long) LONG_LE.get(docs, offset + r);
            sum0 += Long.bitCount((long) LONG_LE.get(query, r) & d);
            sum1 += Long.bitCount((long) LONG_LE.get(query, r + len) & d);
            sum2 += Long.bitCount((long) LONG_LE.get(query, r + len * 2) & d);
            sum3 += Long.bitCount((long) LONG_LE.get(query, r + len * 3) & d);
        }
        // Scalar tail for remaining bytes not aligned to 8
        for (; r < len; r++) {
            int d = docs[offset + r] & 0xFF;
            sum0 += Integer.bitCount((query[r] & d) & 0xFF);
            sum1 += Integer.bitCount((query[r + len] & d) & 0xFF);
            sum2 += Integer.bitCount((query[r + len * 2] & d) & 0xFF);
            sum3 += Integer.bitCount((query[r + len * 3] & d) & 0xFF);
        }
        return sum0 + sum1 * 2L + sum2 * 4L + sum3 * 8L;
    }

    private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;
    private static final int VECTOR_BYTE_SIZE = BYTE_SPECIES.vectorByteSize();

    /**
     * Panama Vector API variant: 4-bit query × 1-bit doc dot product reading directly from MemorySegment.
     *
     * <p>Same logic as the byte[] overload but uses {@code ByteVector.fromMemorySegment()} for
     * zero-copy SIMD loads from mmap'd storage. Doc bytes are loaded directly from the mapped
     * page cache into SIMD registers — no intermediate heap copy. Query stripes are loaded
     * from the heap byte[] via {@code ByteVector.fromArray()}.
     *
     * <p>Each iteration: AND query stripe with doc vector, vectorized BIT_COUNT, accumulate
     * into LongVector accumulators. After the main loop, reduceLanes sums each accumulator
     * and a scalar tail handles the remaining bytes.
     *
     * @param query  transposed query bit-planes: 4 stripes of {@code len} bytes each (heap)
     * @param docs   MemorySegment backed by mmap'd file (zero-copy access)
     * @param offset byte offset of this doc vector within the segment
     * @param len    packed byte length per doc vector
     * @return weighted popcount sum: sum0 + sum1*2 + sum2*4 + sum3*8
     */
    static float int4BitDotProductOffset(byte[] query, MemorySegment docs, long offset, int len) {
        LongVector acc0 = LongVector.zero(LONG_SPECIES);
        LongVector acc1 = LongVector.zero(LONG_SPECIES);
        LongVector acc2 = LongVector.zero(LONG_SPECIES);
        LongVector acc3 = LongVector.zero(LONG_SPECIES);

        int r = 0;
        for (final int upperBound = BYTE_SPECIES.loopBound(len); r < upperBound; r += VECTOR_BYTE_SIZE) {
            LongVector d = ByteVector.fromMemorySegment(BYTE_SPECIES, docs, offset + r, ByteOrder.LITTLE_ENDIAN)
                .reinterpretAsLongs();
            LongVector q0 = ByteVector.fromArray(BYTE_SPECIES, query, r).reinterpretAsLongs();
            LongVector q1 = ByteVector.fromArray(BYTE_SPECIES, query, r + len).reinterpretAsLongs();
            LongVector q2 = ByteVector.fromArray(BYTE_SPECIES, query, r + len * 2).reinterpretAsLongs();
            LongVector q3 = ByteVector.fromArray(BYTE_SPECIES, query, r + len * 3).reinterpretAsLongs();

            acc0 = acc0.add(q0.and(d).lanewise(VectorOperators.BIT_COUNT));
            acc1 = acc1.add(q1.and(d).lanewise(VectorOperators.BIT_COUNT));
            acc2 = acc2.add(q2.and(d).lanewise(VectorOperators.BIT_COUNT));
            acc3 = acc3.add(q3.and(d).lanewise(VectorOperators.BIT_COUNT));
        }

        long sum0 = acc0.reduceLanes(VectorOperators.ADD);
        long sum1 = acc1.reduceLanes(VectorOperators.ADD);
        long sum2 = acc2.reduceLanes(VectorOperators.ADD);
        long sum3 = acc3.reduceLanes(VectorOperators.ADD);

        for (; r < len; r++) {
            int d = docs.get(ValueLayout.JAVA_BYTE, offset + r) & 0xFF;
            sum0 += Integer.bitCount((query[r] & d) & 0xFF);
            sum1 += Integer.bitCount((query[r + len] & d) & 0xFF);
            sum2 += Integer.bitCount((query[r + len * 2] & d) & 0xFF);
            sum3 += Integer.bitCount((query[r + len * 3] & d) & 0xFF);
        }
        return sum0 + sum1 * 2L + sum2 * 4L + sum3 * 8L;
    }

    static void int4BitDotProductBulk4(
        byte[] query, MemorySegment docs,
        long off0, long off1, long off2, long off3,
        int len, float[] results, int ri0, int ri1, int ri2, int ri3
    ) {
        LongVector a0_0 = LongVector.zero(LONG_SPECIES), a0_1 = LongVector.zero(LONG_SPECIES);
        LongVector a0_2 = LongVector.zero(LONG_SPECIES), a0_3 = LongVector.zero(LONG_SPECIES);
        LongVector a1_0 = LongVector.zero(LONG_SPECIES), a1_1 = LongVector.zero(LONG_SPECIES);
        LongVector a1_2 = LongVector.zero(LONG_SPECIES), a1_3 = LongVector.zero(LONG_SPECIES);
        LongVector a2_0 = LongVector.zero(LONG_SPECIES), a2_1 = LongVector.zero(LONG_SPECIES);
        LongVector a2_2 = LongVector.zero(LONG_SPECIES), a2_3 = LongVector.zero(LONG_SPECIES);
        LongVector a3_0 = LongVector.zero(LONG_SPECIES), a3_1 = LongVector.zero(LONG_SPECIES);
        LongVector a3_2 = LongVector.zero(LONG_SPECIES), a3_3 = LongVector.zero(LONG_SPECIES);

        int r = 0;
        for (final int upperBound = BYTE_SPECIES.loopBound(len); r < upperBound; r += VECTOR_BYTE_SIZE) {
            LongVector q0 = ByteVector.fromArray(BYTE_SPECIES, query, r).reinterpretAsLongs();
            LongVector q1 = ByteVector.fromArray(BYTE_SPECIES, query, r + len).reinterpretAsLongs();
            LongVector q2 = ByteVector.fromArray(BYTE_SPECIES, query, r + len * 2).reinterpretAsLongs();
            LongVector q3 = ByteVector.fromArray(BYTE_SPECIES, query, r + len * 3).reinterpretAsLongs();

            LongVector d0 = ByteVector.fromMemorySegment(BYTE_SPECIES, docs, off0 + r, ByteOrder.LITTLE_ENDIAN).reinterpretAsLongs();
            LongVector d1 = ByteVector.fromMemorySegment(BYTE_SPECIES, docs, off1 + r, ByteOrder.LITTLE_ENDIAN).reinterpretAsLongs();
            LongVector d2 = ByteVector.fromMemorySegment(BYTE_SPECIES, docs, off2 + r, ByteOrder.LITTLE_ENDIAN).reinterpretAsLongs();
            LongVector d3 = ByteVector.fromMemorySegment(BYTE_SPECIES, docs, off3 + r, ByteOrder.LITTLE_ENDIAN).reinterpretAsLongs();

            a0_0 = a0_0.add(q0.and(d0).lanewise(VectorOperators.BIT_COUNT));
            a0_1 = a0_1.add(q1.and(d0).lanewise(VectorOperators.BIT_COUNT));
            a0_2 = a0_2.add(q2.and(d0).lanewise(VectorOperators.BIT_COUNT));
            a0_3 = a0_3.add(q3.and(d0).lanewise(VectorOperators.BIT_COUNT));

            a1_0 = a1_0.add(q0.and(d1).lanewise(VectorOperators.BIT_COUNT));
            a1_1 = a1_1.add(q1.and(d1).lanewise(VectorOperators.BIT_COUNT));
            a1_2 = a1_2.add(q2.and(d1).lanewise(VectorOperators.BIT_COUNT));
            a1_3 = a1_3.add(q3.and(d1).lanewise(VectorOperators.BIT_COUNT));

            a2_0 = a2_0.add(q0.and(d2).lanewise(VectorOperators.BIT_COUNT));
            a2_1 = a2_1.add(q1.and(d2).lanewise(VectorOperators.BIT_COUNT));
            a2_2 = a2_2.add(q2.and(d2).lanewise(VectorOperators.BIT_COUNT));
            a2_3 = a2_3.add(q3.and(d2).lanewise(VectorOperators.BIT_COUNT));

            a3_0 = a3_0.add(q0.and(d3).lanewise(VectorOperators.BIT_COUNT));
            a3_1 = a3_1.add(q1.and(d3).lanewise(VectorOperators.BIT_COUNT));
            a3_2 = a3_2.add(q2.and(d3).lanewise(VectorOperators.BIT_COUNT));
            a3_3 = a3_3.add(q3.and(d3).lanewise(VectorOperators.BIT_COUNT));
        }

        long s0_0 = a0_0.reduceLanes(VectorOperators.ADD), s0_1 = a0_1.reduceLanes(VectorOperators.ADD);
        long s0_2 = a0_2.reduceLanes(VectorOperators.ADD), s0_3 = a0_3.reduceLanes(VectorOperators.ADD);
        long s1_0 = a1_0.reduceLanes(VectorOperators.ADD), s1_1 = a1_1.reduceLanes(VectorOperators.ADD);
        long s1_2 = a1_2.reduceLanes(VectorOperators.ADD), s1_3 = a1_3.reduceLanes(VectorOperators.ADD);
        long s2_0 = a2_0.reduceLanes(VectorOperators.ADD), s2_1 = a2_1.reduceLanes(VectorOperators.ADD);
        long s2_2 = a2_2.reduceLanes(VectorOperators.ADD), s2_3 = a2_3.reduceLanes(VectorOperators.ADD);
        long s3_0 = a3_0.reduceLanes(VectorOperators.ADD), s3_1 = a3_1.reduceLanes(VectorOperators.ADD);
        long s3_2 = a3_2.reduceLanes(VectorOperators.ADD), s3_3 = a3_3.reduceLanes(VectorOperators.ADD);

        for (; r < len; r++) {
            int q0 = query[r] & 0xFF;
            int q1 = query[r + len] & 0xFF;
            int q2 = query[r + len * 2] & 0xFF;
            int q3 = query[r + len * 3] & 0xFF;

            int d0 = docs.get(ValueLayout.JAVA_BYTE, off0 + r) & 0xFF;
            int d1 = docs.get(ValueLayout.JAVA_BYTE, off1 + r) & 0xFF;
            int d2 = docs.get(ValueLayout.JAVA_BYTE, off2 + r) & 0xFF;
            int d3 = docs.get(ValueLayout.JAVA_BYTE, off3 + r) & 0xFF;

            s0_0 += Integer.bitCount(q0 & d0); s0_1 += Integer.bitCount(q1 & d0);
            s0_2 += Integer.bitCount(q2 & d0); s0_3 += Integer.bitCount(q3 & d0);
            s1_0 += Integer.bitCount(q0 & d1); s1_1 += Integer.bitCount(q1 & d1);
            s1_2 += Integer.bitCount(q2 & d1); s1_3 += Integer.bitCount(q3 & d1);
            s2_0 += Integer.bitCount(q0 & d2); s2_1 += Integer.bitCount(q1 & d2);
            s2_2 += Integer.bitCount(q2 & d2); s2_3 += Integer.bitCount(q3 & d2);
            s3_0 += Integer.bitCount(q0 & d3); s3_1 += Integer.bitCount(q1 & d3);
            s3_2 += Integer.bitCount(q2 & d3); s3_3 += Integer.bitCount(q3 & d3);
        }

        results[ri0] = s0_0 + s0_1 * 2L + s0_2 * 4L + s0_3 * 8L;
        results[ri1] = s1_0 + s1_1 * 2L + s1_2 * 4L + s1_3 * 8L;
        results[ri2] = s2_0 + s2_1 * 2L + s2_2 * 4L + s2_3 * 8L;
        results[ri3] = s3_0 + s3_1 * 2L + s3_2 * 4L + s3_3 * 8L;
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
