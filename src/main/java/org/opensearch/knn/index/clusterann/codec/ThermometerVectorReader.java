/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * Flow B ("IVFaster absolute") posting reader / scan tier.
 *
 * <p>UNVERIFIED: written blind (compile-checked only). Mirrors {@link ThermometerVectorWriter}'s
 * block layout byte-for-byte; the round-trip has NOT been run on real data.
 *
 * <p>The scan reads ONLY the coarse thermometer planes and scores each document by Hamming distance
 * to the (rotated, thermometer-coded) query — {@link Nitrox2#hamming}. It skips the inert int8
 * block. It does NOT produce final scores: it fills the caller's ordinal + coarse-distance buffers
 * so the caller can shortlist and hand survivors to the existing exact rescorer.
 */
public final class ThermometerVectorReader {

    private final ClusterANNFieldState fieldState;
    private final int dimension;
    private final int coarseBytes;

    /** The query's thermometer code (set per query via {@link #prepareQuery}). */
    private final byte[] queryCoarse;

    // Block scratch
    private final byte[] coarseBlock;

    public ThermometerVectorReader(ClusterANNFieldState fieldState) {
        this.fieldState = fieldState;
        this.dimension = fieldState.dimension;
        this.coarseBytes = Nitrox2.bytesPerVector(dimension);
        this.queryCoarse = new byte[coarseBytes];
        this.coarseBlock = new byte[BLOCK_SIZE * coarseBytes];
    }

    /**
     * Quantize the query once: {@code rotatedQuery} is the Hadamard-rotated query vector. Packs its
     * thermometer code, reused across every block of every probed cell for this query.
     */
    public void prepareQuery(float[] rotatedQuery) {
        Nitrox2.packPlanes(rotatedQuery, dimension, queryCoarse, 0);
    }

    /** Bytes one block of {@code blockSize} vectors occupies on disk (must match the writer). */
    public long blockBytes(int blockSize) {
        return (long) blockSize * coarseBytes
            + (long) blockSize * dimension
            + (long) blockSize * Float.BYTES
            + (long) blockSize * Integer.BYTES
            + (long) blockSize * Float.BYTES;
    }

    /**
     * Coarse-scan one block from the current input position. Reads the block's coarse planes,
     * computes Hamming distance query-vs-doc into {@code hammingOut[0..blockSize)}, then advances
     * the input past the inert int8 region so the next block is positioned correctly.
     *
     * @return bytes consumed
     */
    public long scanBlockCoarse(IndexInput input, int blockSize, int[] hammingOut) throws IOException {
        input.readBytes(coarseBlock, 0, blockSize * coarseBytes);
        for (int j = 0; j < blockSize; j++) {
            hammingOut[j] = Nitrox2.hamming(queryCoarse, 0, coarseBlock, j * coarseBytes, coarseBytes);
        }
        // Skip the inert int8 block + corrections; final ranking uses the exact rescorer.
        long skip = (long) blockSize * dimension
            + (long) blockSize * Float.BYTES
            + (long) blockSize * Integer.BYTES
            + (long) blockSize * Float.BYTES;
        input.skipBytes(skip);
        return (long) blockSize * coarseBytes + skip;
    }

    /**
     * Coarse-scan one block AND buffer its int8 codes + corrections (for int8 rerank). Computes
     * Hamming into {@code hammingOut}, copies each doc's int8 code into {@code int8Out} at
     * {@code (posBase+j)*dimension}, and corrections into the parallel arrays at {@code posBase+j}.
     */
    public void scanBlockCoarseAndInt8(
        IndexInput input, int blockSize, int posBase, int[] hammingOut,
        byte[] int8Out, float[] scaleOut, int[] sumOut, float[] normOut
    ) throws IOException {
        input.readBytes(coarseBlock, 0, blockSize * coarseBytes);
        for (int j = 0; j < blockSize; j++) {
            hammingOut[j] = Nitrox2.hamming(queryCoarse, 0, coarseBlock, j * coarseBytes, coarseBytes);
        }
        // int8 codes
        input.readBytes(int8Out, posBase * dimension, blockSize * dimension);
        // corrections: scale[], sum[], norm[]
        for (int j = 0; j < blockSize; j++) scaleOut[posBase + j] = Float.intBitsToFloat(input.readInt());
        for (int j = 0; j < blockSize; j++) sumOut[posBase + j] = input.readInt();
        for (int j = 0; j < blockSize; j++) normOut[posBase + j] = Float.intBitsToFloat(input.readInt());
    }

    /**
     * Convenience: rescore a shortlisted ordinal with the existing exact scorer. Kept here so the
     * scan and rescore hand-off lives in one place; callers may also call the scorer directly.
     */
    public float rescore(RandomVectorScorer exactScorer, int ord) throws IOException {
        return exactScorer.score(ord);
    }

    // ---------------------------------------------------------------- int8 rerank tier

    private static final int OFFSET = 128;

    /** Query int8 code + per-query constants (set by {@link #prepareInt8Query}). */
    private byte[] qInt8;
    private int qSum;
    private double qSqNorm;
    private float qScale;

    /**
     * Quantize the rotated query into the int8 form (faithful to IVFaster Int8Quantizer), so the
     * int8 rerank can compute an unsigned dot against stored doc codes with an exact correction.
     */
    public void prepareInt8Query(float[] rotatedQuery) {
        this.qInt8 = new byte[dimension];
        float maxAbs = 0f;
        double sq = 0;
        for (int d = 0; d < dimension; d++) {
            float v = rotatedQuery[d];
            float a = Math.abs(v);
            if (a > maxAbs) maxAbs = a;
            sq += (double) v * v;
        }
        this.qSqNorm = sq;
        if (maxAbs == 0f) {
            java.util.Arrays.fill(qInt8, (byte) OFFSET);
            this.qScale = 1f;
            this.qSum = 0;
            return;
        }
        this.qScale = maxAbs / 127f;
        float inv = 127f / maxAbs;
        int sum = 0;
        for (int d = 0; d < dimension; d++) {
            int q = Math.round(rotatedQuery[d] * inv);
            if (q > 127) q = 127;
            else if (q < -127) q = -127;
            sum += q;
            qInt8[d] = (byte) (q + OFFSET);
        }
        this.qSum = sum;
    }

    /**
     * Int8 rerank score for one doc, from its stored code + corrections. Faithful to IVFaster:
     * recover the signed dot exactly, dequantize, then the similarity transform.
     *
     * @param code doc int8 code (unsigned-offset), length dimension
     * @param dScale doc scale correction
     * @param dSum doc signed code sum correction
     * @param dSqNorm doc true squared norm correction
     */
    public float int8Score(byte[] code, int codeOff, float dScale, int dSum, float dSqNorm, VectorSimilarityFunction sim) {
        long unsignedDot = 0;
        for (int d = 0; d < dimension; d++) {
            unsignedDot += (long) (qInt8[d] & 0xFF) * (code[codeOff + d] & 0xFF);
        }
        long offsetConst = (long) OFFSET * qSum + 16384L * dimension;
        long signedDot = unsignedDot - offsetConst - (long) OFFSET * (long) dSum;
        double dot = (double) signedDot * qScale * dScale;
        switch (sim) {
            case EUCLIDEAN: {
                double s = qSqNorm + dSqNorm - 2.0 * dot;
                if (s < 0) s = 0;
                return (float) (1.0 / (1.0 + s));
            }
            case MAXIMUM_INNER_PRODUCT:
                return dot >= 0 ? (float) (dot + 1.0) : (float) (1.0 / (1.0 - dot));
            default: // DOT_PRODUCT, COSINE
                return (float) Math.max(0.0, (1.0 + dot) / 2.0);
        }
    }

    /** Coarse bytes per vector (for callers computing block offsets). */
    public int coarseBytes() {
        return coarseBytes;
    }
}
