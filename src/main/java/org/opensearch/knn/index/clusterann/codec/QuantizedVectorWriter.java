/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * Writes quantized vectors in block-columnar format for SIMD-friendly scoring.
 *
 * <p>Block layout (BLOCK_SIZE=32) — corrections first (so search can early-skip a block from
 * the corrections alone), codes last (read only if the block is competitive):
 * <pre>
 *   lower[0..31]   ← bulk writeInts
 *   upper[0..31]   ← bulk writeInts
 *   add[0..31]     ← bulk writeInts
 *   sum[0..31]     ← bulk writeInts
 *   codes[0..31]   ← contiguous packed codes for SIMD scoreBulk
 * </pre>
 *
 * <p><b>{@code add} vs the posting's {@code sortedDistances} column (L2 redundancy, kept on purpose).</b>
 * {@code add} is {@link OptimizedScalarQuantizer.QuantizationResult#additionalCorrection()}: for L2 it is
 * {@code ‖v−c‖²}, for IP it is the centroid dot {@code ⟨v,c⟩}. The posting header separately stores
 * {@code sortedDistances[i] = ‖c−v‖} (raw space, original centroid). For <em>L2</em> these are the same
 * quantity — {@code add == sortedDistances²} — exactly, not approximately: the rotation is orthonormal
 * and {@code transformedCentroid = R·originalCentroid}, so the rotated residual keeps its length
 * ({@code ‖Rv−Rc‖ = ‖v−c‖}). We could therefore derive one from the other and drop {@code add} for L2.
 * We deliberately don't: for IP {@code add} is a dot product with no relation to {@code ‖c−v‖}, so
 * dropping it would make the block layout and the scorer's correction-read branch by metric, couple
 * block scoring back to the header column, and square-then-widen the float error on the term that feeds
 * the L2 score. The ~4 bytes/vector saved isn't worth that; the layout stays metric-uniform.
 */
public final class QuantizedVectorWriter implements Closeable {

    private final OptimizedScalarQuantizer osq;
    private final VectorSimilarityFunction simFunc;
    private final byte docBits;
    private final int dimension;
    private final int packedBytes;

    // Reusable scratch
    private final byte[] scratch;
    private final byte[] packed;
    private final byte[][] destinations;
    private final byte[] bitsArray;
    private final float[] vectorCopy;
    private final float[] centroidCopy;

    // Block buffers
    private final byte[] flatCodesBuf;
    private final int[] blockLowerBits;
    private final int[] blockUpperBits;
    private final int[] blockAddBits;
    private final int[] blockSum;

    public QuantizedVectorWriter(VectorSimilarityFunction simFunc, int dimension, byte docBits) {
        this.osq = new OptimizedScalarQuantizer(simFunc);
        this.simFunc = simFunc;
        this.docBits = docBits;
        this.dimension = dimension;
        this.packedBytes = packedBytesPerVector(dimension, docBits);
        this.scratch = new byte[dimension];
        this.packed = new byte[packedBytes];
        this.destinations = new byte[][] { scratch };
        this.bitsArray = new byte[] { docBits };
        this.vectorCopy = new float[dimension];
        this.centroidCopy = new float[dimension];

        this.flatCodesBuf = new byte[BLOCK_SIZE * packedBytes];
        this.blockLowerBits = new int[BLOCK_SIZE];
        this.blockUpperBits = new int[BLOCK_SIZE];
        this.blockAddBits = new int[BLOCK_SIZE];
        this.blockSum = new int[BLOCK_SIZE];
    }

    /**
     * Write a posting list's quantized vectors in block-columnar format.
     */
    public void writeBlocked(int[] ordinals, int count, VectorSupplier vectors, float[] centroid, IndexOutput output) throws IOException {
        System.arraycopy(centroid, 0, centroidCopy, 0, dimension);
        if (simFunc == VectorSimilarityFunction.COSINE) {
            normalize(centroidCopy);
        }

        int pos = 0;
        while (pos < count) {
            int blockSize = Math.min(BLOCK_SIZE, count - pos);

            // Quantize block into flat buffer + correction arrays
            for (int j = 0; j < blockSize; j++) {
                float[] vec = vectors.get(ordinals[pos + j]);
                quantizeOne(vec, centroidCopy, j);
            }

            // Write corrections FIRST (enables early skip during search)
            for (int j = 0; j < blockSize; j++)
                output.writeInt(blockLowerBits[j]);
            for (int j = 0; j < blockSize; j++)
                output.writeInt(blockUpperBits[j]);
            for (int j = 0; j < blockSize; j++)
                output.writeInt(blockAddBits[j]);
            for (int j = 0; j < blockSize; j++)
                output.writeInt(blockSum[j]);
            // Codes last (can be skipped if block is non-competitive)
            output.writeBytes(flatCodesBuf, 0, blockSize * packedBytes);

            pos += blockSize;
        }
    }

    private void quantizeOne(float[] vector, float[] centroid, int idx) {
        System.arraycopy(vector, 0, vectorCopy, 0, dimension);
        if (simFunc == VectorSimilarityFunction.COSINE) {
            normalize(vectorCopy);
        }

        Arrays.fill(scratch, (byte) 0);
        OptimizedScalarQuantizer.QuantizationResult result = osq.multiScalarQuantize(vectorCopy, destinations, bitsArray, centroid)[0];

        // Pack directly into flat buffer at correct offset
        int offset = idx * packedBytes;
        Arrays.fill(flatCodesBuf, offset, offset + packedBytes, (byte) 0);
        if (docBits == 1) {
            OptimizedScalarQuantizer.packAsBinary(scratch, packed);
        } else if (docBits == 2) {
            OptimizedScalarQuantizer.transposeDibit(scratch, packed);
        } else {
            OptimizedScalarQuantizer.transposeHalfByte(scratch, packed);
        }
        System.arraycopy(packed, 0, flatCodesBuf, offset, packedBytes);

        blockLowerBits[idx] = Float.floatToIntBits(result.lowerInterval());
        blockUpperBits[idx] = Float.floatToIntBits(result.upperInterval());
        blockAddBits[idx] = Float.floatToIntBits(result.additionalCorrection());
        blockSum[idx] = result.quantizedComponentSum();
    }

    @Override
    public void close() {}

    @FunctionalInterface
    public interface VectorSupplier {
        float[] get(int ordinal) throws IOException;
    }

    private static void normalize(float[] vec) {
        float norm = 0f;
        for (float v : vec)
            norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0f) {
            for (int d = 0; d < vec.length; d++)
                vec[d] /= norm;
        }
    }

    public static int packedBytesPerVector(int dimension, int bits) {
        if (bits == 1) return (dimension + 7) / 8;
        if (bits == 2) return ((dimension + 7) / 8) * 2;
        return ((dimension + 7) / 8) * 4;
    }
}
