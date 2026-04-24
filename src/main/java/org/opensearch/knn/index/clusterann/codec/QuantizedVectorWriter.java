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
 * <p>Block layout (BLOCK_SIZE=32):
 * <pre>
 *   codes[0..31]   ← contiguous for SIMD scoreBulk
 *   lower[0..31]   ← bulk readFloats
 *   upper[0..31]   ← bulk readFloats
 *   add[0..31]     ← bulk readFloats
 *   sum[0..31]     ← bulk readInts
 * </pre>
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
    private final byte[][] blockCodes;
    private final float[] blockLower;
    private final float[] blockUpper;
    private final float[] blockAdd;
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

        this.blockCodes = new byte[BLOCK_SIZE][packedBytes];
        this.blockLower = new float[BLOCK_SIZE];
        this.blockUpper = new float[BLOCK_SIZE];
        this.blockAdd = new float[BLOCK_SIZE];
        this.blockSum = new int[BLOCK_SIZE];
    }

    /** Bytes per block of blockSize vectors. */
    public long blockBytes(int blockSize) {
        return (long) blockSize * packedBytes + (long) blockSize * Float.BYTES * 3 + (long) blockSize * Integer.BYTES;
    }

    /**
     * Write a posting list's quantized vectors in block-columnar format.
     * Caller provides ordinals and a way to get vectors.
     */
    public void writeBlocked(int[] ordinals, int count, VectorSupplier vectors, float[] centroid, IndexOutput output) throws IOException {
        System.arraycopy(centroid, 0, centroidCopy, 0, dimension);
        if (simFunc == VectorSimilarityFunction.COSINE) {
            normalize(centroidCopy);
        }

        int pos = 0;
        while (pos < count) {
            int blockSize = Math.min(BLOCK_SIZE, count - pos);

            // Quantize block
            for (int j = 0; j < blockSize; j++) {
                float[] vec = vectors.get(ordinals[pos + j]);
                quantizeOne(vec, centroidCopy, blockCodes[j], j);
            }

            // Write columnar: codes (one bulk read), then corrections (bulk each)
            for (int j = 0; j < blockSize; j++) {
                output.writeBytes(blockCodes[j], 0, packedBytes);
            }
            writeFloats(output, blockLower, blockSize);
            writeFloats(output, blockUpper, blockSize);
            writeFloats(output, blockAdd, blockSize);
            writeInts(output, blockSum, blockSize);

            pos += blockSize;
        }
    }

    private void quantizeOne(float[] vector, float[] centroid, byte[] codesOut, int idx) {
        System.arraycopy(vector, 0, vectorCopy, 0, dimension);
        if (simFunc == VectorSimilarityFunction.COSINE) {
            normalize(vectorCopy);
        }

        Arrays.fill(scratch, (byte) 0);
        OptimizedScalarQuantizer.QuantizationResult result = osq.multiScalarQuantize(vectorCopy, destinations, bitsArray, centroid)[0];

        Arrays.fill(codesOut, (byte) 0);
        if (docBits == 1) {
            OptimizedScalarQuantizer.packAsBinary(scratch, codesOut);
        } else if (docBits == 2) {
            OptimizedScalarQuantizer.transposeDibit(scratch, codesOut);
        } else {
            OptimizedScalarQuantizer.transposeHalfByte(scratch, codesOut);
        }

        blockLower[idx] = result.lowerInterval();
        blockUpper[idx] = result.upperInterval();
        blockAdd[idx] = result.additionalCorrection();
        blockSum[idx] = result.quantizedComponentSum();
    }

    @Override
    public void close() {}

    @FunctionalInterface
    public interface VectorSupplier {
        float[] get(int ordinal) throws IOException;
    }

    private static void writeFloats(IndexOutput out, float[] values, int count) throws IOException {
        for (int i = 0; i < count; i++)
            out.writeInt(Float.floatToIntBits(values[i]));
    }

    private static void writeInts(IndexOutput out, int[] values, int count) throws IOException {
        for (int i = 0; i < count; i++)
            out.writeInt(values[i]);
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
