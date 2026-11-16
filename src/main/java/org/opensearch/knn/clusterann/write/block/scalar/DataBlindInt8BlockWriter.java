/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.block.scalar;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.quantization.DataBlindInt8Quantizer;

import java.io.IOException;

/**
 * The data-blind int8 block codec's write side: quantizes a cluster's already-rotation-prepared vectors
 * <em>absolutely</em> (no centroid residual) with {@link DataBlindInt8Quantizer} and appends fixed-size
 * blocks — the write counterpart of
 * {@link org.opensearch.knn.clusterann.read.block.scalar.datablind.DataBlindInt8BlockReader}.
 *
 * <p>The field's rotation (Hadamard, for every metric) is applied upstream by the posting writer's
 * {@code PreparedFloatVectorValues}, exactly as it is for the optimized-scalar family — so this writer sees
 * rotated vectors and never rotates. Unlike the optimized-scalar writer, nothing is centred on a centroid:
 * the scheme is absolute, the centroid is used only for routing. One unsigned byte per dimension, stored
 * whole (no bit-plane transpose, no nibble packing).
 *
 * <p>Each block is written struct-of-arrays, corrective terms first then the codes:
 * <pre>
 *   one block of n vectors (n = blockSize, last block may be partial):
 *     scale   n x int   (floatToIntBits of the per-vector maxAbs/127 scale)
 *     sum     n x int   (signed quantized component sum Σq)
 *     sqNorm  n x int   (floatToIntBits of ‖rotated vector‖²)
 *     codes   n x dim bytes (one unsigned byte per dimension)
 * </pre>
 * Blocks carry no header; the vector count comes from the sequence length recorded in {@code .clam}.
 */
public final class DataBlindInt8BlockWriter implements BlockVectorFormat.Writer {

    private final IndexOutput out;
    private final int blockSize;
    private final int dimension;
    private final DataBlindInt8Quantizer quantizer;

    private final byte[] code;                 // one vector's codes (dim bytes)

    private final float[] scale;
    private final int[] sum;
    private final float[] sqNorm;
    private final byte[] codes;                // one block's codes, blockSize * dim

    private int buffered;

    public DataBlindInt8BlockWriter(final IndexOutput out, final int blockSize, final int dimension) {
        this.out = out;
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.quantizer = new DataBlindInt8Quantizer(dimension);
        this.code = new byte[dimension];
        this.scale = new float[blockSize];
        this.sum = new int[blockSize];
        this.sqNorm = new float[blockSize];
        this.codes = new byte[blockSize * dimension];
    }

    @Override
    public int blockSize() {
        return blockSize;
    }

    @Override
    public void writeBlocks(final FloatVectorValues source) throws IOException {
        if (source.dimension() != dimension) {
            throw new IllegalArgumentException("source dimension " + source.dimension() + " != writer dimension " + dimension);
        }
        final int count = source.size();
        if (count == 0) {
            return;
        }
        buffered = 0;
        for (int ord = 0; ord < count; ord++) {
            buffer(source.vectorValue(ord));
            if (buffered == blockSize) {
                writeBlock();
            }
        }
        writeBlock(); // partial last block; a no-op when the count was a multiple of blockSize
    }

    /** Quantize one already-rotated vector (absolute, no centroid) into the next buffer slot. */
    private void buffer(final float[] rotated) {
        final DataBlindInt8Quantizer.Int8Code c = quantizer.encode(rotated, code);
        scale[buffered] = c.scale();
        sum[buffered] = c.sum();
        sqNorm[buffered] = c.sqNorm();
        System.arraycopy(code, 0, codes, buffered * dimension, dimension);
        buffered++;
    }

    private void writeBlock() throws IOException {
        if (buffered < 1) {
            return;
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(Float.floatToIntBits(scale[j]));
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(sum[j]);
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(Float.floatToIntBits(sqNorm[j]));
        }
        out.writeBytes(codes, 0, buffered * dimension);
        buffered = 0;
    }
}
