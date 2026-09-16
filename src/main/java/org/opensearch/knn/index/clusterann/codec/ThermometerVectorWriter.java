/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexOutput;

import java.io.Closeable;
import java.io.IOException;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * Flow B ("IVFaster absolute") posting writer.
 *
 * <p>UNVERIFIED: written blind (compile-checked only). The write/read round-trip has NOT been
 * exercised on real data; {@link ThermometerVectorReader} must mirror this layout byte-for-byte.
 *
 * <p>Documents are quantized ABSOLUTELY (no per-centroid residual): the caller supplies the already
 * Hadamard-rotated vector, from which this writer derives the 2-bit {@link Nitrox2} thermometer
 * coarse code (the scan tier) and an inert 8-bit int8 fine code (stored per the design, not on the
 * scan hot path — final ranking uses the existing exact rescorer over raw vectors).
 *
 * <h2>Block layout (BLOCK_SIZE=32 vectors per block)</h2>
 * <pre>
 *   coarse[0..blockSize)  : each Nitrox2.bytesPerVector(dim) bytes, contiguous  (scan tier)
 *   int8Code[0..blockSize): each dim bytes (unsigned-offset), contiguous        (inert fine tier)
 *   int8Scale[0..blockSize]: float per vector                                   (fine corrections)
 *   int8Sum[0..blockSize]:   int per vector (signed code sum)
 *   int8Norm[0..blockSize]:  float per vector (true squared norm)
 * </pre>
 * The coarse planes lead so the scan reads only them ({@code Nitrox2.bytesPerVector}, ~1/16 of the
 * int8) before deciding a shortlist; the int8 block follows and is skipped by the scan.
 */
public final class ThermometerVectorWriter implements Closeable {

    private static final byte INT8_OFFSET = (byte) 128;

    private final int dimension;
    private final int coarseBytes;

    // Reusable block scratch
    private final byte[] coarseBlock;
    private final byte[] int8Block;
    private final float[] scaleBlock;
    private final int[] sumBlock;
    private final float[] normBlock;

    public ThermometerVectorWriter(int dimension) {
        this.dimension = dimension;
        this.coarseBytes = Nitrox2.bytesPerVector(dimension);
        this.coarseBlock = new byte[BLOCK_SIZE * coarseBytes];
        this.int8Block = new byte[BLOCK_SIZE * dimension];
        this.scaleBlock = new float[BLOCK_SIZE];
        this.sumBlock = new int[BLOCK_SIZE];
        this.normBlock = new float[BLOCK_SIZE];
    }

    /** Bytes one block of {@code blockSize} vectors occupies on disk. */
    public long blockBytes(int blockSize) {
        return (long) blockSize * coarseBytes                 // coarse planes
            + (long) blockSize * dimension                     // int8 codes
            + (long) blockSize * Float.BYTES                   // scale
            + (long) blockSize * Integer.BYTES                 // signed sum
            + (long) blockSize * Float.BYTES;                  // squared norm
    }

    /**
     * Write a posting list's vectors in flow-B block-columnar format. {@code vectors.get(ord)} MUST
     * return the already Hadamard-rotated vector (the caller rotates; this writer never touches the
     * centroid — codes are absolute).
     */
    public void writeBlocked(int[] ordinals, int count, VectorSupplier vectors, IndexOutput output) throws IOException {
        int pos = 0;
        while (pos < count) {
            int blockSize = Math.min(BLOCK_SIZE, count - pos);
            for (int j = 0; j < blockSize; j++) {
                float[] rotated = vectors.get(ordinals[pos + j]);
                encodeOne(rotated, j);
            }
            // Coarse planes first (scan tier).
            output.writeBytes(coarseBlock, 0, blockSize * coarseBytes);
            // Inert int8 codes + corrections.
            output.writeBytes(int8Block, 0, blockSize * dimension);
            for (int j = 0; j < blockSize; j++) output.writeInt(Float.floatToIntBits(scaleBlock[j]));
            for (int j = 0; j < blockSize; j++) output.writeInt(sumBlock[j]);
            for (int j = 0; j < blockSize; j++) output.writeInt(Float.floatToIntBits(normBlock[j]));
            pos += blockSize;
        }
    }

    private void encodeOne(float[] rotated, int idx) {
        // Coarse 2-bit thermometer (absolute, data-blind grid).
        Nitrox2.packPlanes(rotated, dimension, coarseBlock, idx * coarseBytes);

        // Inert 8-bit int8 (per-vector max-abs scale, unsigned-offset). Faithful to Int8Quantizer.
        float maxAbs = 0f;
        double sqNorm = 0;
        for (int d = 0; d < dimension; d++) {
            float v = rotated[d];
            float a = Math.abs(v);
            if (a > maxAbs) maxAbs = a;
            sqNorm += (double) v * v;
        }
        int base = idx * dimension;
        if (maxAbs == 0f) {
            for (int d = 0; d < dimension; d++) int8Block[base + d] = INT8_OFFSET;
            scaleBlock[idx] = 1f;
            sumBlock[idx] = 0;
            normBlock[idx] = 0f;
            return;
        }
        float scale = maxAbs / 127f;
        float inv = 127f / maxAbs;
        int sum = 0;
        for (int d = 0; d < dimension; d++) {
            int q = Math.round(rotated[d] * inv);
            if (q > 127) q = 127;
            else if (q < -127) q = -127;
            sum += q;
            int8Block[base + d] = (byte) (q + 128);
        }
        scaleBlock[idx] = scale;
        sumBlock[idx] = sum;
        normBlock[idx] = (float) sqNorm;
    }

    @Override
    public void close() {}

    @FunctionalInterface
    public interface VectorSupplier {
        float[] get(int ordinal) throws IOException;
    }
}
