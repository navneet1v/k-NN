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

    /**
     * Write-time toggle: when false, the inert int8 fine tier (code + scale + sum + norm) is NOT
     * written, leaving only the coarse thermometer planes on disk. Set via
     * {@code -Dclusterann.thermo.storeInt8=false}. MUST match the reader's flag for the index that
     * is queried — the block layout differs, so an index written with one setting can only be read
     * with the same setting. Default true (unchanged legacy layout).
     */
    static final boolean STORE_INT8 =
        Boolean.parseBoolean(System.getProperty("clusterann.thermo.storeInt8", "true"));

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
        long bytes = (long) blockSize * coarseBytes;          // coarse planes
        if (STORE_INT8) {
            bytes += (long) blockSize * dimension             // int8 codes
                + (long) blockSize * Float.BYTES              // scale
                + (long) blockSize * Integer.BYTES            // signed sum
                + (long) blockSize * Float.BYTES;             // squared norm
        }
        return bytes;
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
            if (STORE_INT8) {
                // Inert int8 codes + corrections.
                output.writeBytes(int8Block, 0, blockSize * dimension);
                for (int j = 0; j < blockSize; j++) output.writeInt(Float.floatToIntBits(scaleBlock[j]));
                for (int j = 0; j < blockSize; j++) output.writeInt(sumBlock[j]);
                for (int j = 0; j < blockSize; j++) output.writeInt(Float.floatToIntBits(normBlock[j]));
            }
            pos += blockSize;
        }
    }

    private void encodeOne(float[] rotated, int idx) {
        // Coarse 2-bit thermometer (absolute, data-blind grid).
        Nitrox2.packPlanes(rotated, dimension, coarseBlock, idx * coarseBytes);

        // Skip the inert int8 tier entirely when not storing it.
        if (!STORE_INT8) return;

        // Inert 8-bit int8. Two interval choices:
        //  - default: per-vector max-abs (scale = maxAbs/127), faithful to IVFaster Int8Quantizer.
        //  - OSQ8 (clusterann.thermo.osq8=true): per-vector MSE-grid-optimized SYMMETRIC interval
        //    (Lucene OptimizedScalarQuantizer): init half-width = MSE_GRID[7]*std, refine by
        //    anisotropic coordinate descent, then scale = halfWidth/127. Symmetric so the existing
        //    signed-dot int8Score reconstruction stays valid; the win is a tighter, MSE-optimal step.
        float scale;
        double sqNorm = 0;
        for (int d = 0; d < dimension; d++) sqNorm += (double) rotated[d] * rotated[d];
        if (OSQ8) {
            scale = osqSymmetricScale(rotated, dimension) ;
        } else {
            float maxAbs = 0f;
            for (int d = 0; d < dimension; d++) { float a = Math.abs(rotated[d]); if (a > maxAbs) maxAbs = a; }
            scale = maxAbs / 127f;
        }
        int base = idx * dimension;
        if (scale <= 0f) {
            for (int d = 0; d < dimension; d++) int8Block[base + d] = INT8_OFFSET;
            scaleBlock[idx] = 1f;
            sumBlock[idx] = 0;
            normBlock[idx] = 0f;
            return;
        }
        float inv = 1f / scale;
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

    /** OSQ toggle: use MSE-grid-optimized symmetric interval for the int8 tier. */
    static final boolean OSQ8 = Boolean.getBoolean("clusterann.thermo.osq8");
    private static final float MSE_GRID_8BIT = 3.922f;   // Lucene MINIMUM_MSE_GRID[7] half-width (in std units)
    private static final float OSQ_LAMBDA = 0.1f;
    private static final int OSQ_ITERS = 5;

    /**
     * Symmetric OSQ interval half-width for an 8-bit code, faithful to Lucene
     * OptimizedScalarQuantizer but constrained to a symmetric [-h, h] interval so the existing
     * signed int8 scorer stays valid. Returns scale = h / 127.
     */
    private static float osqSymmetricScale(float[] v, int dim) {
        double mean = 0, var = 0, norm2 = 0, maxAbs = 0;
        for (int i = 0; i < dim; i++) {
            double x = v[i]; norm2 += x * x; double a = Math.abs(x); if (a > maxAbs) maxAbs = a;
            double delta = x - mean; mean += delta / (i + 1); var += delta * (x - mean);
        }
        var /= dim; double std = Math.sqrt(var);
        if (norm2 == 0 || maxAbs == 0) return (float) (maxAbs / 127.0);
        double h = Math.min(MSE_GRID_8BIT * std, maxAbs);   // symmetric init half-width, clamped to range
        final int points = 256;
        double scale = (1.0 - OSQ_LAMBDA) / norm2;
        if (Double.isFinite(scale)) {
            double curLoss = symLoss(v, dim, h, points, norm2);
            for (int it = 0; it < OSQ_ITERS; it++) {
                // Coordinate descent on symmetric half-width h: value = h*(2k/(P-1) - 1).
                double stepInv = (points - 1.0) / (2.0 * h);
                double num = 0, den = 0;
                for (int i = 0; i < dim; i++) {
                    double xi = v[i];
                    double cx = Math.max(-h, Math.min(h, xi));
                    double k = Math.round((cx + h) * stepInv);
                    double s = 2.0 * k / (points - 1.0) - 1.0;   // reconstruction unit in [-1,1]
                    num += xi * s; den += s * s;
                }
                if (den == 0) break;
                double hOpt = num / den;
                if (hOpt <= 0 || Math.abs(hOpt - h) < 1e-8) break;
                double nl = symLoss(v, dim, hOpt, points, norm2);
                if (nl > curLoss) break;
                h = hOpt; curLoss = nl;
            }
        }
        return (float) (h / 127.0);
    }

    private static double symLoss(float[] v, int dim, double h, int points, double norm2) {
        double step = (2.0 * h) / (points - 1.0); if (step == 0) return Double.MAX_VALUE;
        double xe = 0, e = 0;
        for (int i = 0; i < dim; i++) {
            double xi = v[i];
            double cx = Math.max(-h, Math.min(h, xi));
            double xq = -h + step * Math.round((cx + h) / step);
            xe += xi * (xi - xq); e += (xi - xq) * (xi - xq);
        }
        return (1.0 - OSQ_LAMBDA) * xe * xe / norm2 + OSQ_LAMBDA * e;
    }

    @Override
    public void close() {}

    @FunctionalInterface
    public interface VectorSupplier {
        float[] get(int ordinal) throws IOException;
    }
}
