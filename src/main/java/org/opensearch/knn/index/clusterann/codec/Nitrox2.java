/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * The nitrox2 coarse tier: a symmetric 3-level thermometer code over two bit planes, 2 bits/dim.
 *
 * <p>Data-independent (leaf-free): the grid thresholds are a function of {@code dim} alone, valid
 * because {@link org.opensearch.knn.index.clusterann.algorithm.HadamardRotation} makes every rotated
 * dimension's standard deviation analytically {@code 1/sqrt(dim)}. There is no per-centroid and no
 * per-segment fitted statistic, so a document's code never depends on which cluster it lands in.
 *
 * <p>The thermometer identity {@code popcount(q ^ d) == sum_d |level_q[d] - level_d[d]|} holds
 * EXACTLY, so scoring is one XOR + popcount over the concatenated planes and there is no
 * per-document correction term. Both the document and the query encode through {@link #level}, so a
 * writer and reader cannot disagree on the grid.
 *
 * <p>Faithful reproduction of the Lucene sandbox {@code ivfaster.Nitrox2} at the default 2-bit
 * setting (LEVELS = 3, PLANES = 2).
 */
public final class Nitrox2 {

    /** Bits per dimension = plane count. Fixed at 2 (3-level thermometer). */
    public static final int PLANES = 2;

    /** Thermometer levels: one more than the bit count. */
    public static final int LEVELS = PLANES + 1;

    /** Grid half-width, in units of the analytic per-dimension std {@code 1/sqrt(dim)}. */
    private static final float CLIP_SIGMA = 1.0f;

    private Nitrox2() {}

    /** Bytes per plane for a given dimension. */
    public static int planeBytes(int dim) {
        return (dim + 7) >>> 3;
    }

    /** Total coarse bytes per vector: one plane per level boundary. */
    public static int bytesPerVector(int dim) {
        return PLANES * planeBytes(dim);
    }

    /** Grid half-width for {@code dim}: {@code CLIP_SIGMA / sqrt(dim)}. */
    public static float clipFor(int dim) {
        return (float) (CLIP_SIGMA / Math.sqrt(dim));
    }

    /**
     * Threshold at which thermometer plane {@code t} fires, in units of the rotated coordinate.
     *
     * <p>Derived from {@code L >= t+1  <=>  v >= ((2t+1)/(LEVELS-1) - 1) * clip}. This exact
     * comparison form (not an affine map + round) avoids ULP-scale rounding that would be
     * format-affecting.
     */
    public static float thresholdFor(int t, float clip) {
        return ((2f * t + 1f) / (LEVELS - 1) - 1f) * clip;
    }

    /**
     * The thermometer level in {@code [0, LEVELS)} for one rotated coordinate. The single source of
     * truth for the coarse grid; document, centroid, and query encoding all route through here.
     */
    public static int level(float v, float clip) {
        int l = 0;
        for (int t = 0; t < LEVELS - 1; t++) {
            if (v >= thresholdFor(t, clip)) {
                l++;
            }
        }
        return l;
    }

    /**
     * Encodes {@code vector[0..dim)} into {@code PLANES} consecutive thermometer planes at
     * {@code destOff}, each {@code planeBytes(dim)} long. Plane {@code t} carries the bit
     * {@code (level > t)} for each dimension, little-endian within each byte.
     *
     * <p>{@code dest} must have room for {@code destOff + bytesPerVector(dim)} bytes; the plane
     * region is zeroed by this method before bits are set.
     */
    public static void packPlanes(float[] vector, int dim, byte[] dest, int destOff) {
        final float clip = clipFor(dim);
        final int pb = planeBytes(dim);
        // Zero the plane region.
        java.util.Arrays.fill(dest, destOff, destOff + PLANES * pb, (byte) 0);
        for (int d = 0; d < dim; d++) {
            final int lvl = level(vector[d], clip);
            // Thermometer: plane t bit set iff lvl > t.
            for (int t = 0; t < PLANES; t++) {
                if (lvl > t) {
                    final int planeBase = destOff + t * pb;
                    dest[planeBase + (d >>> 3)] |= (byte) (1 << (d & 7));
                }
            }
        }
    }

    /**
     * Hamming distance between two packed codes of {@code coarseBytes} length: {@code
     * popcount(a XOR b)} over the concatenated planes. Smaller is nearer. This IS the summed
     * per-dimension level distance, exactly.
     */
    public static int hamming(byte[] a, int aOff, byte[] b, int bOff, int coarseBytes) {
        int dist = 0;
        int i = 0;
        // Word-at-a-time over 8-byte chunks using Long.bitCount (pure Java, no SIMD).
        final int limit = coarseBytes - 7;
        for (; i < limit; i += 8) {
            long wa = readLongLE(a, aOff + i);
            long wb = readLongLE(b, bOff + i);
            dist += Long.bitCount(wa ^ wb);
        }
        // Tail bytes.
        for (; i < coarseBytes; i++) {
            dist += Integer.bitCount((a[aOff + i] ^ b[bOff + i]) & 0xFF);
        }
        return dist;
    }

    private static long readLongLE(byte[] arr, int off) {
        return (arr[off] & 0xFFL)
            | (arr[off + 1] & 0xFFL) << 8
            | (arr[off + 2] & 0xFFL) << 16
            | (arr[off + 3] & 0xFFL) << 24
            | (arr[off + 4] & 0xFFL) << 32
            | (arr[off + 5] & 0xFFL) << 40
            | (arr[off + 6] & 0xFFL) << 48
            | (arr[off + 7] & 0xFFL) << 56;
    }
}
