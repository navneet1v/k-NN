/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Random;

/**
 * Randomized orthogonal rotation built from random sign flips, a random permutation, and a
 * block-diagonal Fast Walsh-Hadamard Transform (FWHT): {@code R = F * P * S}.
 *
 * <p>Drop-in sibling of {@link RandomRotation} (same {@code transform} / {@code write} / {@code read}
 * shape) but {@code O(dim log dim)} to apply and {@code O(1)} to persist — only {@code (dim, seed)}
 * are stored, and the rotation is rebuilt from them on read.
 *
 * <p>Each factor is orthogonal, so {@code R} is orthogonal: it preserves L2 norm and dot products,
 * and its inverse is its transpose. The rotation is fully determined by {@code (dim, seed)} and is
 * immutable and thread-safe.
 *
 * <p>Purpose: "Gaussianize" the per-dimension value distribution so every rotated dimension has an
 * analytic standard deviation of {@code 1/sqrt(dim)}. This is the property the data-independent
 * thermometer grid ({@link org.opensearch.knn.index.clusterann.codec.Nitrox2}) relies on: fixed
 * quantization thresholds are valid for any data without per-segment fitting.
 */
public final class HadamardRotation {

    /** Default seed, matching {@link RandomRotation}'s deterministic build. */
    private static final long DEFAULT_SEED = 42L;

    /**
     * Cache of built rotations, keyed by (dim, seed). The rotation is a pure function of
     * (dim, seed) and this class is immutable and thread-safe, so one instance is shared across
     * every segment and query. This avoids rebuilding the sign flips, permutation, and block
     * decomposition on every per-segment query — the rotation is identical for all segments of a
     * given dimension, so it is built once (shard/process level) and reused.
     */
    private static final java.util.concurrent.ConcurrentHashMap<Long, HadamardRotation> CACHE =
        new java.util.concurrent.ConcurrentHashMap<>();

    private static long cacheKey(int dimension, long seed) {
        // seed is fixed (DEFAULT_SEED) in practice; fold both into one key defensively.
        return (((long) dimension) << 20) ^ (seed * 0x9E3779B97F4A7C15L);
    }

    private final int dim;
    private final long seed;

    /** Random +/-1 per dimension (the diagonal of {@code S}). */
    private final float[] signs;

    /** Fisher-Yates permutation; {@code perm[i]} is the source index gathered into position {@code i}. */
    private final int[] perm;

    /** Start offset of each power-of-two FWHT block. */
    private final int[] blockOffsets;

    /** Length of each FWHT block (a power of two). */
    private final int[] blockLengths;

    private HadamardRotation(int dim, long seed, float[] signs, int[] perm, int[] blockOffsets, int[] blockLengths) {
        this.dim = dim;
        this.seed = seed;
        this.signs = signs;
        this.perm = perm;
        this.blockOffsets = blockOffsets;
        this.blockLengths = blockLengths;
    }

    /** Create the rotation for the given dimension with the default seed. */
    public static HadamardRotation create(int dimension) {
        return create(dimension, DEFAULT_SEED);
    }

    /**
     * Build the rotation for {@code dimension}, deterministically seeded. The same {@code (dim, seed)}
     * always produces the same rotation. Cached and shared across segments/queries: the rotation is
     * identical for every segment of a given dimension, so it is built once and reused rather than
     * reconstructed on each per-segment query.
     */
    public static HadamardRotation create(int dimension, long seed) {
        return CACHE.computeIfAbsent(cacheKey(dimension, seed), k -> build(dimension, seed));
    }

    /** Builds a fresh rotation (uncached). See {@link #create(int, long)} for the cached entry point. */
    private static HadamardRotation build(int dimension, long seed) {
        if (dimension < 1) {
            throw new IllegalArgumentException("dim must be >= 1, got " + dimension);
        }
        Random random = new Random(seed);

        float[] signs = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            signs[i] = random.nextBoolean() ? 1f : -1f;
        }

        // Fisher-Yates permutation.
        int[] perm = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            perm[i] = i;
        }
        for (int i = dimension - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }

        // Decompose dim into power-of-two blocks (its set bits), largest first.
        int numBlocks = Integer.bitCount(dimension);
        int[] blockOffsets = new int[numBlocks];
        int[] blockLengths = new int[numBlocks];
        int offset = 0;
        int b = 0;
        for (int bit = Integer.highestOneBit(dimension); bit != 0; bit >>>= 1) {
            if ((dimension & bit) != 0) {
                blockOffsets[b] = offset;
                blockLengths[b] = bit;
                offset += bit;
                b++;
            }
        }
        assert offset == dimension;

        return new HadamardRotation(dimension, seed, signs, perm, blockOffsets, blockLengths);
    }

    /** The dimension this rotation operates on. */
    public int dimension() {
        return dim;
    }

    /**
     * Apply the forward rotation {@code out = R * in}. {@code in} is not modified; {@code in} and
     * {@code out} must both have length {@link #dimension()} and must not be the same array.
     */
    public void transform(float[] in, float[] out) {
        checkArgs(in, out);
        // Fused sign-flip + permutation gather: out[i] = sign[perm[i]] * in[perm[i]].
        for (int i = 0; i < dim; i++) {
            int src = perm[i];
            out[i] = signs[src] * in[src];
        }
        // Block-diagonal normalized FWHT, in place on out.
        for (int blk = 0; blk < blockOffsets.length; blk++) {
            fwht(out, blockOffsets[blk], blockLengths[blk]);
        }
    }

    /** Transform in-place (allocates a temp buffer), mirroring {@link RandomRotation#transformInPlace}. */
    public void transformInPlace(float[] vector) {
        float[] out = new float[vector.length];
        transform(vector, out);
        System.arraycopy(out, 0, vector, 0, vector.length);
    }

    /**
     * Apply the inverse rotation {@code out = R^T * in}. {@code in} is not modified; {@code in} and
     * {@code out} must both have length {@link #dimension()} and must not be the same array.
     */
    public void inverseTransform(float[] in, float[] out) {
        checkArgs(in, out);
        // F is symmetric and its own inverse (normalized), so apply it first, on a copy.
        float[] f = new float[dim];
        System.arraycopy(in, 0, f, 0, dim);
        for (int blk = 0; blk < blockOffsets.length; blk++) {
            fwht(f, blockOffsets[blk], blockLengths[blk]);
        }
        // Inverse permutation then sign-flip (scatter): out[perm[i]] = sign[perm[i]] * f[i].
        for (int i = 0; i < dim; i++) {
            int dst = perm[i];
            out[dst] = signs[dst] * f[i];
        }
    }

    /** Persist only {@code (dim, seed)}; the rotation is rebuilt from them on {@link #read}. */
    public void write(IndexOutput out) throws IOException {
        out.writeInt(dim);
        out.writeLong(seed);
    }

    /** Rebuild the rotation from the persisted {@code (dim, seed)}. */
    public static HadamardRotation read(IndexInput in) throws IOException {
        int dim = in.readInt();
        long seed = in.readLong();
        return create(dim, seed);
    }

    private void checkArgs(float[] in, float[] out) {
        if (in.length != dim || out.length != dim) {
            throw new IllegalArgumentException(
                "in/out length must equal dim=" + dim + ", got in=" + in.length + " out=" + out.length
            );
        }
        if (in == out) {
            throw new IllegalArgumentException("in and out must be different arrays");
        }
    }

    /**
     * In-place normalized Fast Walsh-Hadamard Transform over {@code a[offset, offset+len)}, where
     * {@code len} is a power of two. Normalized by {@code 1/sqrt(len)} so the transform is orthogonal
     * and self-inverse. A length-1 block is the identity.
     */
    private static void fwht(float[] a, int offset, int len) {
        for (int h = 1; h < len; h <<= 1) {
            for (int i = 0; i < len; i += h << 1) {
                for (int j = i; j < i + h; j++) {
                    int p = offset + j;
                    int q = p + h;
                    float x = a[p];
                    float y = a[q];
                    a[p] = x + y;
                    a[q] = x - y;
                }
            }
        }
        if (len > 1) {
            float scale = (float) (1.0 / Math.sqrt(len));
            for (int i = offset; i < offset + len; i++) {
                a[i] *= scale;
            }
        }
    }
}
