/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.util.RamUsageEstimator;

import java.util.Random;

/**
 * A data-blind Hadamard rotation: a random sign flip, a random permutation, then a normalized Fast
 * Walsh–Hadamard Transform (FWHT) applied independently over each power-of-two block that composes the
 * dimension. {@code R = F·P·S}.
 *
 * <p>Orthonormal, so distances, dot products and the ranking survive it ({@code ⟨Rq, Rd⟩ = ⟨q, d⟩}). Its
 * purpose is to <em>flatten per-dimension variance</em> so a single fixed quantizer (data-blind int8) is
 * accurate for every dimension — the same role a dense random rotation plays, at {@code O(d log d)} cost and
 * storing almost nothing: just the dimension and a seed, from which the signs and permutation are
 * regenerated. This is what makes it "data-blind" — nothing about the corpus is learned or stored.
 *
 * <p><b>Non-power-of-two dimensions.</b> {@code dim} is split into its set bits, largest first
 * ({@code 768 → [512][256]}, {@code 384 → [256][128]}), and an independent normalized FWHT runs over each
 * block. The sign flip and permutation span the whole dimension before the per-block transforms, so energy
 * from any input coordinate can reach any block — the variance-flattening a single FWHT could not give across
 * block boundaries.
 *
 * <p>Thread-safe and immutable once constructed; one per field, shared by every search of the segment.
 */
public final class HadamardRotation implements Rotation {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(HadamardRotation.class);

    /** The default seed, when a field is created without an explicit one. Recorded in {@code .clar}. */
    public static final long DEFAULT_SEED = 42L;

    private final int dimension;
    private final long seed;

    /** Per-coordinate sign flips (±1), length {@code dimension}. */
    private final float[] signs;
    /** Permutation of {@code [0, dimension)} applied before the block transforms. */
    private final int[] permutation;
    /** Start offset of each power-of-two FWHT block. */
    private final int[] blockOffsets;
    /** Length of each FWHT block (a power of two). */
    private final int[] blockLengths;

    private HadamardRotation(int dimension, long seed, float[] signs, int[] permutation, int[] blockOffsets, int[] blockLengths) {
        this.dimension = dimension;
        this.seed = seed;
        this.signs = signs;
        this.permutation = permutation;
        this.blockOffsets = blockOffsets;
        this.blockLengths = blockLengths;
    }

    /** Build a Hadamard rotation for {@code dimension} from {@code seed}; both sides reconstruct from these two. */
    public static HadamardRotation create(int dimension, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive, got " + dimension);
        }
        final Random random = new Random(seed);

        final float[] signs = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            signs[i] = random.nextBoolean() ? 1f : -1f;
        }

        final int[] permutation = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            permutation[i] = i;
        }
        for (int i = dimension - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);
            final int tmp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tmp;
        }

        // Decompose dim into power-of-two blocks (its set bits), largest first.
        final int numBlocks = Integer.bitCount(dimension);
        final int[] blockOffsets = new int[numBlocks];
        final int[] blockLengths = new int[numBlocks];
        int b = 0;
        int offset = 0;
        for (int bit = Integer.highestOneBit(dimension); bit != 0; bit >>>= 1) {
            if ((dimension & bit) != 0) {
                blockOffsets[b] = offset;
                blockLengths[b] = bit;
                offset += bit;
                b++;
            }
        }
        return new HadamardRotation(dimension, seed, signs, permutation, blockOffsets, blockLengths);
    }

    public long seed() {
        return seed;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public void rotate(float[] src, float[] dest) {
        if (src.length != dimension || dest.length != dimension) {
            throw new IllegalArgumentException(
                "This rotation is " + dimension + "-dimensional, got src=" + src.length + " dest=" + dest.length
            );
        }
        if (src == dest) {
            throw new IllegalArgumentException("src and dest must be distinct arrays");
        }
        // S then P: dest[i] = signs[perm[i]] * src[perm[i]]
        for (int i = 0; i < dimension; i++) {
            final int p = permutation[i];
            dest[i] = signs[p] * src[p];
        }
        // F: normalized FWHT over each power-of-two block, in place on dest.
        for (int blk = 0; blk < blockOffsets.length; blk++) {
            fwht(dest, blockOffsets[blk], blockLengths[blk]);
        }
    }

    /** In-place normalized Fast Walsh–Hadamard Transform over {@code a[off, off+len)}, len a power of two. */
    private static void fwht(float[] a, int off, int len) {
        for (int step = 1; step < len; step <<= 1) {
            for (int i = 0; i < len; i += step << 1) {
                for (int j = i; j < i + step; j++) {
                    final int x = off + j;
                    final int y = off + j + step;
                    final float u = a[x];
                    final float v = a[y];
                    a[x] = u + v;
                    a[y] = u - v;
                }
            }
        }
        // Normalize so the transform is orthonormal (preserves norms and dot products).
        final float inv = (float) (1.0 / Math.sqrt(len));
        for (int i = off; i < off + len; i++) {
            a[i] *= inv;
        }
    }

    @Override
    public long ramBytesUsed() {
        return BASE_RAM_USAGE + RamUsageEstimator.sizeOf(signs) + RamUsageEstimator.sizeOf(permutation) + RamUsageEstimator.sizeOf(
            blockOffsets
        ) + RamUsageEstimator.sizeOf(blockLengths);
    }
}
