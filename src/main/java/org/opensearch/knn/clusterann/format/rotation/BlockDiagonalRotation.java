/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.VectorUtil;

import org.opensearch.common.Nullable;
import java.io.IOException;
import java.util.Objects;

/**
 * A rotation by a permutation of the dimensions followed by an independent orthogonal block over each group.
 *
 * <p>It is one {@code d × d} orthogonal matrix that is zero outside its diagonal blocks, so only the blocks are stored
 * and applied. Orthogonal exactly when every block is, so distances, dot products and the ranking survive it. Cost is
 * {@code O(Σ bDim²)} rather than {@code O(d²)}.
 *
 * <p>The permutation scatters correlated dimensions across different blocks, recovering the variance-flattening a dense
 * rotation gives; it is applied even for a single block, or the rotation would land in a different space from the one
 * its file describes.
 *
 * <h2>Layout</h2>
 *
 * <pre>
 * numBlocks     vInt
 * blocks        per block:  bDim vInt, then bDim × bDim floats, row-major
 * permutation   per block:  bDim vInt, then bDim ints
 * </pre>
 *
 * <p>Three conventions, each silent if reversed, so each is pinned by a test:
 *
 * <ul>
 *   <li><b>Row-major; {@code out[j]} is row {@code j} dotted with the block's input.</b> A transposed rotation is still
 *       a rotation, scoring against the wrong space.
 *   <li><b>The permutation gathers, with global indices.</b> {@code indices[j]} is the {@code src} dimension feeding
 *       slot {@code j}, not the destination of {@code src[j]}.
 *   <li><b>A block's output is contiguous.</b> Block {@code g} writes {@code dest[offset .. offset + bDim)}.
 * </ul>
 *
 * <p>Every dimension belongs to exactly one block ({@code Σ bDim == dimension}) and the indices form a permutation of
 * {@code 0 … dimension-1}; both are verified on read. Read lazily and once — an unqueried field never pays for it.
 */
public final class BlockDiagonalRotation implements Rotation {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(BlockDiagonalRotation.class);

    private final int dimension;

    /**
     * The blocks, once they are in hand. Set from the start for a generated rotation, and on the first {@link #rotate}
     * for one read from a region.
     *
     * <p>Exactly one of this and {@link #region} is non-null at construction, which is why there are two constructors
     * rather than one taking both: a rotation is either the blocks or the promise of them, never a pair of maybes.
     */
    private volatile Blocks blocks;

    /** The region {@link #blocks} is read from on first use; {@code null} for a rotation that was generated. */
    @Nullable
    private final IndexInput region;

    /** A rotation over blocks already in memory: nothing to read, so {@link #rotate} can go straight to them. */
    BlockDiagonalRotation(int dimension, Blocks blocks) {
        this.dimension = dimension;
        this.blocks = blocks;
        this.region = null;
    }

    /** A rotation that is still only its region: the blocks are read on the first {@link #rotate} and kept. */
    BlockDiagonalRotation(int dimension, IndexInput region) {
        this.dimension = dimension;
        this.blocks = null;
        this.region = region;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public void rotate(float[] src, float[] dest) throws IOException {
        if (src.length != dimension || dest.length != dimension) {
            throw new IllegalArgumentException(
                "This rotation is " + dimension + "-dimensional, got src=" + src.length + " dest=" + dest.length
            );
        }

        final Blocks loaded = blocks();
        int offset = 0;
        for (int block = 0; block < loaded.matrices.length; block++) {
            final float[][] matrix = loaded.matrices[block];
            final int[] indices = loaded.indices[block];
            final int blockDimension = indices.length;

            // Gathered into its own array so each row is a contiguous, vectorised dot product.
            final float[] gathered = new float[blockDimension];
            for (int slot = 0; slot < blockDimension; slot++) {
                gathered[slot] = src[indices[slot]];
            }

            for (int row = 0; row < blockDimension; row++) {
                dest[offset + row] = VectorUtil.dotProduct(matrix[row], gathered);
            }
            offset += blockDimension;
        }
    }

    @Override
    public long ramBytesUsed() {
        final Blocks loaded = blocks;
        if (loaded == null) {
            return BASE_RAM_USAGE;
        }

        long bytes = BASE_RAM_USAGE + RamUsageEstimator.shallowSizeOf(loaded.matrices) + RamUsageEstimator.shallowSizeOf(loaded.indices);
        for (float[][] matrix : loaded.matrices) {
            bytes += RamUsageEstimator.shallowSizeOf(matrix);
            for (float[] row : matrix) {
                bytes += RamUsageEstimator.sizeOf(row);
            }
        }
        for (int[] indices : loaded.indices) {
            bytes += RamUsageEstimator.sizeOf(indices);
        }
        return bytes;
    }

    /**
     * The blocks, read on first use. Synchronised so two queries arriving together on a cold field do not both read
     * and allocate, throwing one copy away.
     *
     * <p>Package-private because {@link BlockDiagonalRotationFormat#write} needs them to lay the rotation down, and
     * because the parsing itself belongs to the format — this only decides <em>when</em> it happens.
     */
    Blocks blocks() throws IOException {
        Blocks loaded = blocks;
        if (loaded != null) {
            return loaded;
        }
        synchronized (this) {
            if (blocks == null) {
                // Non-null by construction: the only rotation without blocks is one built from a region.
                blocks = BlockDiagonalRotationFormat.readBlocks(Objects.requireNonNull(region), dimension);
            }
            return blocks;
        }
    }

    /**
     * The blocks and the dimensions each gathers, as one value so {@link #blocks} publishes them together — a matrix
     * visible without its permutation would be applied to the wrong coordinates.
     *
     * @param matrices {@code matrices[block][row][column]}, row-major
     * @param indices {@code indices[block][slot]} is the {@code src} dimension feeding that slot
     */
    record Blocks(float[][][] matrices, int[][] indices) {
    }
}
