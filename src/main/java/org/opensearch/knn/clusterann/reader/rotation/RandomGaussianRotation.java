/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.rotation;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;

/**
 * A block-diagonal rotation held in {@code .clar}: a permutation of the dimensions, then an independent small
 * rotation applied within each group.
 *
 * <p>It is one {@code d × d} orthogonal matrix that is zero outside its diagonal blocks, so only the blocks are
 * stored and applied. The transform is orthogonal exactly when every block is, so distances, dot products and the
 * ranking survive it. Cost falls from {@code O(d²)} to {@code O(Σ bDim²)}.
 *
 * <p>The permutation scatters correlated dimensions across different blocks, which is what recovers the
 * variance-flattening a dense rotation gives — contiguous groups would mix already-correlated coordinates to little
 * effect.
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
 *   <li><b>Row-major; {@code out[j]} is row {@code j} dotted with the block's input.</b> A transposed rotation is
 *       still a rotation, so the wrong orientation scores against the wrong space.
 *   <li><b>The permutation gathers, with global indices.</b> {@code indices[j]} is the {@code src} dimension feeding
 *       slot {@code j}, not the destination of {@code src[j]}.
 *   <li><b>A block's output is contiguous.</b> Block {@code g} writes {@code dest[offset .. offset + bDim)}. Nothing
 *       downstream depends on which output position an input reaches, so contiguous writes are chosen.
 * </ul>
 *
 * <p>Every dimension belongs to exactly one block: {@code Σ bDim == dimension}, and the indices form a permutation
 * of {@code 0 … dimension-1}. Both are verified on read.
 *
 * <p>The input is this field's region of {@code .clar}, already sliced to it, so the header begins at zero. The
 * blocks are read once, on the first {@link #rotate}, and kept — a field that is never queried never pays.
 */
public final class RandomGaussianRotation implements Rotation {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(RandomGaussianRotation.class);

    private final IndexInput rotation;
    private final int dimension;

    /**
     * The blocks and their permutation, {@code null} until the first {@link #rotate}. Volatile so a reader that sees a
     * non-null reference sees the fully populated arrays behind it.
     */
    private volatile Blocks blocks;

    /**
     * Records where the rotation is. Reads nothing.
     *
     * @param clar this field's region of {@code .clar}, sliced so the header begins at zero. Cloned here, so this
     *     instance never moves the caller's file pointer.
     * @param dimension the field's vector dimension, which the blocks must account for exactly
     */
    public RandomGaussianRotation(IndexInput clar, int dimension) {
        this.rotation = clar.clone();
        this.dimension = dimension;
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

        Blocks loaded = blocks();
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

    /**
     * The blocks, read on first use. Synchronised so two queries arriving together on a cold field do not both read
     * and allocate, throwing one copy away.
     */
    private Blocks blocks() throws IOException {
        Blocks loaded = blocks;
        if (loaded != null) {
            return loaded;
        }
        synchronized (this) {
            if (blocks == null) {
                blocks = read();
            }
            return blocks;
        }
    }

    private Blocks read() throws IOException {
        rotation.seek(0L);

        // A block holds at least one dimension, so there can be no more blocks than dimensions — bounding the
        // allocation below before a corrupt count can size an array.
        int numBlocks = rotation.readVInt();
        if (numBlocks < 0 || numBlocks > dimension) {
            throw corrupt("numBlocks must be in [0, " + dimension + "], got: " + numBlocks);
        }

        float[][][] matrices = new float[numBlocks][][];
        int[] widths = new int[numBlocks];
        int covered = 0;
        for (int block = 0; block < numBlocks; block++) {
            int blockDimension = blockDimension(block, covered);
            widths[block] = blockDimension;
            covered += blockDimension;

            requireAvailable((long) blockDimension * blockDimension * Float.BYTES, "block " + block);
            float[][] matrix = new float[blockDimension][];
            for (int row = 0; row < blockDimension; row++) {
                float[] values = new float[blockDimension];
                rotation.readFloats(values, 0, blockDimension);
                matrix[row] = values;
            }
            matrices[block] = matrix;
        }

        // The blocks must account for every dimension exactly, or a tail of dest would be left untouched.
        if (covered != dimension) {
            throw corrupt("blocks cover " + covered + " dimensions, but the field has " + dimension);
        }

        return new Blocks(matrices, permutation(widths));
    }

    /** One block's width, rejected if it cannot fit in what the earlier blocks have left of the vector. */
    private int blockDimension(int block, int covered) throws IOException {
        int blockDimension = rotation.readVInt();
        if (blockDimension <= 0 || blockDimension > dimension - covered) {
            throw corrupt("block " + block + " has dimension " + blockDimension + ", with " + (dimension - covered) + " left to cover");
        }
        return blockDimension;
    }

    /**
     * The gather indices, one set per block, verified to form a permutation of the whole vector: every dimension
     * named once, none out of range, each set as wide as the block it feeds.
     */
    private int[][] permutation(int[] widths) throws IOException {
        int[][] indices = new int[widths.length][];
        boolean[] claimed = new boolean[dimension];

        for (int block = 0; block < widths.length; block++) {
            int blockDimension = rotation.readVInt();
            if (blockDimension != widths[block]) {
                throw corrupt(
                    "block " + block + " is " + widths[block] + " wide, but its permutation names " + blockDimension + " dimensions"
                );
            }

            requireAvailable((long) blockDimension * Integer.BYTES, "the permutation of block " + block);
            int[] gathered = new int[blockDimension];
            rotation.readInts(gathered, 0, blockDimension);
            for (int slot = 0; slot < blockDimension; slot++) {
                int source = gathered[slot];
                if (source < 0 || source >= dimension) {
                    throw corrupt("block " + block + " gathers dimension " + source + ", outside [0, " + dimension + ")");
                }
                if (claimed[source]) {
                    throw corrupt("dimension " + source + " is gathered by more than one block");
                }
                claimed[source] = true;
            }
            indices[block] = gathered;
        }
        return indices;
    }

    /**
     * Rejects a read the region cannot satisfy before anything is allocated for it, so a corrupt width fails clearly
     * rather than as an EOF partway through a matrix.
     */
    private void requireAvailable(long bytes, String what) throws IOException {
        long remaining = rotation.length() - rotation.getFilePointer();
        if (bytes > remaining) {
            throw corrupt(what + " needs " + bytes + " bytes, but only " + remaining + " remain");
        }
    }

    private CorruptIndexException corrupt(String message) {
        return new CorruptIndexException(message, rotation);
    }

    /** What the rotation costs once resident, and only then — an unqueried field leaves this at the base size. */
    @Override
    public long ramBytesUsed() {
        Blocks loaded = blocks;
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
        for (int[] gathered : loaded.indices) {
            bytes += RamUsageEstimator.sizeOf(gathered);
        }
        return bytes;
    }

    /**
     * The blocks and the dimensions each gathers, as one value so {@link #blocks} publishes them together — a matrix
     * visible without its permutation would be applied to the wrong coordinates.
     *
     * @param matrices {@code matrices[block][row][column]}, row-major
     * @param indices {@code indices[block][slot]} is the {@code src} dimension feeding that slot
     */
    private record Blocks(float[][][] matrices, int[][] indices) {
    }
}
