/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import org.opensearch.common.Nullable;
import java.io.IOException;
import java.util.Random;

/**
 * The {@code .clar} layout of a {@link BlockDiagonalRotation}, in both directions, plus the generation of one.
 *
 * <pre>
 * numBlocks     vInt
 * blocks        per block:  bDim vInt, then bDim × bDim floats, row-major
 * permutation   per block:  bDim vInt, then bDim ints
 * </pre>
 *
 * <p>Read and write share a class so the three conventions the layout rests on — row-major rows, a permutation that
 * gathers with global indices, contiguous per-block output — cannot drift between a separate reader and writer, each
 * being silent if reversed.
 *
 * <p>Every dimension must belong to exactly one block ({@code Σ bDim == dimension}) and the indices must form a
 * permutation of {@code 0 … dimension-1}; both are verified on read, since a repeated index would quietly drop a
 * dimension yet still produce plausible numbers.
 */
final class BlockDiagonalRotationFormat implements RotationFormat<BlockDiagonalRotation> {

    /** The one instance. Stateless; every rotation it hands out carries its own blocks or its own region. */
    static final BlockDiagonalRotationFormat INSTANCE = new BlockDiagonalRotationFormat();

    /** Widest block generated unless a caller asks otherwise — beyond this the {@code bDim²} cost stops paying. */
    private static final int DEFAULT_BLOCK_DIMENSION = 64;

    /**
     * Fixed so a segment rebuilt from the same vectors rotates the same way, keeping a rebuild diffable. Plain
     * {@link Random}, not {@code SecureRandom}: reproducibility is the point, and nothing here is secret — the whole
     * matrix is written to {@code .clar}.
     */
    private static final long SEED = 42L;

    private BlockDiagonalRotationFormat() {}

    @Override
    public BlockDiagonalRotation create(int dimension) {
        return create(dimension, Math.min(DEFAULT_BLOCK_DIMENSION, dimension));
    }

    /**
     * Generate at an explicit block width rather than {@link #DEFAULT_BLOCK_DIMENSION}.
     *
     * <p>The width is the cost/quality dial: narrower blocks mean less work and less heap per query, but spread each
     * spike over fewer coordinates. For a caller that has measured; {@link #create(int)} picks the default.
     */
    private BlockDiagonalRotation create(int dimension, int blockDimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive, got: " + dimension);
        }
        final int width = Math.min(dimension, blockDimension);
        final Random rng = new Random(SEED);

        final int fullBlocks = dimension / width;
        final int remainder = dimension % width;
        final int totalBlocks = fullBlocks + (remainder > 0 ? 1 : 0);

        final float[][][] matrices = new float[totalBlocks][][];
        for (int block = 0; block < fullBlocks; block++) {
            matrices[block] = randomOrthogonal(width, rng);
        }
        if (remainder > 0) {
            matrices[fullBlocks] = randomOrthogonal(remainder, rng);
        }

        return new BlockDiagonalRotation(dimension, new BlockDiagonalRotation.Blocks(matrices, shuffledInto(matrices, dimension, rng)));
    }

    @Override
    public long write(IndexOutput out, BlockDiagonalRotation rotation) throws IOException {
        final BlockDiagonalRotation.Blocks blocks = rotation.blocks();
        final long start = out.getFilePointer();

        out.writeVInt(blocks.matrices().length);
        for (float[][] matrix : blocks.matrices()) {
            out.writeVInt(matrix.length);
            for (float[] row : matrix) {
                for (float value : row) {
                    out.writeInt(Float.floatToIntBits(value));
                }
            }
        }
        for (int[] indices : blocks.indices()) {
            out.writeVInt(indices.length);
            for (int index : indices) {
                out.writeInt(index);
            }
        }
        return out.getFilePointer() - start;
    }

    @Override
    public BlockDiagonalRotation read(@Nullable IndexInput region, int dimension) {
        // This implementation stores its blocks, so a missing region is a caller error, not the "stores nothing" case.
        if (region == null) {
            throw new IllegalArgumentException("A block-diagonal rotation needs its .clar region, got none");
        }
        return new BlockDiagonalRotation(dimension, region.clone());
    }

    /**
     * Parse a whole region into blocks. Called by {@link BlockDiagonalRotation} on its first rotate — the rotation
     * decides <em>when</em>, this class decides what the bytes mean.
     */
    static BlockDiagonalRotation.Blocks readBlocks(IndexInput region, int dimension) throws IOException {
        region.seek(0L);

        // No more blocks than dimensions (each holds at least one), bounding the allocation before a corrupt count
        // can size an array.
        final int numBlocks = region.readVInt();
        if (numBlocks < 0 || numBlocks > dimension) {
            throw corrupt(region, "numBlocks must be in [0, " + dimension + "], got: " + numBlocks);
        }

        final float[][][] matrices = new float[numBlocks][][];
        final int[] widths = new int[numBlocks];
        int covered = 0;
        for (int block = 0; block < numBlocks; block++) {
            final int blockDimension = region.readVInt();
            if (blockDimension <= 0 || blockDimension > dimension - covered) {
                throw corrupt(
                    region,
                    "block " + block + " has dimension " + blockDimension + ", with " + (dimension - covered) + " left to cover"
                );
            }
            widths[block] = blockDimension;
            covered += blockDimension;

            requireAvailable(region, (long) blockDimension * blockDimension * Float.BYTES, "block " + block);
            final float[][] matrix = new float[blockDimension][];
            for (int row = 0; row < blockDimension; row++) {
                final float[] values = new float[blockDimension];
                region.readFloats(values, 0, blockDimension);
                matrix[row] = values;
            }
            matrices[block] = matrix;
        }

        // The blocks must account for every dimension exactly, or a tail of dest would be left untouched.
        if (covered != dimension) {
            throw corrupt(region, "blocks cover " + covered + " dimensions, but the field has " + dimension);
        }

        return new BlockDiagonalRotation.Blocks(matrices, permutation(region, widths, dimension));
    }

    /**
     * The gather indices, one set per block, verified to form a permutation of the whole vector: every dimension
     * named once, none out of range, each set as wide as the block it feeds.
     */
    private static int[][] permutation(IndexInput region, int[] widths, int dimension) throws IOException {
        final int[][] indices = new int[widths.length][];
        final boolean[] claimed = new boolean[dimension];

        for (int block = 0; block < widths.length; block++) {
            final int blockDimension = region.readVInt();
            if (blockDimension != widths[block]) {
                throw corrupt(
                    region,
                    "block " + block + " is " + widths[block] + " wide, but its permutation names " + blockDimension + " dimensions"
                );
            }

            requireAvailable(region, (long) blockDimension * Integer.BYTES, "the permutation of block " + block);
            final int[] gathered = new int[blockDimension];
            region.readInts(gathered, 0, blockDimension);
            for (int slot = 0; slot < blockDimension; slot++) {
                final int source = gathered[slot];
                if (source < 0 || source >= dimension) {
                    throw corrupt(region, "block " + block + " gathers dimension " + source + ", outside [0, " + dimension + ")");
                }
                if (claimed[source]) {
                    throw corrupt(region, "dimension " + source + " is gathered by more than one block");
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
    private static void requireAvailable(IndexInput region, long bytes, String what) throws IOException {
        final long remaining = region.length() - region.getFilePointer();
        if (bytes > remaining) {
            throw corrupt(region, what + " needs " + bytes + " bytes, but only " + remaining + " remain");
        }
    }

    private static CorruptIndexException corrupt(IndexInput region, String message) {
        return new CorruptIndexException(message, region);
    }

    /** Deals every dimension into exactly one block's slots, in shuffled order — the grouping the blocks mix over. */
    private static int[][] shuffledInto(float[][][] matrices, int dimension, Random rng) {
        final int[] shuffled = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            shuffled[i] = i;
        }
        for (int i = dimension - 1; i > 0; i--) {
            final int j = rng.nextInt(i + 1);
            final int swap = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = swap;
        }

        final int[][] indices = new int[matrices.length][];
        int position = 0;
        for (int block = 0; block < matrices.length; block++) {
            final int width = matrices[block].length;
            indices[block] = new int[width];
            for (int slot = 0; slot < width; slot++) {
                indices[block][slot] = shuffled[position++];
            }
        }
        return indices;
    }

    /** A random orthogonal matrix, by modified Gram-Schmidt over Gaussian rows. */
    private static float[][] randomOrthogonal(int width, Random rng) {
        final float[][] matrix = new float[width][width];
        for (int row = 0; row < width; row++) {
            for (int column = 0; column < width; column++) {
                matrix[row][column] = (float) rng.nextGaussian();
            }
        }

        for (int row = 0; row < width; row++) {
            double norm = 0;
            for (int column = 0; column < width; column++) {
                norm += matrix[row][column] * matrix[row][column];
            }
            norm = Math.sqrt(norm);
            if (norm < 1e-10) {
                continue;
            }
            for (int column = 0; column < width; column++) {
                matrix[row][column] /= (float) norm;
            }
            for (int other = row + 1; other < width; other++) {
                double dot = 0;
                for (int column = 0; column < width; column++) {
                    dot += matrix[row][column] * matrix[other][column];
                }
                for (int column = 0; column < width; column++) {
                    matrix[other][column] -= (float) (dot * matrix[row][column]);
                }
            }
        }
        return matrix;
    }
}
