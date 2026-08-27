/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Random orthogonal rotation applied to vectors before quantization.
 *
 * <p>Purpose: redistribute variance evenly across all dimensions so that
 * every quantized bit captures meaningful signal. Without this, high-variance
 * dimensions lose information while low-variance dimensions waste bits on noise.
 *
 * <p>The transform is a block-diagonal orthogonal matrix with random permutation
 * of input dimensions. Block size controls the tradeoff between transform quality
 * (larger blocks = better mixing) and cost (O(blockDim²) per vector).
 *
 * <p>Applied at index time to vectors before quantization, and at query time
 * to the query vector before scoring. The transform preserves distances
 * (orthogonal = isometry) so search results are unchanged — only quantization
 * accuracy improves.
 */
public final class RandomRotation {

    private static final int DEFAULT_BLOCK_DIM = 64;
    private static final long SEED = 42L;

    private final int blockDim;
    private final int[][] permutation;
    private final float[][][] blocks;

    private RandomRotation(int blockDim, int[][] permutation, float[][][] blocks) {
        this.blockDim = blockDim;
        this.permutation = permutation;
        this.blocks = blocks;
    }

    /**
     * Apply the rotation: out = R * vector (with permutation).
     * Both vector and out must have length = total dimension.
     */
    public void transform(float[] vector, float[] out) {
        // Always apply the permutation, even for a single block. The read-side query rotation
        // (the streaming transform(IndexInput, ...) that reads the matrix from .clar) applies it
        // unconditionally, so a matMul-without-permutation shortcut here would rotate query and
        // documents into different bases for dimension <= blockDim.
        int outIdx = 0;
        for (int b = 0; b < blocks.length; b++) {
            float[][] block = blocks[b];
            int bDim = block.length;
            int[] perm = permutation[b];
            // Gather permuted input
            for (int i = 0; i < bDim; i++) {
                float dot = 0f;
                for (int j = 0; j < bDim; j++) {
                    dot += block[i][j] * vector[perm[j]];
                }
                out[outIdx + i] = dot;
            }
            outIdx += bDim;
        }
    }

    /**
     * Transform in-place (allocates a temp buffer).
     */
    public void transformInPlace(float[] vector) {
        float[] out = new float[vector.length];
        transform(vector, out);
        System.arraycopy(out, 0, vector, 0, vector.length);
    }

    /**
     * Create a random rotation for the given vector dimension.
     */
    public static RandomRotation create(int dimension) {
        return create(dimension, Math.min(DEFAULT_BLOCK_DIM, dimension));
    }

    /**
     * Create a random rotation with specified block dimension.
     */
    public static RandomRotation create(int dimension, int blockDim) {
        blockDim = Math.min(dimension, blockDim);
        Random rng = new Random(SEED);

        // Generate block-diagonal orthogonal matrix
        int nFullBlocks = dimension / blockDim;
        int remainder = dimension % blockDim;
        int totalBlocks = nFullBlocks + (remainder > 0 ? 1 : 0);

        float[][][] blocks = new float[totalBlocks][][];
        for (int i = 0; i < nFullBlocks; i++) {
            blocks[i] = randomOrthogonal(blockDim, rng);
        }
        if (remainder > 0) {
            blocks[nFullBlocks] = randomOrthogonal(remainder, rng);
        }

        // Random permutation of dimensions into blocks
        List<Integer> indices = new ArrayList<>(dimension);
        for (int i = 0; i < dimension; i++)
            indices.add(i);
        Collections.shuffle(indices, rng);

        int[][] permutation = new int[totalBlocks][];
        int pos = 0;
        for (int i = 0; i < totalBlocks; i++) {
            int bDim = blocks[i].length;
            permutation[i] = new int[bDim];
            for (int j = 0; j < bDim; j++) {
                permutation[i][j] = indices.get(pos++);
            }
        }

        return new RandomRotation(blockDim, permutation, blocks);
    }

    /**
     * Write to index output for persistence in .clam file.
     */
    public void write(IndexOutput out) throws IOException {
        out.writeInt(blocks.length);
        out.writeInt(blockDim);
        // Write blocks
        for (float[][] block : blocks) {
            out.writeInt(block.length);
            for (float[] row : block) {
                for (float v : row) {
                    out.writeInt(Float.floatToIntBits(v));
                }
            }
        }
        // Write permutation
        for (int[] perm : permutation) {
            out.writeInt(perm.length);
            for (int idx : perm) {
                out.writeInt(idx);
            }
        }
    }

    /**
     * Read from index input.
     */
    public static RandomRotation read(IndexInput in) throws IOException {
        int numBlocks = in.readInt();
        int blockDim = in.readInt();
        float[][][] blocks = new float[numBlocks][][];
        for (int b = 0; b < numBlocks; b++) {
            int bDim = in.readInt();
            blocks[b] = new float[bDim][bDim];
            for (int i = 0; i < bDim; i++) {
                for (int j = 0; j < bDim; j++) {
                    blocks[b][i][j] = Float.intBitsToFloat(in.readInt());
                }
            }
        }
        int[][] permutation = new int[numBlocks][];
        for (int b = 0; b < numBlocks; b++) {
            int pLen = in.readInt();
            permutation[b] = new int[pLen];
            for (int j = 0; j < pLen; j++) {
                permutation[b][j] = in.readInt();
            }
        }
        return new RandomRotation(blockDim, permutation, blocks);
    }

    /**
     * Apply a serialized rotation (as written by {@link #write}) to {@code vector} → {@code out},
     * streaming the matrix directly from {@code in} at its current position — the off-heap query
     * transform. Only the permutation ({@code O(dimension)} ints) and one {@code O(blockDim)} row buffer
     * are held; the full {@code O(dimension × blockDim)} matrix is never materialized on heap. Caller
     * seeks {@code in} to the rotation start; {@code out} must be at least {@code dimension} long.
     *
     * <p>Two passes are needed because permutations follow the blocks in the file but each block's
     * permutation is required before applying it: pass 1 skips blocks to read the permutation, pass 2
     * re-reads the block rows and applies {@code out[i] = Σ_j block[i][j] · vector[perm[j]]}.
     */
    public static void transform(IndexInput in, float[] vector, float[] out) throws IOException {
        int numBlocks = in.readInt();
        int blockDim = in.readInt();
        long blocksStart = in.getFilePointer();

        for (int b = 0; b < numBlocks; b++) {
            int bDim = in.readInt();
            in.skipBytes((long) bDim * bDim * Float.BYTES);
        }
        int[][] permutation = new int[numBlocks][];
        for (int b = 0; b < numBlocks; b++) {
            int pLen = in.readInt();
            permutation[b] = new int[pLen];
            for (int j = 0; j < pLen; j++) {
                permutation[b][j] = in.readInt();
            }
        }

        in.seek(blocksStart);
        float[] row = new float[blockDim];
        int outIdx = 0;
        for (int b = 0; b < numBlocks; b++) {
            int bDim = in.readInt();
            int[] perm = permutation[b];
            for (int i = 0; i < bDim; i++) {
                in.readFloats(row, 0, bDim);
                float dot = 0f;
                for (int j = 0; j < bDim; j++) {
                    dot += row[j] * vector[perm[j]];
                }
                out[outIdx + i] = dot;
            }
            outIdx += bDim;
        }
    }

    // === Private helpers ===

    /** Generate a random orthogonal matrix via Gram-Schmidt on Gaussian random matrix. */
    private static float[][] randomOrthogonal(int dim, Random rng) {
        float[][] m = new float[dim][dim];
        // Fill with Gaussian random
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                m[i][j] = (float) rng.nextGaussian();
            }
        }
        // Modified Gram-Schmidt orthogonalization
        for (int i = 0; i < dim; i++) {
            // Normalize row i
            double norm = 0;
            for (int j = 0; j < dim; j++)
                norm += m[i][j] * m[i][j];
            norm = Math.sqrt(norm);
            if (norm < 1e-10) continue;
            for (int j = 0; j < dim; j++)
                m[i][j] /= (float) norm;
            // Subtract projection from remaining rows
            for (int k = i + 1; k < dim; k++) {
                double dot = 0;
                for (int j = 0; j < dim; j++)
                    dot += m[i][j] * m[k][j];
                for (int j = 0; j < dim; j++)
                    m[k][j] -= (float) (dot * m[i][j]);
            }
        }
        return m;
    }
}
