package org.opensearch.knn.clusterann.writer.rotation;

import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;

import java.io.IOException;
import java.util.Random;

/**
 * Generates and persists the block-diagonal rotation that {@code RandomGaussianRotation} reads back.
 *
 * <p>A permutation of the dimensions, then an independent orthonormal rotation within each group. Orthonormal per
 * block makes the whole transform orthogonal, so distances, dot products and therefore the ranking survive it; the
 * permutation is what makes the blocks useful, by scattering correlated dimensions into different groups.
 *
 * <p>Blocks rather than one dense {@code d × d} matrix: the cost of applying it falls from {@code O(d²)} to
 * {@code O(Σ bDim²)}, and the stored size falls with it.
 *
 * <h2>Layout — the inverse of {@code RandomGaussianRotation}</h2>
 *
 * <pre>
 * numBlocks     vInt
 * blocks        per block:  bDim vInt, then bDim × bDim floats, row-major
 * permutation   per block:  bDim vInt, then bDim ints
 * </pre>
 *
 * <p>Three conventions the read side depends on and cannot check, so they are stated here and pinned by a
 * round-trip test:
 *
 * <ul>
 *   <li><b>Row-major:</b> {@code out[j]} is row {@code j} dotted with the block's gathered input. A transposed
 *       rotation is still a rotation, so getting this backwards scores against a different space rather than
 *       failing.
 *   <li><b>The permutation gathers, with global indices:</b> {@code indices[j]} names the {@code src} dimension
 *       that feeds slot {@code j}.
 *   <li><b>A block's output is contiguous:</b> block {@code g} writes {@code dest[offset .. offset + bDim)}.
 * </ul>
 */
public final class RandomGaussianRotationWriter implements RotationWriter {

    private final int dimension;
    private final float[][][] matrices;
    private final int[][] indices;

    /**
     * Builds the rotation. Deterministic in {@code seed}, so the same field clustered twice rotates the same way —
     * which is what makes a written segment reproducible.
     *
     * @param blockDimension the target width of a block; the last block takes the remainder when the dimension does
     *     not divide evenly
     */
    public RandomGaussianRotationWriter(int dimension, int blockDimension, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive, got: " + dimension);
        }
        if (blockDimension <= 0) {
            throw new IllegalArgumentException("blockDimension must be positive, got: " + blockDimension);
        }
        this.dimension = dimension;

        Random random = new Random(seed);
        int[] widths = widths(dimension, blockDimension);
        this.matrices = new float[widths.length][][];
        for (int block = 0; block < widths.length; block++) {
            matrices[block] = orthonormal(widths[block], random);
        }
        this.indices = gatherIndices(widths, dimension, random);
    }

    @Override
    public int rotationId() {
        return ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN;
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
        int offset = 0;
        for (int block = 0; block < matrices.length; block++) {
            float[][] matrix = matrices[block];
            int[] gather = indices[block];
            int blockDimension = gather.length;

            float[] gathered = new float[blockDimension];
            for (int slot = 0; slot < blockDimension; slot++) {
                gathered[slot] = src[gather[slot]];
            }
            for (int row = 0; row < blockDimension; row++) {
                dest[offset + row] = VectorUtil.dotProduct(matrix[row], gathered);
            }
            offset += blockDimension;
        }
    }

    @Override
    public long write(IndexOutput clar) throws IOException {
        long start = clar.getFilePointer();

        clar.writeVInt(matrices.length);
        for (float[][] matrix : matrices) {
            clar.writeVInt(matrix.length);
            for (float[] row : matrix) {
                for (float value : row) {
                    clar.writeInt(Float.floatToIntBits(value));
                }
            }
        }
        // The permutation follows every block, not each block's matrix, because that is the order the reader reads
        // it in: all matrices first, then all index sets.
        for (int[] gather : indices) {
            clar.writeVInt(gather.length);
            for (int index : gather) {
                clar.writeInt(index);
            }
        }
        return clar.getFilePointer() - start;
    }

    /** Block widths covering {@code dimension} exactly, the last absorbing whatever does not divide evenly. */
    private static int[] widths(int dimension, int blockDimension) {
        int full = dimension / blockDimension;
        int remainder = dimension % blockDimension;
        int count = full + (remainder > 0 ? 1 : 0);
        if (count == 0) {
            // A dimension smaller than one block is still one block.
            return new int[] { dimension };
        }
        int[] widths = new int[count];
        for (int i = 0; i < full; i++) {
            widths[i] = blockDimension;
        }
        if (remainder > 0) {
            widths[count - 1] = remainder;
        }
        return widths;
    }

    /**
     * A random orthonormal {@code n × n} matrix, by Gram-Schmidt over Gaussian rows.
     *
     * <p>Gaussian entries make the result uniform over rotations, which is the property that flattens variance
     * across dimensions. A degenerate row — one that Gram-Schmidt leaves with no length, which is possible but
     * vanishingly unlikely — is redrawn rather than normalised, since normalising it would break orthogonality.
     */
    private static float[][] orthonormal(int n, Random random) {
        float[][] matrix = new float[n][];
        for (int row = 0; row < n; row++) {
            for (int attempt = 0; attempt < 100; attempt++) {
                float[] candidate = new float[n];
                for (int i = 0; i < n; i++) {
                    candidate[i] = (float) random.nextGaussian();
                }
                for (int done = 0; done < row; done++) {
                    float projection = VectorUtil.dotProduct(candidate, matrix[done]);
                    for (int i = 0; i < n; i++) {
                        candidate[i] -= projection * matrix[done][i];
                    }
                }
                double norm = Math.sqrt(VectorUtil.dotProduct(candidate, candidate));
                if (norm > 1e-6) {
                    for (int i = 0; i < n; i++) {
                        candidate[i] /= (float) norm;
                    }
                    matrix[row] = candidate;
                    break;
                }
            }
            if (matrix[row] == null) {
                throw new IllegalStateException("Could not build an orthonormal row " + row + " of " + n);
            }
        }
        return matrix;
    }

    /**
     * A random permutation of {@code 0 .. dimension-1}, cut into one gather set per block.
     *
     * <p>Every dimension is named exactly once, which is what the reader verifies: any dimension left out would
     * leave part of the rotated vector untouched, and any named twice would double-count it.
     */
    private static int[][] gatherIndices(int[] widths, int dimension, Random random) {
        int[] permutation = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            permutation[i] = i;
        }
        for (int i = dimension - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int swap = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = swap;
        }

        int[][] indices = new int[widths.length][];
        int offset = 0;
        for (int block = 0; block < widths.length; block++) {
            indices[block] = new int[widths[block]];
            System.arraycopy(permutation, offset, indices[block], 0, widths[block]);
            offset += widths[block];
        }
        return indices;
    }
}
