/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

/**
 * Flat contiguous vector storage optimized for cache-line-friendly sequential access.
 *
 * <p>Stores all vectors in a single {@code float[]} array laid out as:
 * {@code [v0d0, v0d1, ..., v0dD-1, v1d0, v1d1, ..., v1dD-1, ...]}
 *
 * <p>This layout eliminates pointer chasing (vs {@code List<float[]>}) and enables
 * hardware prefetching during sequential scans — critical for k-means assignment steps
 * where every vector is compared against every centroid.
 */
public final class VectorData {

    private final float[] data;
    private final int numVectors;
    private final int dimension;

    /**
     * Creates vector storage from a flat array.
     *
     * @param data       contiguous float array of length {@code numVectors * dimension}
     * @param numVectors number of vectors stored
     * @param dimension  dimensionality of each vector
     * @throws IllegalArgumentException if array length doesn't match numVectors * dimension
     */
    public VectorData(float[] data, int numVectors, int dimension) {
        if (data.length != numVectors * dimension) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " != numVectors(" + numVectors + ") * dimension(" + dimension + ")"
            );
        }
        this.data = data;
        this.numVectors = numVectors;
        this.dimension = dimension;
    }

    /**
     * Creates vector storage from a list of float arrays.
     *
     * @param vectors list of vectors, each of length {@code dimension}
     * @param dimension dimensionality of each vector
     * @return new VectorData instance
     */
    public static VectorData fromList(java.util.List<float[]> vectors, int dimension) {
        int n = vectors.size();
        float[] flat = new float[n * dimension];
        for (int i = 0; i < n; i++) {
            System.arraycopy(vectors.get(i), 0, flat, i * dimension, dimension);
        }
        return new VectorData(flat, n, dimension);
    }

    /**
     * Returns the offset into the backing array for the given vector index.
     */
    public int offset(int vectorIndex) {
        return vectorIndex * dimension;
    }

    /** Returns the backing flat array. */
    public float[] data() {
        return data;
    }

    /** Returns the number of vectors. */
    public int numVectors() {
        return numVectors;
    }

    /** Returns the dimension of each vector. */
    public int dimension() {
        return dimension;
    }

    /**
     * Copies a single vector into the provided destination array.
     *
     * @param vectorIndex index of the vector to copy
     * @param dest        destination array of length >= dimension
     */
    public void getVector(int vectorIndex, float[] dest) {
        System.arraycopy(data, vectorIndex * dimension, dest, 0, dimension);
    }

    /**
     * Returns a copy of the vector at the given index.
     */
    public float[] getVectorCopy(int vectorIndex) {
        float[] copy = new float[dimension];
        System.arraycopy(data, vectorIndex * dimension, copy, 0, dimension);
        return copy;
    }
}
