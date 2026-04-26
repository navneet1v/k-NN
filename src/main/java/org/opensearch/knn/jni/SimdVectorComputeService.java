/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

/**
 * A service that computes vector similarity using native SIMD acceleration.
 * This service relies on a shared native library that implements optimized SIMD instructions to achieve faster performance during
 * similarity computations. The library must be properly loaded and available on the system before invoking any methods
 * that depend on native code.
 */
public class SimdVectorComputeService {
    static {
        KNNLibraryLoader.loadSimdLibrary();
    }

    /**
     * Similarity calculation type to passed down to native code.
     */
    public enum SimilarityFunctionType {
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.
        FP16_MAXIMUM_INNER_PRODUCT,
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.EUCLIDEAN.
        FP16_L2,
        SQ_IP,
        SQ_L2
    }

    public static native void saveSQSearchContext(
        byte[] quantizedQuery,
        float queryLowerInterval,
        float queryUpperInterval,
        float queryAdditionalCorrection,
        int queryQuantizedComponentSum,
        long[] addressAndSize,
        int nativeFunctionTypeOrd,
        int dimension,
        float centroidDp
    );

    /**
     * With vector ids, performing bulk SIMD similarity calculations and put the results into `scores`.
     *
     * @param internalVectorIds Vectors to load for similarity calculations.
     * @param scores            Results will be put into this array.
     * @param numVectors        The number of valid vector ids in `internalVectorIds`. Therefore, this will put exactly `numVectors` result
     *                          values into `scores`.
     */
    public native static float scoreSimilarityInBulk(int[] internalVectorIds, float[] scores, int numVectors);

    /**
     * Before vector search starts, it persists required information into a storage. Those persisted information will be used during search.
     * This must be called prior to each search.
     *
     * @param query                  Query vector
     * @param addressAndSize         An array describing vector chunks, where each pair of elements represents a chunk.
     *                               addressAndSize[i] is the starting memory address of the j-th chunk,
     *                               and addressAndSize[i + 1] is the size (in bytes) of that chunk where i = 2 * j.
     *                               Ex: addressAndSize[6] is the starting memory address of 3rd chunk, addressAndSize[7] is the size of
     *                               that chunk.
     * @param nativeFunctionTypeOrd  Similarity function type index.
     */
    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    /**
     * Perform similarity search on a single vector.
     *
     * @param internalVectorId Vector id
     * @return Similarity score.
     */
    public native static float scoreSimilarity(int internalVectorId);

    // ===== ClusterANN bulk operations (pure functions, no state, thread-safe) =====

    /**
     * Batch quantized dot product for block-columnar ADC scoring.
     * Supports 1-bit, 2-bit, and 4-bit document quantization against 4-bit query.
     *
     * @param queryTransposed 4-bit query transposed into bit planes
     * @param docCodesBatch   contiguous packed codes (numVectors × bytesPerCode)
     * @param results         output raw dot products
     * @param bytesPerCode    packed bytes per vector
     * @param numVectors      vectors to score (≤ 32)
     * @param docBits         quantization bits (1, 2, or 4)
     */
    public static native void bulkQuantizedDotProduct(
        byte[] queryTransposed,
        byte[] docCodesBatch,
        float[] results,
        int bytesPerCode,
        int numVectors,
        int docBits
    );

    /**
     * Compute distances from one vector to multiple centroids.
     * Returns index of nearest centroid.
     *
     * @param vector       query vector
     * @param centroids    flat array of centroids (numCentroids × dimension)
     * @param distances    output distances
     * @param dimension    vector dimension
     * @param numCentroids number of centroids
     * @param metricOrd    0=L2, 1=DOT_PRODUCT
     * @return index of nearest centroid
     */
    public static native int bulkCentroidDistance(
        float[] vector,
        float[] centroids,
        float[] distances,
        int dimension,
        int numCentroids,
        int metricOrd
    );

    /**
     * Compute SOAR distances from a vector to candidate centroids.
     * SOAR distance = L2² + lambda × (projection² / ||residual||²)
     *
     * @param vector          the vector
     * @param primaryCentroid the primary assigned centroid
     * @param candidates      flat array of candidate centroids (numCandidates × dimension)
     * @param distances       output SOAR distances
     * @param dimension       vector dimension
     * @param numCandidates   number of candidates
     * @param soarLambda      SOAR lambda parameter
     */
    public static native void bulkSOARDistance(
        float[] vector,
        float[] primaryCentroid,
        float[] candidates,
        float[] distances,
        int dimension,
        int numCandidates,
        float soarLambda
    );
}
