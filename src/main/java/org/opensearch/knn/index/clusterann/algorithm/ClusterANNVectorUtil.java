/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

import org.apache.lucene.util.VectorUtil;

/**
 * SIMD-accelerated vector distance utilities for ClusterANN algorithms.
 *
 * <p>Single-pair methods delegate to Lucene's {@link VectorUtil} which uses Panama SIMD
 * on capable JVMs via Lucene's {@code VectorizationProvider}.
 *
 * <p>Bulk 4-at-a-time methods delegate to {@link ClusterANNVectorizationProvider} which
 * loads {@code PanamaBulkVectorOps} on Java 21+ or falls back to {@link DefaultBulkVectorOps}.
 */
public final class ClusterANNVectorUtil {

    private static final BulkVectorOps BULK = ClusterANNVectorizationProvider.getInstance();

    private ClusterANNVectorUtil() {}

    // ========== Single-pair (Lucene SIMD) ==========

    /** L2 squared distance. SIMD-accelerated via Lucene. */
    public static float squareDistance(float[] a, float[] b) {
        return VectorUtil.squareDistance(a, b);
    }

    /** Dot product. SIMD-accelerated via Lucene. */
    public static float dotProduct(float[] a, float[] b) {
        return VectorUtil.dotProduct(a, b);
    }

    /** Cosine similarity. SIMD-accelerated via Lucene. */
    public static float cosine(float[] a, float[] b) {
        return VectorUtil.cosine(a, b);
    }

    // ========== Bulk 4-at-a-time (Panama or scalar) ==========

    /**
     * Compute L2² from query to 4 vectors in a single pass.
     * Uses Panama SIMD on Java 21+, scalar fallback otherwise.
     */
    public static void squareDistanceBulk(float[] q, float[] v0, float[] v1, float[] v2, float[] v3, float[] distances) {
        BULK.squareDistanceBulk(q, v0, v1, v2, v3, distances);
    }

    /**
     * Compute SOAR adjusted distance from a vector to 4 candidate centroids.
     * Uses Panama SIMD on Java 21+, scalar fallback otherwise.
     */
    public static void soarDistanceBulk(
        float[] vec,
        float[] c0,
        float[] c1,
        float[] c2,
        float[] c3,
        float[] residual,
        float soarLambda,
        float residualNormSq,
        float[] distances
    ) {
        BULK.soarDistanceBulk(vec, c0, c1, c2, c3, residual, soarLambda, residualNormSq, distances);
    }

    /**
     * Find the nearest centroid using bulk 4-at-a-time distance.
     *
     * @param vector    the query vector
     * @param centroids array of centroid vectors
     * @return index of the nearest centroid
     */
    public static int findNearestCentroid(float[] vector, float[][] centroids) {
        float[] distances = new float[4];
        int bestIdx = 0;
        float bestDist = Float.MAX_VALUE;
        int limit = centroids.length - 3;
        int i = 0;
        for (; i < limit; i += 4) {
            squareDistanceBulk(vector, centroids[i], centroids[i + 1], centroids[i + 2], centroids[i + 3], distances);
            for (int j = 0; j < 4; j++) {
                if (distances[j] < bestDist) {
                    bestDist = distances[j];
                    bestIdx = i + j;
                }
            }
        }
        for (; i < centroids.length; i++) {
            float d = squareDistance(vector, centroids[i]);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
