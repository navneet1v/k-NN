/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

/**
 * SPI for bulk vector operations. Default implementation uses scalar loops with {@code Math.fma}.
 * A Panama SIMD implementation can be loaded on Java 21+ via {@link ClusterANNVectorizationProvider}.
 */
interface BulkVectorOps {

    /** Compute L2² from query to 4 vectors in one pass. */
    void squareDistanceBulk(float[] q, float[] v0, float[] v1, float[] v2, float[] v3, float[] distances);

    /** Compute SOAR distance from vector to 4 candidate centroids in one pass. */
    void soarDistanceBulk(
        float[] vec,
        float[] c0,
        float[] c1,
        float[] c2,
        float[] c3,
        float[] residual,
        float soarLambda,
        float residualNormSq,
        float[] distances
    );
}
