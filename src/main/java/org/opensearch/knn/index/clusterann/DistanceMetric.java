/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.opensearch.knn.index.clusterann.algorithm.ClusterANNVectorUtil;

/**
 * Distance metrics for vector similarity computation.
 *
 * <p>Delegates to {@link ClusterANNVectorUtil} which uses Lucene's SIMD-accelerated
 * {@code VectorUtil} under the hood (Panama on capable JVMs).
 *
 * <p>All metrics return a value where <b>lower is more similar</b> (distance semantics).
 * For inner product and cosine, the raw similarity is negated/inverted to maintain this invariant.
 */
public enum DistanceMetric {

    /** Squared Euclidean distance: sum((a[i] - b[i])^2). */
    L2 {
        @Override
        public float distance(float[] a, float[] b) {
            return ClusterANNVectorUtil.squareDistance(a, b);
        }
    },

    /** Negated inner product: -sum(a[i] * b[i]). Lower = more similar. */
    INNER_PRODUCT {
        @Override
        public float distance(float[] a, float[] b) {
            return -ClusterANNVectorUtil.dotProduct(a, b);
        }
    },

    /** Cosine distance: 1 - cosine_similarity(a, b). Range [0, 2]. */
    COSINE {
        @Override
        public float distance(float[] a, float[] b) {
            return 1f - ClusterANNVectorUtil.cosine(a, b);
        }
    };

    /** Compute distance between two vectors. Lower = more similar. SIMD-accelerated. */
    public abstract float distance(float[] a, float[] b);
}
