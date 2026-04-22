/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

/**
 * Distance metrics for vector similarity computation.
 *
 * <p>Single-pair {@link #distance(float[], float[])} delegates to {@link ClusterANNVectorUtil}
 * which uses Lucene's SIMD-accelerated {@code VectorUtil} under the hood.
 *
 * <p>Offset-based overloads operate on flat centroid arrays where vectors are packed contiguously.
 *
 * <p>All metrics return a value where <b>lower is more similar</b> (distance semantics).
 * For inner product and cosine, the raw similarity is negated/inverted to maintain this invariant.
 */
public enum DistanceMetric {

    /**
     * Squared Euclidean distance: sum((a[i] - b[i])^2).
     * No square root — monotonic, avoids expensive sqrt in comparisons.
     */
    L2 {
        @Override
        public float distance(float[] a, float[] b) {
            return ClusterANNVectorUtil.squareDistance(a, b);
        }

        @Override
        public float distance(float[] data, int offsetA, int offsetB, int dimension) {
            float sum = 0f;
            for (int i = 0; i < dimension; i++) {
                float diff = data[offsetA + i] - data[offsetB + i];
                sum = Math.fma(diff, diff, sum);
            }
            return sum;
        }

        @Override
        public float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension) {
            float sum = 0f;
            for (int i = 0; i < dimension; i++) {
                float diff = a[offsetA + i] - b[offsetB + i];
                sum = Math.fma(diff, diff, sum);
            }
            return sum;
        }
    },

    /**
     * Negated inner product: -sum(a[i] * b[i]).
     * Negated so that lower = more similar (higher dot product = closer).
     */
    INNER_PRODUCT {
        @Override
        public float distance(float[] a, float[] b) {
            return -ClusterANNVectorUtil.dotProduct(a, b);
        }

        @Override
        public float distance(float[] data, int offsetA, int offsetB, int dimension) {
            float dot = 0f;
            for (int i = 0; i < dimension; i++) {
                dot = Math.fma(data[offsetA + i], data[offsetB + i], dot);
            }
            return -dot;
        }

        @Override
        public float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension) {
            float dot = 0f;
            for (int i = 0; i < dimension; i++) {
                dot = Math.fma(a[offsetA + i], b[offsetB + i], dot);
            }
            return -dot;
        }
    },

    /**
     * Cosine distance: 1 - cosine_similarity(a, b).
     * Range [0, 2] where 0 = identical direction.
     */
    COSINE {
        @Override
        public float distance(float[] a, float[] b) {
            return 1f - ClusterANNVectorUtil.cosine(a, b);
        }

        @Override
        public float distance(float[] data, int offsetA, int offsetB, int dimension) {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < dimension; i++) {
                float ai = data[offsetA + i];
                float bi = data[offsetB + i];
                dot = Math.fma(ai, bi, dot);
                normA = Math.fma(ai, ai, normA);
                normB = Math.fma(bi, bi, normB);
            }
            float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
            return denom < 1e-10f ? 1f : 1f - dot / denom;
        }

        @Override
        public float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension) {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < dimension; i++) {
                float ai = a[offsetA + i];
                float bi = b[offsetB + i];
                dot = Math.fma(ai, bi, dot);
                normA = Math.fma(ai, ai, normA);
                normB = Math.fma(bi, bi, normB);
            }
            float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
            return denom < 1e-10f ? 1f : 1f - dot / denom;
        }
    };

    /** Compute distance between two vectors. Lower = more similar. SIMD-accelerated. */
    public abstract float distance(float[] a, float[] b);

    /**
     * Compute distance between two vectors in the same flat array.
     *
     * @param data      flat backing array (packed centroids)
     * @param offsetA   offset of first vector
     * @param offsetB   offset of second vector
     * @param dimension vector dimensionality
     */
    public abstract float distance(float[] data, int offsetA, int offsetB, int dimension);

    /**
     * Compute distance between vectors in different arrays at given offsets.
     */
    public abstract float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension);
}
