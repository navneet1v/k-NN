/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

/**
 * Distance metrics for vector similarity computation.
 *
 * <p>Each metric operates directly on the flat backing array of {@link VectorData}
 * to avoid allocation and enable SIMD-friendly sequential access patterns.
 *
 * <p>All metrics return a value where <b>lower is more similar</b> (distance semantics).
 * For inner product and cosine, the raw similarity is negated to maintain this invariant.
 */
public enum DistanceMetric {

    /**
     * Squared Euclidean distance: sum((a[i] - b[i])^2).
     * No square root — monotonic, avoids expensive sqrt in comparisons.
     */
    L2 {
        @Override
        public float distance(float[] data, int offsetA, int offsetB, int dimension) {
            float sum = 0f;
            for (int i = 0; i < dimension; i++) {
                float diff = data[offsetA + i] - data[offsetB + i];
                sum += diff * diff;
            }
            return sum;
        }

        @Override
        public float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension) {
            float sum = 0f;
            for (int i = 0; i < dimension; i++) {
                float diff = a[offsetA + i] - b[offsetB + i];
                sum += diff * diff;
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
        public float distance(float[] data, int offsetA, int offsetB, int dimension) {
            float dot = 0f;
            for (int i = 0; i < dimension; i++) {
                dot += data[offsetA + i] * data[offsetB + i];
            }
            return -dot;
        }

        @Override
        public float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension) {
            float dot = 0f;
            for (int i = 0; i < dimension; i++) {
                dot += a[offsetA + i] * b[offsetB + i];
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
        public float distance(float[] data, int offsetA, int offsetB, int dimension) {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < dimension; i++) {
                float ai = data[offsetA + i];
                float bi = data[offsetB + i];
                dot += ai * bi;
                normA += ai * ai;
                normB += bi * bi;
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
                dot += ai * bi;
                normA += ai * ai;
                normB += bi * bi;
            }
            float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
            return denom < 1e-10f ? 1f : 1f - dot / denom;
        }
    };

    /**
     * Compute distance between two vectors in the same backing array.
     *
     * @param data      flat backing array
     * @param offsetA   byte offset of first vector
     * @param offsetB   byte offset of second vector
     * @param dimension vector dimensionality
     * @return distance (lower = more similar)
     */
    public abstract float distance(float[] data, int offsetA, int offsetB, int dimension);

    /**
     * Compute distance between vectors in different arrays.
     *
     * @param a         first array
     * @param offsetA   offset into first array
     * @param b         second array
     * @param offsetB   offset into second array
     * @param dimension vector dimensionality
     * @return distance (lower = more similar)
     */
    public abstract float distance(float[] a, int offsetA, float[] b, int offsetB, int dimension);
}
