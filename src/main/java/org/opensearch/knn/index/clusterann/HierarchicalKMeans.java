/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import java.util.ArrayList;
import java.util.List;

/**
 * Adaptive hierarchical K-Means that recursively splits clusters until each is ≤ targetSize.
 *
 * <p>Uses an adaptive splitting formula: {@code k = clamp(ceil(sqrt(n / targetSize)) * 2, 2, maxK)}
 * to determine how many sub-clusters to create at each level. Delegates actual clustering
 * to {@link KMeans} which provides proper Lloyd's iterations with convergence detection.
 *
 * <p>This produces a flat list of centroids suitable for IVF indexing, where the number of
 * centroids adapts to the dataset size rather than being fixed upfront.
 *
 * <p>Algorithm:
 * <ol>
 *   <li>If n ≤ targetSize → return mean as single centroid (leaf)</li>
 *   <li>Compute k using adaptive formula</li>
 *   <li>Run k-means to get k sub-clusters</li>
 *   <li>Recursively split any sub-cluster still &gt; targetSize</li>
 * </ol>
 */
public final class HierarchicalKMeans {

    private HierarchicalKMeans() {} // static utility

    /**
     * Cluster vectors with adaptive hierarchical splitting.
     *
     * @param vectors    input vectors
     * @param config     hierarchical clustering configuration
     * @return flat list of centroids (as contiguous float array) and vector assignments
     */
    public static Result cluster(VectorData vectors, Config config) {
        int n = vectors.numVectors();
        int dim = vectors.dimension();

        if (n == 0) {
            return new Result(new float[0], new int[0], 0, dim);
        }

        // Recursive clustering to get flat centroid list
        List<float[]> centroidList = new ArrayList<>();
        clusterRecursive(vectors, indices(n), centroidList, config, 0);

        // Build flat centroid array
        int numCentroids = centroidList.size();
        float[] centroids = new float[numCentroids * dim];
        for (int i = 0; i < numCentroids; i++) {
            System.arraycopy(centroidList.get(i), 0, centroids, i * dim, dim);
        }

        // Assign all vectors to nearest centroid
        int[] assignments = new int[n];
        float[] data = vectors.data();
        for (int i = 0; i < n; i++) {
            int offset = i * dim;
            float bestDist = Float.MAX_VALUE;
            int bestCentroid = 0;
            for (int c = 0; c < numCentroids; c++) {
                float dist = config.kmeansConfig.metric.distance(data, offset, centroids, c * dim, dim);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCentroid = c;
                }
            }
            assignments[i] = bestCentroid;
        }

        return new Result(centroids, assignments, numCentroids, dim);
    }

    // ========== Recursive Splitting ==========

    private static void clusterRecursive(VectorData allVectors, int[] subset, List<float[]> centroidList, Config config, int depth) {
        int n = subset.length;
        int dim = allVectors.dimension();

        // Base case: small enough or max depth reached
        if (n <= config.targetSize || depth >= config.maxDepth) {
            centroidList.add(computeMean(allVectors, subset));
            return;
        }

        // Adaptive k: use square root scaling for balanced cluster sizes.
        // For small n/targetSize ratios, this produces fewer but larger clusters (less overhead).
        // For large ratios, it grows sub-linearly, avoiding excessive fragmentation.
        // k = clamp(ceil(sqrt(n / targetSize)) * 2, 2, maxK)
        int k = Math.max(2, Math.min(config.maxK, (int) Math.ceil(Math.sqrt((double) n / config.targetSize) * 2)));

        // Extract subset into contiguous VectorData for k-means
        VectorData subsetData = extractSubset(allVectors, subset);

        // Run k-means on subset
        KMeans.Result kmeansResult = KMeans.cluster(subsetData, k, config.kmeansConfig);

        // Group indices by cluster
        int[] subAssignments = kmeansResult.assignments();
        List<List<Integer>> groups = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            groups.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            groups.get(subAssignments[i]).add(subset[i]);
        }

        // Recursively split oversized clusters
        for (int c = 0; c < k; c++) {
            List<Integer> group = groups.get(c);
            if (group.isEmpty()) continue;

            if (group.size() > config.targetSize && depth + 1 < config.maxDepth) {
                int[] subIndices = group.stream().mapToInt(Integer::intValue).toArray();
                clusterRecursive(allVectors, subIndices, centroidList, config, depth + 1);
            } else {
                // Use k-means centroid directly
                centroidList.add(kmeansResult.getCentroid(c));
            }
        }
    }

    // ========== Helpers ==========

    private static float[] computeMean(VectorData vectors, int[] subset) {
        int dim = vectors.dimension();
        float[] data = vectors.data();
        float[] mean = new float[dim];
        for (int idx : subset) {
            int offset = idx * dim;
            for (int d = 0; d < dim; d++) {
                mean[d] += data[offset + d];
            }
        }
        float invN = 1f / subset.length;
        for (int d = 0; d < dim; d++) {
            mean[d] *= invN;
        }
        return mean;
    }

    private static VectorData extractSubset(VectorData allVectors, int[] subset) {
        int dim = allVectors.dimension();
        float[] data = allVectors.data();
        float[] subData = new float[subset.length * dim];
        for (int i = 0; i < subset.length; i++) {
            System.arraycopy(data, subset[i] * dim, subData, i * dim, dim);
        }
        return new VectorData(subData, subset.length, dim);
    }

    private static int[] indices(int n) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++)
            idx[i] = i;
        return idx;
    }

    // ========== Result ==========

    /**
     * Hierarchical clustering result.
     */
    public static final class Result {
        private final float[] centroids;
        private final int[] assignments;
        private final int numCentroids;
        private final int dimension;

        Result(float[] centroids, int[] assignments, int numCentroids, int dimension) {
            this.centroids = centroids;
            this.assignments = assignments;
            this.numCentroids = numCentroids;
            this.dimension = dimension;
        }

        /** Flat centroid array. */
        public float[] centroids() {
            return centroids;
        }

        /** Vector-to-centroid assignments. */
        public int[] assignments() {
            return assignments;
        }

        /** Number of centroids produced. */
        public int numCentroids() {
            return numCentroids;
        }

        /** Vector dimension. */
        public int dimension() {
            return dimension;
        }

        /** Get a single centroid as a copy. */
        public float[] getCentroid(int index) {
            float[] c = new float[dimension];
            System.arraycopy(centroids, index * dimension, c, 0, dimension);
            return c;
        }
    }

    // ========== Configuration ==========

    /**
     * Hierarchical K-Means configuration.
     */
    public static final class Config {
        final int targetSize;
        final int maxK;
        final int maxDepth;
        final KMeans.Config kmeansConfig;

        private Config(Builder b) {
            this.targetSize = b.targetSize;
            this.maxK = b.maxK;
            this.maxDepth = b.maxDepth;
            this.kmeansConfig = b.kmeansConfig;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private int targetSize = 512;
            private int maxK = 128;
            private int maxDepth = 10;
            private KMeans.Config kmeansConfig = KMeans.Config.defaults();

            public Builder targetSize(int t) {
                this.targetSize = t;
                return this;
            }

            public Builder maxK(int m) {
                this.maxK = m;
                return this;
            }

            public Builder maxDepth(int d) {
                this.maxDepth = d;
                return this;
            }

            public Builder kmeansConfig(KMeans.Config c) {
                this.kmeansConfig = c;
                return this;
            }

            public Config build() {
                return new Config(this);
            }
        }
    }
}
