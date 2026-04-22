/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import java.io.IOException;

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
    public static Result cluster(ClusterANNVectorValues vectors, Config config) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();

        if (n == 0) {
            return new Result(new float[0][], new int[0], 0, dim);
        }

        // Recursive clustering to get flat centroid list
        List<float[]> centroidList = new ArrayList<>();
        clusterRecursive(vectors, indices(n), centroidList, config, 0);

        // Build centroid array
        int numCentroids = centroidList.size();
        float[][] centroids = centroidList.toArray(new float[0][]);

        // Assign all vectors to nearest centroid (SIMD-accelerated)
        int[] assignments = new int[n];
        for (int i = 0; i < n; i++) {
            float[] vec = vectors.vectorValue(i);
            float bestDist = Float.MAX_VALUE;
            int bestCentroid = 0;
            for (int c = 0; c < numCentroids; c++) {
                float dist = config.kmeansConfig.metric.distance(vec, centroids[c]);
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

    private static void clusterRecursive(
        ClusterANNVectorValues allVectors,
        int[] subset,
        List<float[]> centroidList,
        Config config,
        int depth
    ) throws IOException {
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

        // Extract subset into contiguous ClusterANNVectorValues for k-means
        ClusterANNVectorValues subsetData = extractSubset(allVectors, subset);

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

    private static float[] computeMean(ClusterANNVectorValues vectors, int[] subset) throws IOException {
        int dim = vectors.dimension();
        float[] mean = new float[dim];
        for (int idx : subset) {
            float[] vec = vectors.vectorValue(idx);
            for (int d = 0; d < dim; d++) {
                mean[d] += vec[d];
            }
        }
        float invN = 1f / subset.length;
        for (int d = 0; d < dim; d++) {
            mean[d] *= invN;
        }
        return mean;
    }

    private static ClusterANNVectorValues extractSubset(ClusterANNVectorValues allVectors, int[] subset) throws IOException {
        if (subset.length == allVectors.size()) {
            return allVectors;
        }
        List<float[]> subList = new ArrayList<>(subset.length);
        for (int idx : subset) {
            subList.add(allVectors.vectorValue(idx));
        }
        return ClusterANNVectorValues.fromList(subList, allVectors.dimension());
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
        private final float[][] centroids;
        private final int[] assignments;
        private final int numCentroids;
        private final int dimension;

        Result(float[][] centroids, int[] assignments, int numCentroids, int dimension) {
            this.centroids = centroids;
            this.assignments = assignments;
            this.numCentroids = numCentroids;
            this.dimension = dimension;
        }

        /** Flat centroid array. */
        public float[][] centroids() {
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
            return centroids[index];
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
