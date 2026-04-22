/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import java.io.IOException;
import java.io.UncheckedIOException;

import java.util.Arrays;
import java.util.BitSet;
import java.util.PriorityQueue;
import java.util.stream.IntStream;

/**
 * Inverted File (IVF) index with SOAR (Spilling with Orthogonal Augmented Residuals)
 * secondary assignments for improved recall.
 *
 * <p>Architecture:
 * <ul>
 *   <li>Small datasets (≤256 vectors): flat k-means clustering</li>
 *   <li>Large datasets: {@link HierarchicalKMeans} with adaptive splitting</li>
 *   <li>SOAR: each vector gets a secondary centroid assignment based on orthogonal
 *       residual projection, stored as inverted posting lists for O(1) lookup</li>
 * </ul>
 *
 * <p>Search probes the top-nprobe centroids, scanning both primary and SOAR posting
 * lists for each probed centroid. SOAR typically improves recall by 5-15% at minimal
 * cost since posting lists are pre-built.
 *
 * <p>Reference: "SOAR: Improved Indexing for Approximate Nearest Neighbor Search"
 * (Sun et al., NeurIPS 2023)
 */
public final class IVFIndex {

    /** Threshold below which flat k-means is used instead of hierarchical. */
    /** Maximum sub-clusters per hierarchical level. */
    /** Maximum SOAR candidate centroids to evaluate per vector. */
    private static final int SOAR_CANDIDATE_LIMIT = 10;

    private final float[][] centroids;
    private final int numCentroids;
    private final int dimension;
    private final DistanceMetric metric;

    // Primary posting lists: primaryPostings[centroidId] = sorted docIds
    private final int[][] primaryPostings;

    // SOAR posting lists: soarPostings[centroidId] = sorted docIds assigned as secondary
    private final int[][] soarPostings;

    private IVFIndex(
        float[][] centroids,
        int numCentroids,
        int dimension,
        DistanceMetric metric,
        int[][] primaryPostings,
        int[][] soarPostings
    ) {
        this.centroids = centroids;
        this.numCentroids = numCentroids;
        this.dimension = dimension;
        this.metric = metric;
        this.primaryPostings = primaryPostings;
        this.soarPostings = soarPostings;
    }

    /**
     * Build an IVF index from vectors.
     *
     * @param vectors input vectors
     * @param config  index configuration
     * @return built IVF index ready for search
     */
    public static IVFIndex build(ClusterANNVectorValues vectors, Config config) throws IOException {
        return build(vectors, config, null);
    }

    /**
     * Build an IVF index, optionally with pre-selected initial centroids (skips k-means++ init).
     */
    public static IVFIndex build(ClusterANNVectorValues vectors, Config config, float[][] initialCentroids) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();

        // Single path: HierarchicalKMeans handles everything
        // - Small k: effectively one level of flat k-means (no recursion needed)
        // - Large k: automatically splits oversized clusters
        HierarchicalKMeans.Config hConfig = HierarchicalKMeans.Config.builder()
            .targetSize(config.targetClusterSize)
            .maxIterations(config.kmeansIterations)
            .samplePercentage(0.1f)
            .metric(config.metric)
            .seed(config.seed)
            .parallel(config.parallel)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, hConfig, initialCentroids);
        float[][] centroids = result.centroids();
        int[] assignments = result.assignments();
        int numCentroids = result.numCentroids();

        // Build primary posting lists
        int[][] primaryPostings = buildPostingLists(assignments, numCentroids);

        // Compute SOAR secondary assignments and build inverted posting lists
        int[][] soarPostings;
        if (config.soarLambda > 0) {
            int[] soarAssignments = computeSOAR(vectors, centroids, assignments, numCentroids, config);
            soarPostings = buildPostingLists(soarAssignments, numCentroids);
        } else {
            soarPostings = new int[numCentroids][0];
        }

        return new IVFIndex(centroids, numCentroids, dim, config.metric, primaryPostings, soarPostings);
    }

    /**
     * Search for k nearest neighbors.
     *
     * @param query  query vector (length = dimension)
     * @param k      number of results
     * @param nprobe number of centroids to probe
     * @param vectors original vectors for distance computation
     * @return top-k results sorted by distance (ascending)
     */
    public SearchResult[] search(float[] query, int k, int nprobe, ClusterANNVectorValues vectors) throws IOException {
        nprobe = Math.min(nprobe, numCentroids);

        // Find nprobe nearest centroids
        int[] probeCentroids = findNearestCentroids(query, nprobe);

        // Max-heap for top-k (evict farthest when full)
        PriorityQueue<SearchResult> topK = new PriorityQueue<>(k + 1, (a, b) -> Float.compare(b.distance, a.distance));

        // Track seen docIds to avoid duplicate distance computations
        BitSet seen = new BitSet(vectors.size());
        for (int centId : probeCentroids) {
            // Scan primary posting list
            for (int docId : primaryPostings[centId]) {
                if (!seen.get(docId)) {
                    seen.set(docId);
                    float dist = metric.distance(query, vectors.vectorValue(docId));
                    insertResult(topK, k, docId, dist);
                }
            }

            // Scan SOAR posting list
            for (int docId : soarPostings[centId]) {
                if (!seen.get(docId)) {
                    seen.set(docId);
                    float dist = metric.distance(query, vectors.vectorValue(docId));
                    insertResult(topK, k, docId, dist);
                }
            }
        }

        // Convert to sorted array
        SearchResult[] results = topK.toArray(new SearchResult[0]);
        Arrays.sort(results, (a, b) -> Float.compare(a.distance, b.distance));
        return results;
    }

    // ========== SOAR Computation ==========

    private static int[] computeSOAR(
        ClusterANNVectorValues vectors,
        float[][] centroids,
        int[] assignments,
        int numCentroids,
        Config config
    ) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        int[] soarAssignments = new int[n];
        Arrays.fill(soarAssignments, -1);

        // #8 fix: skip SOAR if only 1 centroid
        if (numCentroids <= 1) return soarAssignments;

        // Precompute nearest centroids per centroid for SOAR candidate selection
        int candidateLimit = Math.min(numCentroids - 1, SOAR_CANDIDATE_LIMIT);
        int[][] nearestCentroids = new int[numCentroids][candidateLimit];
        for (int c = 0; c < numCentroids; c++) {
            float[] dists = new float[numCentroids];
            int[] idx = new int[numCentroids];
            for (int j = 0; j < numCentroids; j++) {
                dists[j] = (j == c) ? Float.MAX_VALUE : config.metric.distance(centroids[c], centroids[j]);
                idx[j] = j;
            }
            for (int i = 0; i < candidateLimit; i++) {
                int minIdx = i;
                for (int j = i + 1; j < numCentroids; j++) {
                    if (dists[idx[j]] < dists[idx[minIdx]]) minIdx = j;
                }
                int tmp = idx[i];
                idx[i] = idx[minIdx];
                idx[minIdx] = tmp;
            }
            System.arraycopy(idx, 0, nearestCentroids[c], 0, candidateLimit);
        }

        IntStream.range(0, n).parallel().forEach(i -> {
            float[] vec;
            try {
                vec = vectors.vectorValue(i);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            int primaryCent = assignments[i];
            float[] primaryCentroid = centroids[primaryCent];

            float residualNormSq = 0f;
            for (int d = 0; d < dim; d++) {
                float r = vec[d] - primaryCentroid[d];
                residualNormSq += r * r;
            }
            float residualNorm = (float) Math.sqrt(residualNormSq);
            if (residualNorm < 1e-10f) return;
            float invNorm = 1f / residualNorm;

            float bestDist = Float.MAX_VALUE;
            int bestCent = -1;

            for (int nc : nearestCentroids[primaryCent]) {
                float[] cent = centroids[nc];
                float dsq = 0f;
                float proj = 0f;
                for (int d = 0; d < dim; d++) {
                    float diff = vec[d] - cent[d];
                    dsq += diff * diff;
                    float residualD = (vec[d] - primaryCentroid[d]) * invNorm;
                    proj += residualD * diff;
                }

                float soarDist = dsq + config.soarLambda * (proj * proj);
                if (soarDist < bestDist) {
                    bestDist = soarDist;
                    bestCent = nc;
                }
            }

            soarAssignments[i] = bestCent;
        });

        return soarAssignments;
    }

    // ========== Helpers ==========

    private int[] findNearestCentroids(float[] query, int nprobe) {
        // Min-heap of (distance, centroidId)
        float[] dists = new float[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            dists[c] = metric.distance(query, centroids[c]);
        }

        // Partial sort: find top-nprobe smallest
        int[] indices = new int[numCentroids];
        for (int i = 0; i < numCentroids; i++)
            indices[i] = i;

        // Simple selection for small nprobe
        for (int i = 0; i < nprobe; i++) {
            int minIdx = i;
            for (int j = i + 1; j < numCentroids; j++) {
                if (dists[indices[j]] < dists[indices[minIdx]]) {
                    minIdx = j;
                }
            }
            int tmp = indices[i];
            indices[i] = indices[minIdx];
            indices[minIdx] = tmp;
        }

        return Arrays.copyOf(indices, nprobe);
    }

    private static int[][] buildPostingLists(int[] assignments, int numCentroids) {
        // Count per centroid
        int[] counts = new int[numCentroids];
        for (int a : assignments) {
            if (a >= 0 && a < numCentroids) counts[a]++;
        }

        // Allocate
        int[][] postings = new int[numCentroids][];
        for (int c = 0; c < numCentroids; c++) {
            postings[c] = new int[counts[c]];
        }

        // Fill
        int[] pos = new int[numCentroids];
        for (int i = 0; i < assignments.length; i++) {
            int a = assignments[i];
            if (a >= 0 && a < numCentroids) {
                postings[a][pos[a]++] = i;
            }
        }

        return postings;
    }

    private static void insertResult(PriorityQueue<SearchResult> topK, int k, int docId, float dist) {
        if (topK.size() < k) {
            topK.offer(new SearchResult(docId, dist));
        } else if (dist < topK.peek().distance) {
            topK.poll();
            topK.offer(new SearchResult(docId, dist));
        }
    }

    // ========== Accessors ==========

    public float[][] centroids() {
        return centroids;
    }

    public int numCentroids() {
        return numCentroids;
    }

    public int dimension() {
        return dimension;
    }

    public int[][] primaryPostings() {
        return primaryPostings;
    }

    public int[][] soarPostings() {
        return soarPostings;
    }

    // ========== Search Result ==========

    public static final class SearchResult {
        public final int docId;
        public final float distance;

        public SearchResult(int docId, float distance) {
            this.docId = docId;
            this.distance = distance;
        }
    }

    // ========== Configuration ==========

    public static final class Config {
        final int numCentroids;
        final int targetClusterSize;
        final int maxDepth;
        final int kmeansIterations;
        final DistanceMetric metric;
        final float soarLambda;
        final long seed;
        final boolean parallel;

        private Config(Builder b) {
            this.numCentroids = b.numCentroids;
            this.targetClusterSize = b.targetClusterSize;
            this.maxDepth = b.maxDepth;
            this.kmeansIterations = b.kmeansIterations;
            this.metric = b.metric;
            this.soarLambda = b.soarLambda;
            this.seed = b.seed;
            this.parallel = b.parallel;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private int numCentroids = 1024;
            private int targetClusterSize = 512;
            private int maxDepth = 10;
            private int kmeansIterations = 20;
            private DistanceMetric metric = DistanceMetric.L2;
            private float soarLambda = 1.0f;
            private long seed = 42L;
            private boolean parallel = true;

            public Builder numCentroids(int n) {
                this.numCentroids = n;
                return this;
            }

            public Builder targetClusterSize(int t) {
                this.targetClusterSize = t;
                return this;
            }

            public Builder maxDepth(int d) {
                this.maxDepth = d;
                return this;
            }

            public Builder kmeansIterations(int i) {
                this.kmeansIterations = i;
                return this;
            }

            public Builder metric(DistanceMetric m) {
                this.metric = m;
                return this;
            }

            public Builder soarLambda(float l) {
                this.soarLambda = l;
                return this;
            }

            public Builder seed(long s) {
                this.seed = s;
                return this;
            }

            public Builder parallel(boolean p) {
                this.parallel = p;
                return this;
            }

            public Config build() {
                return new Config(this);
            }
        }
    }
}
