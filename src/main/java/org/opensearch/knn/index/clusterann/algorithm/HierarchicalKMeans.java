/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

import org.opensearch.knn.index.clusterann.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Adaptive hierarchical k-means with automatic splitting and SOAR.
 *
 * <p>Unlike a rigid flat-vs-hierarchical decision, this always uses the same path:
 * one level of flat k-means, then recursively splits only oversized clusters.
 * For small k (≤ maxK), this is effectively flat k-means. For large k, it
 * naturally becomes hierarchical.
 *
 * <p>Key differences from other implementations:
 * <ul>
 *   <li>ScaNN-inspired: learned split threshold (1.5× target)</li>
 *   <li>Reservoir sampling init at every level (no k-means++ overhead)</li>
 *   <li>Balanced splitting: oversized clusters get sub-k proportional to their excess</li>
 *   <li>Single code path for all dataset sizes</li>
 * </ul>
 */
public final class HierarchicalKMeans {

    private static final int MAX_K_PER_LEVEL = 128;
    private static final int MAX_DEPTH = 10;
    private static final float SPLIT_THRESHOLD = 1.5f;

    private HierarchicalKMeans() {}

    /**
     * Cluster vectors into balanced groups with SOAR secondary assignments.
     *
     * @param vectors       input vectors
     * @param config        clustering configuration
     * @return result with flat centroid array, primary assignments, and centroid count
     */
    public static Result cluster(ClusterANNVectorValues vectors, Config config) throws IOException {
        return cluster(vectors, config, null);
    }

    /**
     * Cluster with optional pre-selected initial centroids (for merge path).
     */
    public static Result cluster(ClusterANNVectorValues vectors, Config config, float[][] initialCentroids) throws IOException {
        return cluster(vectors, config, initialCentroids, null);
    }

    /**
     * Donor-seed merge path: when {@code carriedAssignment} is provided, bypass the hierarchical
     * split/rebuild and run a flat donor-seeded settle ({@link KMeans#cluster} with carried
     * assignments) over the donor's centroids. Donor docs keep their cell; newcomers route.
     */
    public static Result cluster(ClusterANNVectorValues vectors, Config config,
                                 float[][] initialCentroids, int[] carriedAssignment) throws IOException {
        if (carriedAssignment != null && initialCentroids != null) {
            return donorSeedCluster(vectors, config, initialCentroids, carriedAssignment);
        }
        return clusterImpl(vectors, config, initialCentroids);
    }

    /**
     * Donor-seed merge: donor docs keep their carried cell (no re-cluster), newcomers route to the
     * nearest donor centroid. Then split any overfull cell (> targetSize × SPLIT_THRESHOLD) using
     * the SAME local split as the cold path — but re-assign ONLY the split cells' members, so donor
     * docs in non-split cells keep their assignment. This grows the donor's (possibly coarse)
     * centroid set to the granularity the merged size needs, without re-clustering the whole corpus.
     */
    private static Result donorSeedCluster(ClusterANNVectorValues vectors, Config config,
                                            float[][] donorCentroids, int[] carried) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        int k = donorCentroids.length;

        // Donor adequacy gate: if the donor is far too coarse for the merged size, splitting would
        // re-cluster nearly everything (no benefit, sometimes slower), so fall back to cold. When a
        // dominant, well-clustered donor exists (the incremental-merge case this targets), it passes.
        int targetK = Math.max(1, (n + config.targetSize / 2) / config.targetSize);
        if (k < targetK / 2) {
            org.apache.logging.log4j.LogManager.getLogger(HierarchicalKMeans.class).info(
                "[ClusterANN-MERGE-TRACE] donor inadequate: donorCentroids={} < target/2={} (n={}); falling back to cold clustering",
                k, targetK / 2, n);
            return clusterImpl(vectors, config, null);
        }
        int metricOrd = config.metric == DistanceMetric.COSINE ? 2 : (config.metric == DistanceMetric.INNER_PRODUCT ? 1 : 0);

        // 1. Route: carried docs keep their cell; newcomers -> nearest donor centroid (parallel).
        int[] assignments = new int[n];
        float[] flat = new float[k * dim];
        for (int i = 0; i < k; i++) System.arraycopy(donorCentroids[i], 0, flat, i * dim, dim);
        IntStream.range(0, n).parallel().forEach(ord -> {
            int c = carried[ord];
            if (c >= 0 && c < k) { assignments[ord] = c; return; }
            try {
                float[] vec = vectors.vectorValue(ord);
                float[] distBuf = new float[k];
                assignments[ord] = ClusterANNVectorUtil.findNearestCentroidBulk(vec, flat, k, dim, distBuf, metricOrd);
            } catch (IOException e) { throw new java.io.UncheckedIOException(e); }
        });

        // SELF-CHECK: sample carried docs and verify their carried cell IS their nearest donor
        // centroid. A high mismatch rate means the carried ord->cell mapping is misaligned (the
        // bug we chased). Logs the mismatch fraction so we can see correctness at a glance.
        {
            int sampled = 0, mismatch = 0;
            float[] db = new float[k];
            int step = Math.max(1, n / 2000);
            for (int ord = 0; ord < n; ord += step) {
                int cc = carried[ord];
                if (cc < 0 || cc >= k) continue;
                float[] vec = vectors.vectorValue(ord);
                int nearest = ClusterANNVectorUtil.findNearestCentroidBulk(vec, flat, k, dim, db, metricOrd);
                sampled++;
                if (nearest != cc) mismatch++;
            }
            if (sampled > 0) {
                org.apache.logging.log4j.LogManager.getLogger(HierarchicalKMeans.class).info(
                    "[ClusterANN-MERGE-TRACE] carried self-check: {}/{} carried docs are NOT at their nearest donor centroid ({}%)",
                    mismatch, sampled, (100 * mismatch / sampled));
            }
        }

        // 2. Split overfull cells (same threshold as cold path). Re-cluster ONLY those cells'
        //    members; non-split cells keep their assignment. LIRE-style local split.
        int[] counts = clusterCounts(assignments, k);
        int splitThreshold = (int) (config.targetSize * SPLIT_THRESHOLD);
        KMeans.Config kmeansConfig = KMeans.Config.builder()
            .metric(config.metric).maxIterations(config.maxIterations)
            .seed(config.seed).parallel(config.parallel).samplePercentage(config.samplePercentage).build();

        List<float[]> centroidList = new ArrayList<>(k * 2);
        int[] cellRemap = new int[k];                 // old cell -> new index (non-split only)
        java.util.Arrays.fill(cellRemap, -1);
        int[] finalAssign = new int[n];               // built fresh to avoid index collisions
        java.util.Arrays.fill(finalAssign, -1);

        // Non-split cells: keep centroid, remap their docs.
        for (int c = 0; c < k; c++) {
            if (counts[c] <= splitThreshold) {
                cellRemap[c] = centroidList.size();
                centroidList.add(donorCentroids[c]);
            }
        }
        for (int ord = 0; ord < n; ord++) {
            int c = assignments[ord];
            if (c >= 0 && c < k && cellRemap[c] >= 0) finalAssign[ord] = cellRemap[c];
        }
        // Split cells: local sub-k-means, append sub-centroids, assign only their members.
        for (int c = 0; c < k; c++) {
            if (counts[c] <= splitThreshold) continue;
            int[] idx = extractClusterIndices(assignments, c, counts[c]);
            ClusterANNVectorValues subset = extractSubset(vectors, idx);
            int subK = Math.max(2, Math.min(MAX_K_PER_LEVEL, (counts[c] + config.targetSize / 2) / config.targetSize));
            KMeans.Result sub = KMeans.cluster(subset, subK, kmeansConfig);
            int base = centroidList.size();
            for (float[] sc : sub.centroids()) centroidList.add(sc);
            int[] subAssign = sub.assignments();
            for (int i = 0; i < idx.length; i++) finalAssign[idx[i]] = base + subAssign[i];
        }

        float[][] centroids = centroidList.toArray(new float[0][]);
        int fk = centroids.length;

        // Route + split is sufficient AND correct: donor docs keep their (valid) cell, newcomers
        // route to nearest, overfull cells split to the right granularity. A subsequent KMeans
        // "settle" pass was found to DEGRADE recall (its proximity-map neighborhood assignment and
        // re-init desynced centroids from assignments), so we do NOT settle — route+split is final.
        // (Opt-in settle retained behind a flag for experiments only.)
        if (!Boolean.getBoolean("clusterann.mergeSettle")) {
            return new Result(centroids, finalAssign, fk, dim);
        }
        int settleIters = Integer.getInteger("clusterann.mergeSettleIters", 3);
        KMeans.Config settleCfg = KMeans.Config.builder()
            .metric(config.metric).maxIterations(settleIters)
            .seed(config.seed).parallel(config.parallel).samplePercentage(1.0f).build();
        KMeans.Result settled = KMeans.cluster(vectors, fk, settleCfg, centroids);
        return new Result(settled.centroids(), settled.assignments(), settled.k(), dim);
    }

    private static void normalizeInPlace(float[] v) {
        double s = 0; for (float x : v) s += (double) x * x;
        if (s > 0) { float inv = (float) (1.0 / Math.sqrt(s)); for (int i = 0; i < v.length; i++) v[i] *= inv; }
    }

    private static Result clusterImpl(ClusterANNVectorValues vectors, Config config, float[][] initialCentroids) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();

        if (n == 0) {
            return new Result(new float[0][], new int[0], 0, dim);
        }

        // Single centroid for tiny datasets
        if (n <= config.targetSize) {
            float[][] centroids = new float[][] { computeMean(vectors, indices(n), config) };
            int[] assignments = new int[n];
            return new Result(centroids, assignments, 1, dim);
        }

        // Compute k for top level: n / targetSize, capped at maxK
        int k = Math.max(2, Math.min(MAX_K_PER_LEVEL, (n + config.targetSize / 2) / config.targetSize));

        // Build k-means config: use reservoir sampling init (fast, no random I/O)
        KMeans.Config kmeansConfig = KMeans.Config.builder()
            .metric(config.metric)
            .maxIterations(config.maxIterations)
            .seed(config.seed)
            .parallel(config.parallel)
            .samplePercentage(config.samplePercentage)
            .build();

        // Top-level clustering
        KMeans.Result topResult;
        if (initialCentroids != null && initialCentroids.length >= k) {
            // Merge path: use reservoir-sampled centroids, skip k-means++ init
            float[][] trimmed = Arrays.copyOf(initialCentroids, k);
                topResult = KMeans.cluster(vectors, k, kmeansConfig, trimmed);
        } else {
            topResult = KMeans.cluster(vectors, k, kmeansConfig);
        }

        // Check if any cluster needs splitting
        int[] counts = clusterCounts(topResult.assignments(), k);
        int splitThreshold = (int) (config.targetSize * SPLIT_THRESHOLD);
        boolean needsSplit = false;
        for (int count : counts) {
            if (count > splitThreshold) {
                needsSplit = true;
                break;
            }
        }

        float[][] centroids;
        int[] assignments;

        if (!needsSplit) {
            // No oversized clusters — done in one level
            centroids = topResult.centroids();
            assignments = topResult.assignments();
        } else {
            // Recursively split oversized clusters
            List<float[]> centroidList = new ArrayList<>();
            int[] centroidMapping = new int[k]; // maps old centroid idx → new base idx

            for (int c = 0; c < k; c++) {
                centroidMapping[c] = centroidList.size();

                if (counts[c] > splitThreshold) {
                    // Extract vectors for this cluster
                    int[] clusterIndices = extractClusterIndices(topResult.assignments(), c, counts[c]);
                    ClusterANNVectorValues subset = extractSubset(vectors, clusterIndices);

                    // Recurse with proportional sub-k
                    int subK = Math.max(2, Math.min(MAX_K_PER_LEVEL, (counts[c] + config.targetSize / 2) / config.targetSize));
                    KMeans.Result subResult = KMeans.cluster(subset, subK, kmeansConfig);

                    // Check for further splitting needed
                    int[] subCounts = clusterCounts(subResult.assignments(), subK);
                    boolean subNeedsSplit = false;
                    for (int sc : subCounts) {
                        if (sc > splitThreshold) {
                            subNeedsSplit = true;
                            break;
                        }
                    }

                    if (subNeedsSplit && centroidList.size() < 4096) {
                        // Deep recursion via recursive call
                        List<float[]> subCentroids = splitRecursive(vectors, clusterIndices, config, kmeansConfig, 1);
                        centroidList.addAll(subCentroids);
                    } else {
                        for (float[] sc : subResult.centroids()) {
                            centroidList.add(sc);
                        }
                    }
                } else {
                    centroidList.add(topResult.centroids()[c]);
                }
            }

            centroids = centroidList.toArray(new float[0][]);

            // Final assignment: assign all vectors to nearest leaf centroid
            int numCentroids = centroids.length;
            assignments = new int[n];
            float[][] finalCentroids = centroids;
            IntStream.range(0, n).parallel().forEach(i -> {
                try {
                    float[] vec = vectors.vectorValue(i);
                    float bestDist = Float.MAX_VALUE;
                    int bestC = 0;
                    for (int c = 0; c < numCentroids; c++) {
                        float dist = config.metric.distance(vec, finalCentroids[c]);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestC = c;
                        }
                    }
                    assignments[i] = bestC;
                } catch (IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            });
        }

        return new Result(centroids, assignments, centroids.length, dim);
    }

    /**
     * Recursively split a cluster's vectors until all sub-clusters are within target size.
     */
    private static List<float[]> splitRecursive(
        ClusterANNVectorValues allVectors,
        int[] indices,
        Config config,
        KMeans.Config kmeansConfig,
        int depth
    ) throws IOException {
        int n = indices.length;
        int splitThreshold = (int) (config.targetSize * SPLIT_THRESHOLD);

        if (n <= splitThreshold || depth >= MAX_DEPTH) {
            List<float[]> result = new ArrayList<>();
            result.add(computeMean(allVectors, indices, config));
            return result;
        }

        int k = Math.max(2, Math.min(MAX_K_PER_LEVEL, (n + config.targetSize / 2) / config.targetSize));
        ClusterANNVectorValues subset = extractSubset(allVectors, indices);
        KMeans.Result kResult = KMeans.cluster(subset, k, kmeansConfig);

        int[] counts = clusterCounts(kResult.assignments(), k);
        List<float[]> allCentroids = new ArrayList<>();

        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) continue;

            if (counts[c] > splitThreshold) {
                int[] subIndices = extractClusterOriginalIndices(kResult.assignments(), c, counts[c], indices);
                allCentroids.addAll(splitRecursive(allVectors, subIndices, config, kmeansConfig, depth + 1));
            } else {
                allCentroids.add(kResult.centroids()[c]);
            }
        }

        return allCentroids;
    }

    // ========== Helpers ==========

    private static int[] clusterCounts(int[] assignments, int k) {
        int[] counts = new int[k];
        for (int a : assignments) {
            if (a >= 0 && a < k) counts[a]++;
        }
        return counts;
    }

    private static int[] extractClusterIndices(int[] assignments, int cluster, int count) {
        int[] indices = new int[count];
        int pos = 0;
        for (int i = 0; i < assignments.length; i++) {
            if (assignments[i] == cluster) {
                indices[pos++] = i;
            }
        }
        return indices;
    }

    private static int[] extractClusterOriginalIndices(int[] subAssignments, int cluster, int count, int[] parentIndices) {
        int[] indices = new int[count];
        int pos = 0;
        for (int i = 0; i < subAssignments.length; i++) {
            if (subAssignments[i] == cluster) {
                indices[pos++] = parentIndices[i];
            }
        }
        return indices;
    }

    private static ClusterANNVectorValues extractSubset(ClusterANNVectorValues allVectors, int[] indices) throws IOException {
        if (indices.length == allVectors.size()) {
            return allVectors;
        }
        return ClusterANNVectorValues.fromSubset(allVectors, indices);
    }

    private static float[] computeMean(ClusterANNVectorValues vectors, int[] indices, Config config) throws IOException {
        int dim = vectors.dimension();
        float[] mean = new float[dim];
        for (int idx : indices) {
            float[] vec = vectors.vectorValue(idx);
            for (int d = 0; d < dim; d++) {
                mean[d] += vec[d];
            }
        }
        float inv = 1f / indices.length;
        for (int d = 0; d < dim; d++) {
            mean[d] *= inv;
        }
        // Cosine centroids must be unit length: OptimizedScalarQuantizer asserts a unit centroid under
        // cosine, and KMeans.updateCentroids already projects spherical centroids onto the unit sphere.
        // These shortcut paths (single-centroid and splitRecursive leaves) skip KMeans, so project here
        // too — otherwise a cosine corpus with small segments produces non-unit centroids (assertion
        // trip under -ea, silent recall loss otherwise). Mirrors CR-307468808.
        if (config.metric == DistanceMetric.COSINE) {
            normalizeInPlace(mean);
        }
        return mean;
    }

    private static int[] indices(int n) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++)
            idx[i] = i;
        return idx;
    }

    // ========== Result ==========

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

        public float[][] centroids() {
            return centroids;
        }

        public int[] assignments() {
            return assignments;
        }

        public int numCentroids() {
            return numCentroids;
        }

        public int dimension() {
            return dimension;
        }

        public float[] getCentroid(int i) {
            return centroids[i];
        }
    }

    // ========== Config ==========

    public static final class Config {
        final int targetSize;
        final int maxIterations;
        final float samplePercentage;
        final DistanceMetric metric;
        final long seed;
        final boolean parallel;

        private Config(Builder b) {
            this.targetSize = b.targetSize;
            this.maxIterations = b.maxIterations;
            this.samplePercentage = b.samplePercentage;
            this.metric = b.metric;
            this.seed = b.seed;
            this.parallel = b.parallel;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private int targetSize = 512;
            private int maxIterations = 10;
            private float samplePercentage = 0.1f;
            private DistanceMetric metric = DistanceMetric.L2;
            private long seed = 42L;
            private boolean parallel = true;

            public Builder targetSize(int t) {
                this.targetSize = t;
                return this;
            }

            public Builder maxIterations(int m) {
                this.maxIterations = m;
                return this;
            }

            public Builder samplePercentage(float s) {
                this.samplePercentage = s;
                return this;
            }

            public Builder metric(DistanceMetric m) {
                this.metric = m;
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
