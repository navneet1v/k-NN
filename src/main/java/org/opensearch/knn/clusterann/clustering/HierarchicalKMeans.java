/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

import lombok.Builder;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.clusterann.math.VectorMath;
import org.opensearch.knn.clusterann.parallel.ParallelVectorProcessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
    /**
     * Number of nearest top-level centroids whose leaves are scanned during the final coarse-to-fine
     * leaf assignment. Probing more than one bounds the boundary approximation (a vector whose
     * globally nearest leaf descends from a top-level centroid other than its single nearest).
     */
    private static final int ASSIGN_PROBE_TOP = 4;

    private HierarchicalKMeans() {}

    /**
     * Cluster vectors into balanced groups with SOAR secondary assignments.
     *
     * @param vectors       input vectors
     * @param config        clustering configuration
     * @return result with flat centroid array, primary assignments, and centroid count
     */
    public static Result cluster(FloatVectorValues vectors, Config config) throws IOException {
        return cluster(vectors, config, null);
    }

    /**
     * Cluster with optional pre-selected initial centroids (for merge path).
     */
    public static Result cluster(FloatVectorValues vectors, Config config, float[][] initialCentroids) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();

        if (n == 0) {
            return new Result(new float[0][], new int[0], 0, dim);
        }

        // Single centroid for tiny datasets
        if (n <= config.targetSize) {
            float[][] centroids = new float[][] { computeMean(vectors, indices(n)) };
            int[] assignments = new int[n];
            return new Result(centroids, assignments, 1, dim);
        }

        // Compute k for top level: n / targetSize, capped at maxK
        int k = Math.max(2, Math.min(MAX_K_PER_LEVEL, (n + config.targetSize / 2) / config.targetSize));

        // Build k-means config: use reservoir sampling init (fast, no random I/O)
        KMeans.Config kmeansConfig = config.toKMeansConfig();

        // Top-level clustering
        KMeans.Result topResult;
        if (initialCentroids != null && initialCentroids.length >= k) {
            // Merge path: use reservoir-sampled centroids, skip k-means++ init
            float[][] trimmed = Arrays.copyOf(initialCentroids, k);
            topResult = KMeans.cluster(vectors, k, kmeansConfig, trimmed);
        } else {
            topResult = KMeans.cluster(vectors, k, kmeansConfig);
        }

        // Check if any cluster needs splitting (counts already tallied by KMeans)
        int[] counts = topResult.counts();
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
            // No oversized clusters — done in one level. Drop any empty (zero-member) clusters so
            // we don't emit zero-member leaf centroids (which inflate numCentroids and yield empty
            // posting lists), matching the split branch and splitRecursive.
            Compacted compacted = dropEmptyClusters(topResult.centroids(), topResult.assignments(), counts);
            centroids = compacted.centroids();
            assignments = compacted.assignments();
        } else {
            // Recursively split oversized clusters. We keep, for each emitted leaf centroid, the
            // index of the top-level centroid it descends from (leafParents). That mapping lets the
            // final assignment be coarse-to-fine (search leaves under the nearest few top-level
            // centroids) instead of brute-forcing every vector against every leaf, which on the
            // large-merge path is O(n * numCentroids * dim) ~= O(n^2 / targetSize * dim).
            List<float[]> centroidList = new ArrayList<>();
            List<Integer> leafParents = new ArrayList<>();

            for (int c = 0; c < k; c++) {
                if (counts[c] > splitThreshold) {
                    // Oversized top-level cluster: delegate the entire cluster-into-k /
                    // recurse-or-emit / skip-empties logic to splitRecursive rather than
                    // hand-unrolling its first iteration here. splitRecursive terminates on
                    // MAX_DEPTH / SPLIT_THRESHOLD, so no separate leaf-count cap is needed. Each
                    // returned leaf descends from top-level centroid c, so tag it with parent c for
                    // the coarse-to-fine assignment below.
                    int[] clusterIndices = extractClusterIndices(topResult.assignments(), c, counts[c]);
                    List<float[]> leaves = splitRecursive(vectors, clusterIndices, config, kmeansConfig, 1);
                    for (float[] leaf : leaves) {
                        centroidList.add(leaf);
                        leafParents.add(c);
                    }
                } else if (counts[c] > 0) {
                    // Skip empty non-oversized top-level clusters for the same reason
                    // splitRecursive skips them: a zero-member centroid inflates numCentroids and
                    // yields an empty posting list downstream.
                    centroidList.add(topResult.centroids()[c]);
                    leafParents.add(c);
                }
            }

            centroids = centroidList.toArray(new float[0][]);
            assignments = assignLeavesCoarseToFine(vectors, n, centroids, leafParents, topResult.centroids(), config);
        }

        return new Result(centroids, assignments, centroids.length, dim);
    }

    /**
     * Assign every vector to its nearest leaf centroid using a coarse-to-fine search: for each
     * vector, find the nearest {@link #ASSIGN_PROBE_TOP} top-level centroids, then scan only the
     * leaves that descend from those top-level clusters.
     *
     * <p>This replaces a flat scan of all leaf centroids ({@code O(n * numCentroids * dim)}) with
     * roughly {@code O(n * (numTop + probe * avgLeavesPerTop) * dim)}, which matters on the
     * large-merge path where {@code numCentroids} grows with {@code n}. Probing the nearest few
     * top-level centroids (not just the single nearest) keeps the result equal to the exact
     * nearest-leaf assignment except for the rare vector whose globally nearest leaf sits under a
     * top-level centroid outside its probe set — a boundary case bounded by {@code ASSIGN_PROBE_TOP}.
     */
    private static int[] assignLeavesCoarseToFine(
        FloatVectorValues vectors,
        int n,
        float[][] leafCentroids,
        List<Integer> leafParents,
        float[][] topCentroids,
        Config config
    ) throws IOException {
        int numTop = topCentroids.length;
        int numLeaves = leafCentroids.length;
        int probe = Math.min(ASSIGN_PROBE_TOP, numTop);

        // Group leaf indices by their parent top-level centroid, so a probed top-level cluster
        // maps directly to the leaves to scan. Built once, shared read-only across workers.
        List<List<Integer>> leavesByTop = new ArrayList<>(numTop);
        for (int t = 0; t < numTop; t++) {
            leavesByTop.add(new ArrayList<>());
        }
        for (int leaf = 0; leaf < numLeaves; leaf++) {
            leavesByTop.get(leafParents.get(leaf)).add(leaf);
        }

        int[] assignments = new int[n];
        int[] finalAssignments = assignments;
        VectorMath.DistanceFunction distanceFn = VectorMath.distanceFunction(config.metric);

        // Each worker gets its own vector view (thread-safe over encrypted IndexInput);
        // ParallelVectorProcessor runs inline when the executor is null.
        ParallelVectorProcessor.execute(vectors, n, config.executor, (start, end, view) -> {
            // Per-worker scratch for the nearest-`probe` top-level centroids (indices + distances).
            int[] topIdx = new int[probe];
            float[] topDist = new float[probe];
            for (int i = start; i < end; i++) {
                float[] vec = view.vectorValue(i);

                // Coarse step: nearest `probe` top-level centroids via a tiny insertion-sorted list.
                int filled = 0;
                for (int t = 0; t < numTop; t++) {
                    float d = distanceFn.distance(vec, topCentroids[t]);
                    if (filled < probe) {
                        int p = filled++;
                        while (p > 0 && topDist[p - 1] > d) {
                            topDist[p] = topDist[p - 1];
                            topIdx[p] = topIdx[p - 1];
                            p--;
                        }
                        topDist[p] = d;
                        topIdx[p] = t;
                    } else if (d < topDist[probe - 1]) {
                        int p = probe - 1;
                        while (p > 0 && topDist[p - 1] > d) {
                            topDist[p] = topDist[p - 1];
                            topIdx[p] = topIdx[p - 1];
                            p--;
                        }
                        topDist[p] = d;
                        topIdx[p] = t;
                    }
                }

                // Fine step: scan only the leaves under the probed top-level clusters.
                float bestDist = Float.MAX_VALUE;
                int bestC = -1;
                for (int j = 0; j < filled; j++) {
                    for (int leaf : leavesByTop.get(topIdx[j])) {
                        float dist = distanceFn.distance(vec, leafCentroids[leaf]);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestC = leaf;
                        }
                    }
                }
                // Fallback: if the probed subtrees held no leaf (possible only if a probed
                // top-level cluster contributed none), scan all leaves so every vector is assigned.
                if (bestC < 0) {
                    for (int leaf = 0; leaf < numLeaves; leaf++) {
                        float dist = distanceFn.distance(vec, leafCentroids[leaf]);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestC = leaf;
                        }
                    }
                }
                finalAssignments[i] = bestC;
            }
        });
        return assignments;
    }

    /**
     * Recursively split a cluster's vectors until all sub-clusters are within target size.
     *
     * <p>Clusters {@code indices} into k sub-clusters; each oversized sub-cluster recurses, each
     * in-size sub-cluster emits its centroid, and empty sub-clusters are skipped. Recursion
     * terminates on {@link #SPLIT_THRESHOLD} (sub-cluster small enough) or {@link #MAX_DEPTH}.
     */
    private static List<float[]> splitRecursive(
        FloatVectorValues allVectors,
        int[] indices,
        Config config,
        KMeans.Config kmeansConfig,
        int depth
    ) throws IOException {
        int n = indices.length;
        int splitThreshold = (int) (config.targetSize * SPLIT_THRESHOLD);

        if (n <= splitThreshold || depth >= MAX_DEPTH) {
            List<float[]> result = new ArrayList<>();
            result.add(computeMean(allVectors, indices));
            return result;
        }

        int k = Math.max(2, Math.min(MAX_K_PER_LEVEL, (n + config.targetSize / 2) / config.targetSize));
        FloatVectorValues levelSubset = extractSubset(allVectors, indices);
        KMeans.Result kResult = KMeans.cluster(levelSubset, k, kmeansConfig);

        int[] counts = kResult.counts();
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

    /** Compacted (empty-cluster-free) centroids with assignments remapped to the new indices. */
    record Compacted(float[][] centroids, int[] assignments) {
    }

    /**
     * Drops zero-member clusters and remaps {@code assignments} through the old-&gt;new centroid
     * index map. Used by the single-level ({@code !needsSplit}) path, which keeps KMeans'
     * assignments verbatim (no final re-assignment pass), so assignments must be remapped when
     * centroids are compacted or they would point at wrong/removed centroids. Returns the inputs
     * unchanged when there are no empty clusters. Package-private for direct unit testing.
     */
    static Compacted dropEmptyClusters(float[][] centroids, int[] assignments, int[] counts) {
        boolean hasEmpty = false;
        for (int count : counts) {
            if (count == 0) {
                hasEmpty = true;
                break;
            }
        }
        if (!hasEmpty) {
            return new Compacted(centroids, assignments);
        }

        int[] remap = new int[centroids.length];
        Arrays.fill(remap, -1);
        List<float[]> kept = new ArrayList<>(centroids.length);
        for (int c = 0; c < centroids.length; c++) {
            if (counts[c] == 0) {
                continue;
            }
            remap[c] = kept.size();
            kept.add(centroids[c]);
        }

        int[] remapped = new int[assignments.length];
        for (int i = 0; i < assignments.length; i++) {
            int a = assignments[i];
            // Every assigned vector belongs to a non-empty cluster, so remap[a] >= 0.
            remapped[i] = (a >= 0 && a < remap.length) ? remap[a] : 0;
        }
        return new Compacted(kept.toArray(new float[0][]), remapped);
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

    /**
     * Collect the parent (original) ordinals of the sub-cluster's members. {@code subAssignments[i]}
     * is a sub-cluster index for the {@code i}-th member of the parent slice, and
     * {@code parentIndices[i]} is that member's ordinal in the original {@code FloatVectorValues}.
     * For every member assigned to {@code cluster}, we therefore emit {@code parentIndices[i]} (the
     * original ordinal), not {@code i} — so the returned indices index back into the full input,
     * which is what deeper {@code splitRecursive} / {@code computeMean} calls read from.
     */
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

    private static FloatVectorValues extractSubset(FloatVectorValues allVectors, int[] indices) throws IOException {
        if (indices.length == allVectors.size()) {
            return allVectors;
        }
        // Zero-copy ordinal-remapping view: no per-vector cloning, so an oversized top-level
        // cluster (up to ~n/2 vectors on a large merge) costs only the int[] index array rather
        // than a full heap copy of its vectors. copy() returns a view over allVectors.copy(), so
        // each ParallelVectorProcessor worker still reads through its own IndexInput (required for
        // thread-safe reads over the encrypted CryptoBufferedIndexInput).
        return new SubsetView(allVectors, indices);
    }

    /**
     * A read-only, zero-copy <b>view</b> of a subset of a parent {@link FloatVectorValues}: view
     * ordinal {@code i} maps to {@code parent.vectorValue(indices[i])}. It stores only the parent
     * reference and the {@code int[] indices} array — no vector data of its own.
     *
     * <p><b>Why this exists.</b> {@link HierarchicalKMeans} splits an oversized cluster by running
     * k-means over just that cluster's members, and {@link KMeans} needs those members as a
     * {@link FloatVectorValues} it can random-access across multiple Lloyd iterations. There are two
     * ways to produce "the members of cluster c as a FloatVectorValues":
     * <ol>
     *   <li><b>Materialize</b> them into a new on-heap {@code FloatVectorValues.fromFloats(list)} by
     *       cloning each member vector, or</li>
     *   <li><b>View</b> them by remapping ordinals onto the parent (this class).</li>
     * </ol>
     * The materializing approach allocates a second copy of the cluster's vectors. This method is
     * called on <b>top-level</b> oversized clusters (before any recursion), and top-level {@code k}
     * is only clamped to {@code MAX_K_PER_LEVEL}; on a large merge a single oversized cluster can
     * hold a large fraction of {@code n} (millions of vectors). Cloning that many {@code dim}-length
     * arrays can allocate gigabytes and OOM — precisely on the large-merge path this class targets.
     * The view instead costs one {@code int[]} regardless of how big the cluster is, so memory stays
     * bounded by the index array, not the vector data.
     *
     * <p><b>Thread-safety / correctness.</b> {@link ParallelVectorProcessor} hands each worker its
     * own {@link #copy()} so concurrent reads never share a backing reader — required because the
     * underlying (possibly encrypted {@code CryptoBufferedIndexInput}) reader is not thread-safe and
     * a shared reader would corrupt decrypted state under concurrent seeks (surfacing as garbage
     * vectors or {@code CorruptIndexException}). {@link #copy()} therefore returns a new view over a
     * fresh {@code parent.copy()} rather than sharing this view's parent. Reads are safe under
     * buffer reuse because callers ({@code KMeans}, {@link #computeMean}) consume each returned
     * vector immediately and never retain it across the next read; nothing here needs to clone.
     *
     * <p>This mirrors the original implementation's subset supplier, which was dropped during the
     * port to a Lucene-only package and replaced by a materializing copy; the view restores the
     * bounded-memory behavior.
     */
    private static final class SubsetView extends FloatVectorValues {
        private final FloatVectorValues parent;
        private final int[] indices;

        SubsetView(FloatVectorValues parent, int[] indices) {
            this.parent = parent;
            this.indices = indices;
        }

        /**
         * Maps a view ordinal to the parent's vector: {@code viewOrdinal} indexes into this view's
         * {@code indices} array (i.e. {@code [0, indices.length)}), and the value there is the
         * parent ordinal actually read. It is NOT a parent ordinal itself — callers always pass a
         * view ordinal in {@code [0, size())}, which {@code indices[...]} then remaps.
         */
        @Override
        public float[] vectorValue(int viewOrdinal) throws IOException {
            return parent.vectorValue(indices[viewOrdinal]);
        }

        @Override
        public int size() {
            return indices.length;
        }

        @Override
        public int dimension() {
            return parent.dimension();
        }

        /**
         * Returns a new view over an independent {@code parent.copy()} (not {@code this}), so each
         * parallel worker reads through its own backing reader — see the thread-safety note above.
         */
        @Override
        public SubsetView copy() throws IOException {
            return new SubsetView(parent.copy(), indices);
        }

        @Override
        public KnnVectorValues.DocIndexIterator iterator() {
            return createDenseIterator();
        }
    }

    private static float[] computeMean(FloatVectorValues vectors, int[] indices) throws IOException {
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
        return mean;
    }

    private static int[] indices(int n) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
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

    @Builder
    public static final class Config {
        @Builder.Default
        final int targetSize = 512;
        @Builder.Default
        final int maxIterations = 10;
        @Builder.Default
        final float samplePercentage = 0.1f;
        @Builder.Default
        final VectorSimilarityFunction metric = VectorSimilarityFunction.EUCLIDEAN;
        /**
         * Seed for all randomized steps (k-means++ init and training-subset sampling), threaded
         * through to {@link KMeans.Config#seed}. Fixing it makes clustering reproducible: the same
         * vectors and seed yield identical centroids, which keeps segment merges and tests
         * deterministic.
         */
        @Builder.Default
        final long seed = 42L;
        /** Executor driving parallel assignment; {@code null} runs inline (sequential). */
        final TaskExecutor executor;

        /**
         * Derive the inner {@link KMeans.Config} for per-level clustering. This is the single place
         * that maps hierarchical settings onto k-means settings — the omitted k-means fields
         * (e.g. {@code rebalanceEmpty}) intentionally keep their k-means defaults. Oversized-cluster
         * splitting is handled here, not inside k-means, so k-means exposes no cluster-size bound.
         */
        KMeans.Config toKMeansConfig() {
            return KMeans.Config.builder()
                .metric(metric)
                .maxIterations(maxIterations)
                .seed(seed)
                .executor(executor)
                .samplePercentage(samplePercentage)
                .build();
        }
    }
}
