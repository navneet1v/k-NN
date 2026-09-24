/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

import lombok.Builder;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.clusterann.math.VectorMath;
import org.opensearch.knn.clusterann.parallel.ParallelVectorProcessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Production-grade K-Means clustering with k-means++ initialization, Lloyd's iterations,
 * empty cluster rebalancing, and oversized cluster post-processing.
 *
 * <p>Performance optimizations:
 * <ul>
 *   <li>Operates on flat {@link FloatVectorValues} — cache-line friendly sequential access</li>
 *   <li>Incremental centroid updates — O(moved × d) per iteration vs O(n × d)</li>
 *   <li>Thread-safe convergence via {@link AtomicInteger} (no memory fence per vector)</li>
 *   <li>Parallel assignment step with configurable threading</li>
 * </ul>
 *
 * <p><b>Metric handling.</b> Point-to-centroid assignment and k-means++ initialization always use
 * squared L2 distance; {@code config.metric} affects only centroid <em>update</em> (spherical
 * unit-norm normalization for {@code COSINE}). This is correct for {@code EUCLIDEAN} and
 * {@code COSINE} — on unit-normalized centroids, L2 ordering matches cosine ordering — but for
 * inner-product metrics ({@code DOT_PRODUCT} / {@code MAXIMUM_INNER_PRODUCT}) assignment is an L2
 * approximation, not the exact metric. The IVF clustering here targets L2/cosine; exact
 * inner-product assignment is a deliberate non-goal at this layer.
 *
 * <p>Algorithm reference: Arthur &amp; Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (2007)
 */
public final class KMeans {

    private KMeans() {} // static utility

    private static final int PROXIMITY_MAP_SIZE = 8;

    /**
     * Offset added to the configured seed for the training-subset sampler, so its random sequence
     * is decorrelated from the k-means++ init sequence (which uses the raw seed).
     */
    private static final int SAMPLE_SEED_OFFSET = 13;

    /** Minimum vector count below which iteration-subset sampling is skipped (cluster on full data). */
    private static final int MIN_VECTORS_FOR_SAMPLING = 10_000;

    /**
     * Cluster vectors into k groups using k-means++ initialization.
     *
     * <p>Thread-safe: uses {@code vectors.copy()} per worker for parallel execution.
     *
     * @param vectors input vectors (not modified)
     * @param k       number of clusters (clamped to [1, numVectors])
     * @param config  clustering configuration (iterations, seed, parallelism, metric)
     * @return result with centroids, assignments, and inertia
     * @throws IOException if vector access fails
     */
    public static Result cluster(FloatVectorValues vectors, int k, Config config) throws IOException {
        return cluster(vectors, k, config, null);
    }

    /**
     * Cluster vectors into k groups, optionally with pre-selected initial centroids.
     *
     * @param initialCentroids if non-null, skip k-means++ init and use these directly
     */
    public static Result cluster(FloatVectorValues vectors, int k, Config config, float[][] initialCentroids) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        if (n == 0) {
            // Guard the public entry point: initCentroids calls rng.nextInt(n), which throws on
            // n == 0. HierarchicalKMeans guards this before delegating, but direct callers do not.
            return new Result(new float[0][], new int[0], new int[0], 0, dim, 0, true);
        }
        k = Math.max(1, Math.min(k, n));

        // When caller-supplied centroids are used verbatim, their length must match the clamped k:
        // assignmentStep derives its cluster count from centroids.length, while clusterSums/
        // clusterCounts are sized with k here. A mismatch (e.g. more initial centroids than
        // vectors, since k is clamped to n) would index those arrays out of bounds. Reject early
        // with a clear error rather than throwing an ArrayIndexOutOfBoundsException mid-clustering.
        if (initialCentroids != null && initialCentroids.length != k) {
            throw new IllegalArgumentException(
                "initialCentroids.length (" + initialCentroids.length + ") must equal k (" + k + ") after clamping to [1, n=" + n + "]"
            );
        }

        // Deep-copy caller-supplied initial centroids: the k-means iteration mutates centroids[c][d]
        // in place (updateCentroids/sanitizeCentroids/rebalanceEmptyClusters), which would otherwise
        // overwrite the caller's array — and, since HierarchicalKMeans seeds these from a shallow
        // Arrays.copyOf of the reservoir sample, the sample's inner float[] too. A shallow copy is
        // insufficient because the inner rows are what get mutated.
        float[][] centroids;
        if (initialCentroids != null) {
            centroids = new float[k][];
            for (int c = 0; c < k; c++) {
                centroids[c] = initialCentroids[c].clone();
            }
        } else {
            centroids = initCentroids(vectors, k, config);
        }

        float[][] clusterSums = new float[k][dim];
        int[] clusterCounts = new int[k];
        int[] assignments = new int[n];
        Arrays.fill(assignments, -1);

        // Sampling: run iterations on a percentage-based subset, then finalize on full data.
        Sample sample = sampleForIterations(vectors, n, dim, k, config);
        boolean needsSampling = sample.indices() != null;
        FloatVectorValues iterVectors = sample.vectors();

        // Sampled assignments (maps sample ordinal -> cluster)
        int[] iterAssignments = needsSampling ? new int[iterVectors.size()] : assignments;
        if (needsSampling) {
            Arrays.fill(iterAssignments, -1);
        }

        boolean converged = false;
        boolean lastIterationRebalanced = false;
        int iter;
        // Neighborhood-aware: first 2 iterations check all centroids, then use centroidProximityMap
        int[][] centroidProximityMap = null;
        for (iter = 0; iter < config.maxIterations && !converged; iter++) {
            if (iter == 2 && k > PROXIMITY_MAP_SIZE * 2) {
                sanitizeCentroids(centroids, k, dim);
                centroidProximityMap = computeCentroidProximityMap(centroids, k);
            }
            int moved = assignmentStep(iterVectors, centroids, iterAssignments, clusterSums, clusterCounts, config, centroidProximityMap);
            updateCentroids(centroids, clusterSums, clusterCounts, k, config.metric);
            sanitizeCentroids(centroids, k, dim);

            boolean rebalanced = false;
            if (config.rebalanceEmpty) {
                rebalanced = rebalanceEmptyClusters(iterVectors, centroids, clusterCounts, iterAssignments, k, config);
                if (rebalanced) {
                    centroidProximityMap = null; // invalidate only when a rebalance actually moved centroids
                }
            }
            lastIterationRebalanced = rebalanced;

            // A rebalance splits a cluster to re-seed empty ones, mutating `centroids` without
            // updating `assignments`/`clusterCounts`. If we let this iteration converge, the loop
            // exits with the split centroid having no members and (in the non-sampling path, which
            // has no final assignment pass) returns centroids inconsistent with assignments/counts.
            // Treat any rebalancing iteration as not-yet-converged so the next iteration re-assigns
            // against the split centroids and recomputes counts.
            converged = !rebalanced && ((moved == 0) || ((float) moved / iterVectors.size() < config.convergenceThreshold));
        }

        // Final full pass: assign all vectors to the final centroids. Always required in the
        // sampling path (iterations ran on a subset). Also required in the non-sampling path when
        // the loop's last iteration rebalanced — that split centroids without re-assigning, and
        // hitting the iteration cap means the veto above had no next iteration to reconcile them,
        // so counts/assignments would otherwise be inconsistent with the returned centroids.
        if (needsSampling || lastIterationRebalanced) {
            assignmentStep(vectors, centroids, assignments, clusterSums, clusterCounts, config, centroidProximityMap);
            updateCentroids(centroids, clusterSums, clusterCounts, k, config.metric);
            sanitizeCentroids(centroids, k, dim);
        }

        // Recompute counts from the final assignments as the source of truth (robust to any
        // divergence from the running accumulator). Oversized-cluster splitting is owned by
        // HierarchicalKMeans, not KMeans — see Config for why there is no maxClusterSize knob here.
        int[] finalCounts = new int[k];
        for (int a : assignments) {
            if (a >= 0 && a < k) {
                finalCounts[a]++;
            }
        }
        return new Result(centroids, assignments, finalCounts, k, dim, iter, converged);
    }

    // ========== Initialization ==========

    private static float[][] initCentroids(FloatVectorValues vectors, int k, Config config) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        float[][] centroids = new float[k][dim];
        Random rng = new Random(config.seed);

        // First centroid: random
        int first = rng.nextInt(n);
        System.arraycopy(vectors.vectorValue(first), 0, centroids[0], 0, dim);

        if (k == 1) return centroids;

        // K-means++ initialization
        float[] minDistSq = new float[n];
        Arrays.fill(minDistSq, Float.MAX_VALUE);

        for (int c = 1; c < k; c++) {
            float[] prevCentroid = centroids[c - 1];
            float totalWeight = 0f;
            for (int i = 0; i < n; i++) {
                // k-means++ seeding is L2-based regardless of the configured metric (see class doc).
                float d = VectorMath.squareDistance(vectors.vectorValue(i), prevCentroid);
                if (d < minDistSq[i]) {
                    minDistSq[i] = d;
                }
                totalWeight += minDistSq[i];
            }

            // Weighted random selection (D^2 sampling)
            float target = rng.nextFloat() * totalWeight;
            float cumulative = 0f;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) {
                cumulative += minDistSq[i];
                if (cumulative >= target) {
                    chosen = i;
                    break;
                }
            }

            System.arraycopy(vectors.vectorValue(chosen), 0, centroids[c], 0, dim);
        }

        return centroids;
    }

    /**
     * Precompute nearest neighbors for each centroid (L2-based; assignment is L2 — see class doc).
     * Returns centroidProximityMap[c] = sorted array of nearest centroid indices for centroid c.
     */
    private static int[][] computeCentroidProximityMap(float[][] centroids, int k) {
        int proximitySize = Math.min(PROXIMITY_MAP_SIZE, k - 1);
        int[][] centroidProximityMap = new int[k][];
        VectorMath.DistanceFunction l2 = VectorMath::squareDistance;
        for (int c = 0; c < k; c++) {
            centroidProximityMap[c] = VectorMath.nearestCentroids(centroids, c, proximitySize, l2);
        }
        return centroidProximityMap;
    }

    // ========== Assignment Step ==========

    private static int assignmentStep(
        FloatVectorValues vectors,
        float[][] centroids,
        int[] assignments,
        float[][] clusterSums,
        int[] clusterCounts,
        Config config,
        int[][] centroidProximityMap
    ) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        int k = centroids.length;

        float[] flatCentroids = VectorMath.flattenCentroids(centroids);

        // Reset sums and counts
        for (float[] s : clusterSums) {
            Arrays.fill(s, 0f);
        }
        Arrays.fill(clusterCounts, 0);

        AtomicInteger movedCount = new AtomicInteger(0);

        // One partitioned pass over [0, n). Each worker accumulates into its own private
        // sums/counts and records them as a Partial keyed by slice start. We do NOT merge
        // into the shared accumulators inside the workers: float addition is not associative,
        // so merging in nondeterministic thread-arrival order would make clusterSums (and thus
        // the resulting centroids/assignments) differ run-to-run, breaking the reproducibility
        // guarantee documented on Config.seed. Instead we collect the partials and reduce them
        // afterwards in a fixed slice order (see below).
        List<Partial> partials = Collections.synchronizedList(new ArrayList<>());
        ParallelVectorProcessor.execute(vectors, n, config.executor, (start, end, view) -> {
            float[] distBuf = new float[k];
            float[][] localSums = new float[k][dim];
            int[] localCounts = new int[k];
            int localMoved = 0;

            for (int i = start; i < end; i++) {
                float[] vec = view.vectorValue(i);
                int bestCluster;
                int prevCluster = assignments[i];

                if (centroidProximityMap != null && prevCluster >= 0) {
                    float bestDist = VectorMath.squareDistance(vec, centroids[prevCluster]);
                    bestCluster = prevCluster;
                    for (int nc : centroidProximityMap[prevCluster]) {
                        float dist = VectorMath.squareDistance(vec, centroids[nc]);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestCluster = nc;
                        }
                    }
                } else {
                    bestCluster = VectorMath.findNearestCentroidBulk(vec, flatCentroids, k, dim, distBuf);
                }

                if (assignments[i] != bestCluster) {
                    localMoved++;
                    assignments[i] = bestCluster;
                }

                localCounts[bestCluster]++;
                for (int d = 0; d < dim; d++) {
                    localSums[bestCluster][d] += vec[d];
                }
            }

            // Record this worker's partials; the merge happens deterministically after the pass.
            partials.add(new Partial(start, localSums, localCounts));
            movedCount.addAndGet(localMoved);
        });

        // Reduce the per-slice partials in ascending slice-start order. Slices are contiguous and
        // non-overlapping, so `start` is a stable, run-independent key. Reducing in this fixed
        // order makes the (non-associative) float summation identical on every run, whether or not
        // an executor is supplied — preserving the Config.seed reproducibility contract.
        partials.sort(Comparator.comparingInt(p -> p.start));
        for (Partial p : partials) {
            for (int c = 0; c < k; c++) {
                clusterCounts[c] += p.counts[c];
                for (int d = 0; d < dim; d++) {
                    clusterSums[c][d] += p.sums[c][d];
                }
            }
        }

        return movedCount.get();
    }

    /** Per-slice partial sums/counts, reduced in a fixed {@code start} order for reproducibility. */
    private record Partial(int start, float[][] sums, int[] counts) {
    }

    // ========== Centroid Update ==========

    /** Every non-empty cluster's centroid from its member sum; see {@link VectorMath#centroidFromSum} for what that is per metric. */
    private static void updateCentroids(
        float[][] centroids,
        float[][] clusterSums,
        int[] clusterCounts,
        int k,
        VectorSimilarityFunction metric
    ) {
        for (int c = 0; c < k; c++) {
            if (clusterCounts[c] > 0) {
                VectorMath.centroidFromSum(clusterSums[c], clusterCounts[c], metric, centroids[c]);
            }
        }
    }

    // ========== Centroid Sanitization ==========

    /** Replace any non-finite centroid with zeros (will be treated as empty and rebalanced). */
    private static void sanitizeCentroids(float[][] centroids, int k, int dim) {
        for (int c = 0; c < k; c++) {
            for (int d = 0; d < dim; d++) {
                if (!Float.isFinite(centroids[c][d])) {
                    // Zero out entire centroid
                    Arrays.fill(centroids[c], 0f);
                    break;
                }
            }
        }
    }

    // ========== Empty Cluster Rebalancing ==========

    /**
     * Splits the largest cluster along its max-variance dimension to re-seed each empty cluster.
     *
     * @return {@code true} if at least one empty cluster was split (centroids changed), {@code false}
     *     if there was nothing to rebalance. Callers use this to avoid invalidating the centroid
     *     proximity map when no centroids actually moved.
     */
    private static boolean rebalanceEmptyClusters(
        FloatVectorValues vectors,
        float[][] centroids,
        int[] clusterCounts,
        int[] assignments,
        int k,
        Config config
    ) throws IOException {
        int dim = vectors.dimension();

        // Find largest cluster
        int largestCluster = 0;
        for (int c = 1; c < k; c++) {
            if (clusterCounts[c] > clusterCounts[largestCluster]) {
                largestCluster = c;
            }
        }
        if (clusterCounts[largestCluster] < 2) {
            return false;
        }

        boolean rebalanced = false;
        // PCA-inspired splitting: find dimension of max variance in largest cluster
        for (int c = 0; c < k; c++) {
            if (clusterCounts[c] != 0) continue;
            rebalanced = true;

            // Find dimension with max variance in largest cluster
            float[] mean = centroids[largestCluster];
            float maxVar = 0f;
            int splitDim = 0;
            float[] variances = new float[dim];
            int count = 0;
            for (int i = 0; i < assignments.length; i++) {
                if (assignments[i] != largestCluster) continue;
                float[] vec = vectors.vectorValue(i);
                for (int d = 0; d < dim; d++) {
                    float diff = vec[d] - mean[d];
                    variances[d] += diff * diff;
                }
                count++;
            }
            if (count > 0) {
                for (int d = 0; d < dim; d++) {
                    if (variances[d] > maxVar) {
                        maxVar = variances[d];
                        splitDim = d;
                    }
                }
            }

            // Split: perturb along max-variance dimension
            float offset = (float) Math.sqrt(maxVar / Math.max(count, 1)) * 0.5f;
            if (offset < 1e-6f) offset = config.perturbation;
            System.arraycopy(centroids[largestCluster], 0, centroids[c], 0, dim);
            centroids[c][splitDim] += offset;
            centroids[largestCluster][splitDim] -= offset;
        }
        return rebalanced;
    }

    // ========== Post-Processing ==========

    /**
     * Iteration input for k-means: either a random training subset (when {@code indices != null})
     * or the full input unchanged (when {@code indices == null}). {@code indices} maps each sample
     * ordinal back to its original ordinal in the full input.
     */
    private record Sample(FloatVectorValues vectors, int[] indices) {
    }

    /**
     * Selects the subset of vectors to run Lloyd iterations on. Sampling is skipped (returning the
     * original vectors with {@code indices == null}) unless the input is large enough
     * ({@code n > MIN_VECTORS_FOR_SAMPLING}) and the configured percentage actually reduces the set.
     */
    private static Sample sampleForIterations(FloatVectorValues vectors, int n, int dim, int k, Config config) throws IOException {
        int sampleSize = Math.max(k, (int) (n * config.samplePercentage));
        if (sampleSize >= n || n <= MIN_VECTORS_FOR_SAMPLING) {
            return new Sample(vectors, null);
        }
        int[] sampleIndices = createRandomSample(n, sampleSize, config.seed);
        List<float[]> sampleList = new ArrayList<>(sampleSize);
        for (int idx : sampleIndices) {
            sampleList.add(vectors.vectorValue(idx).clone());
        }
        return new Sample(FloatVectorValues.fromFloats(sampleList, dim), sampleIndices);
    }

    /** Fisher-Yates partial shuffle to select sampleSize random indices from [0, n). */
    private static int[] createRandomSample(int n, int sampleSize, long seed) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        Random rng = new Random(seed + SAMPLE_SEED_OFFSET);
        for (int i = 0; i < sampleSize; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        return Arrays.copyOf(indices, sampleSize);
    }

    // ========== Result ==========

    /**
     * Immutable clustering result.
     */
    public static final class Result {
        private final float[][] centroids;
        private final int[] assignments;
        private final int[] counts;
        private final int k;
        private final int dimension;
        private final int iterations;
        private final boolean converged;

        Result(float[][] centroids, int[] assignments, int[] counts, int k, int dimension, int iterations, boolean converged) {
            this.centroids = centroids;
            this.assignments = assignments;
            this.counts = counts;
            this.k = k;
            this.dimension = dimension;
            this.iterations = iterations;
            this.converged = converged;
        }

        /** Flat centroid array: centroids[c * dimension + d]. */
        public float[][] centroids() {
            return centroids;
        }

        /** Assignment of each vector to a cluster index. */
        public int[] assignments() {
            return assignments;
        }

        /** Number of vectors assigned to each cluster (length k). Computed during the final assignment pass. */
        public int[] counts() {
            return counts;
        }

        /** Number of clusters. */
        public int k() {
            return k;
        }

        /** Vector dimension. */
        public int dimension() {
            return dimension;
        }

        /** Number of iterations run. */
        public int iterations() {
            return iterations;
        }

        /** Whether convergence was reached. */
        public boolean converged() {
            return converged;
        }

        /** Get centroid as a copy. */
        public float[] getCentroid(int clusterIndex) {
            return centroids[clusterIndex];
        }
    }

    // ========== Configuration ==========

    /**
     * K-Means configuration. Use {@code Config.builder()} for construction.
     */
    @Builder(toBuilder = true)
    public static final class Config {
        @Builder.Default
        final VectorSimilarityFunction metric = VectorSimilarityFunction.EUCLIDEAN;
        @Builder.Default
        final int maxIterations = 20;
        @Builder.Default
        final float convergenceThreshold = 0.001f;
        /**
         * Seed for all randomized steps: k-means++ initialization and the training-subset
         * sampler (whose sequence is offset from init so the two do not correlate). Fixing it
         * makes clustering reproducible for a given input.
         */
        @Builder.Default
        final long seed = 42L;
        /** Executor driving parallel assignment; {@code null} runs inline (sequential). */
        final TaskExecutor executor;
        @Builder.Default
        final boolean rebalanceEmpty = true;
        @Builder.Default
        final float perturbation = 0.01f;
        @Builder.Default
        final float samplePercentage = 0.1f;

        /** Default config: EUCLIDEAN, 20 iterations, sequential (no executor), seed=42. */
        public static Config defaults() {
            return builder().build();
        }
    }
}
