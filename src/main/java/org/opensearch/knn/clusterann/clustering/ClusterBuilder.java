/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.clusterann.ClusteringResult;
import org.opensearch.knn.clusterann.math.VectorMath;
import org.opensearch.knn.clusterann.parallel.ParallelVectorProcessor;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Builds an IVF index: clustering + SOAR secondary assignments.
 *
 * <p>Consumes Lucene's {@link FloatVectorValues} directly and runs all parallel work on a
 * caller-supplied Lucene {@link TaskExecutor} — typically threaded down from the merge
 * ({@code KnnVectorsWriter}'s merge {@code TaskExecutor}). Passing {@code null} runs everything
 * inline on the calling thread. Tuning parameters (target cluster size, SOAR lambda, seed) live in
 * {@link ClusterBuilderConstants}.
 */
public final class ClusterBuilder {

    private ClusterBuilder() {}

    /**
     * Build IVF clustering with SOAR secondary assignments, using the default spill cap
     * ({@link ClusterBuilderConstants#MAX_SOAR_ASSIGNMENTS}) and margin
     * ({@link ClusterBuilderConstants#SOAR_SPILL_MARGIN}).
     *
     * @param vectors  input vectors (Lucene type)
     * @param metric   distance metric
     * @param executor executor for parallel clustering/SOAR, or {@code null} for inline execution
     * @return clustering result with centroids, primary/SOAR assignments, and their member distances
     */
    public static ClusteringResult build(FloatVectorValues vectors, VectorSimilarityFunction metric, TaskExecutor executor)
        throws IOException {
        return build(vectors, metric, executor, ClusterBuilderConstants.MAX_SOAR_ASSIGNMENTS, ClusterBuilderConstants.SOAR_SPILL_MARGIN);
    }

    /**
     * Build IVF clustering with an explicit SOAR spill cap and margin. Package-private: the public
     * {@link #build(FloatVectorValues, VectorSimilarityFunction, TaskExecutor)} supplies the
     * configured defaults; this overload exists so the spill behavior can be exercised directly.
     *
     * @param maxSoarAssignments max SOAR secondaries per vector ({@code >= 1}; 1 = classic single secondary)
     * @param spillMargin        relative-score margin gating spills beyond the first (see constants)
     */
    static ClusteringResult build(
        FloatVectorValues vectors,
        VectorSimilarityFunction metric,
        TaskExecutor executor,
        int maxSoarAssignments,
        float spillMargin
    ) throws IOException {
        int n = vectors.size();
        if (n == 0) {
            return new ClusteringResult(new float[0][], new int[0], new float[0], new int[0], new float[0], 0);
        }

        // Seed initial centroids from a single-pass reservoir sample of the input, so large
        // merges skip full k-means++ initialization. HierarchicalKMeans falls back to k-means++
        // when the sample is too small to cover the requested centroid count. Tiny segments
        // (n <= target cluster size) become a single cluster inside HierarchicalKMeans and never
        // consult these seeds, so skip the sample pass entirely for them.
        float[][] initialCentroids = n <= ClusterBuilderConstants.TARGET_CLUSTER_SIZE ? null : reservoirSampleCentroids(vectors, n);

        HierarchicalKMeans.Config hConfig = HierarchicalKMeans.Config.builder()
            .targetSize(ClusterBuilderConstants.TARGET_CLUSTER_SIZE)
            .metric(metric)
            .seed(ClusterBuilderConstants.SEED)
            .executor(executor)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, hConfig, initialCentroids);
        float[][] centroids = result.centroids();
        int[] assignments = result.assignments();
        int numCentroids = result.numCentroids();

        // Per-member distance to the assigned (primary) centroid. The .clap writer stores postings
        // sorted by this distance and records each cluster's max distance, so surface it here rather
        // than making the writer recompute it. These recorded distances are squared L2 regardless of
        // the space type: assignment/SOAR selection honor the configured metric, but the posting-list
        // pruning (CLIP early termination, inter-cluster max-distance pruning) is defined in terms of
        // L2 distance for every space type, so the values stored here must be L2.
        float[] distances = computePrimaryDistances(vectors, centroids, assignments, executor);

        int[] soarAssignments;
        float[] soarDistances;
        int[][] extraSoarAssignments = null;
        float[][] extraSoarDistances = null;
        if (ClusterBuilderConstants.SOAR_LAMBDA > 0 && numCentroids > 1) {
            SoarResult soar = computeSOAR(
                vectors,
                centroids,
                assignments,
                numCentroids,
                metric,
                executor,
                new SoarParams(maxSoarAssignments, spillMargin)
            );
            soarAssignments = soar.assignments();
            soarDistances = soar.distances();
            extraSoarAssignments = soar.extraAssignments();
            extraSoarDistances = soar.extraDistances();
        } else {
            soarAssignments = new int[n];
            soarDistances = new float[n];
            Arrays.fill(soarAssignments, -1);
            Arrays.fill(soarDistances, Float.NaN);
        }

        return new ClusteringResult(
            centroids,
            assignments,
            distances,
            soarAssignments,
            soarDistances,
            extraSoarAssignments,
            extraSoarDistances,
            numCentroids
        );
    }

    /**
     * Compute each vector's squared-L2 distance to its assigned centroid, in one parallel pass.
     *
     * <p>Always squared L2, independent of the space type used for assignment: these values feed
     * posting-list pruning (sorted postings + per-cluster max distance), which is defined in terms
     * of L2 distance for every space type.
     */
    private static float[] computePrimaryDistances(FloatVectorValues vectors, float[][] centroids, int[] assignments, TaskExecutor executor)
        throws IOException {
        int n = vectors.size();
        float[] distances = new float[n];
        ParallelVectorProcessor.execute(vectors, n, executor, (start, end, view) -> {
            for (int i = start; i < end; i++) {
                distances[i] = VectorMath.squareDistance(view.vectorValue(i), centroids[assignments[i]]);
            }
        });
        return distances;
    }

    /**
     * Draw a single-pass reservoir sample (Algorithm R) of up to {@code RESERVOIR_SIZE} vectors
     * to seed initial centroids. {@link HierarchicalKMeans} uses these directly when the sample
     * covers the centroid count it needs, and otherwise falls back to k-means++ initialization.
     */
    private static float[][] reservoirSampleCentroids(FloatVectorValues vectors, int n) throws IOException {
        int reservoirSize = Math.min(ClusterBuilderConstants.RESERVOIR_SIZE, n);
        float[][] reservoir = new float[reservoirSize][];
        Random rng = new Random(ClusterBuilderConstants.SEED);

        for (int i = 0; i < n; i++) {
            float[] vec = vectors.vectorValue(i);
            if (i < reservoirSize) {
                reservoir[i] = vec.clone();
            } else {
                int j = rng.nextInt(i + 1);
                if (j < reservoirSize) {
                    reservoir[j] = vec.clone();
                }
            }
        }
        return reservoir;
    }

    // ========== SOAR ==========

    /** Bundled SOAR output: first secondary (flat) plus optional extra spills (ragged, null at cap 1). */
    private record SoarResult(int[] assignments, float[] distances, int[][] extraAssignments, float[][] extraDistances) {
    }

    /** Shared empty rows for boundary vectors that produced no spill beyond the first secondary. */
    private static final int[] EMPTY_INT = new int[0];
    private static final float[] EMPTY_FLOAT = new float[0];

    /** SOAR tuning knobs threaded into {@link #computeSOAR}: spill cap and beyond-first margin. */
    private record SoarParams(int maxAssignments, float spillMargin) {
    }

    private static SoarResult computeSOAR(
        FloatVectorValues vectors,
        float[][] centroids,
        int[] assignments,
        int numCentroids,
        VectorSimilarityFunction metric,
        TaskExecutor executor,
        SoarParams params
    ) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        int[] soarAssignments = new int[n];
        float[] soarDistances = new float[n];
        Arrays.fill(soarAssignments, -1);
        Arrays.fill(soarDistances, Float.NaN);

        // Candidate selection (which neighbors SOAR considers) honors the configured metric.
        VectorMath.DistanceFunction distanceFn = VectorMath.distanceFunction(metric);
        // Precompute nearest centroids per centroid (SOAR neighbor candidates).
        int candidateLimit = Math.min(numCentroids - 1, ClusterBuilderConstants.SOAR_CANDIDATE_LIMIT);
        int[][] nearestCentroids = new int[numCentroids][];
        for (int c = 0; c < numCentroids; c++) {
            nearestCentroids[c] = VectorMath.nearestCentroids(centroids, c, candidateLimit, distanceFn);
        }

        float soarLambda = ClusterBuilderConstants.SOAR_LAMBDA;
        int maxAssign = Math.max(1, params.maxAssignments());
        float spillMargin = params.spillMargin();
        // Extra-spill rows only exist when the cap is > 1; at the default cap of 1 these stay null
        // and the result is the classic single-secondary shape.
        int[][] extraAssignments = maxAssign > 1 ? new int[n][] : null;
        float[][] extraDistances = maxAssign > 1 ? new float[n][] : null;

        // Each worker gets its own vector view; ParallelVectorProcessor runs inline when executor is null.
        ParallelVectorProcessor.execute(vectors, n, executor, (start, end, view) -> {
            float[] residual = new float[dim];
            // Per-worker scratch for ranking candidates by SOAR score (only the top `maxAssign` kept).
            int[] rankCent = new int[maxAssign];
            float[] rankScore = new float[maxAssign];
            for (int i = start; i < end; i++) {
                float[] vec = view.vectorValue(i);
                int primaryCent = assignments[i];
                float[] primaryCentroid = centroids[primaryCent];
                int[] neighbors = nearestCentroids[primaryCent];
                int numCandidates = neighbors.length;
                if (numCandidates == 0) {
                    continue;
                }

                // Residual to the primary centroid, computed once and reused below.
                float residualNormSq = 0f;
                for (int d = 0; d < dim; d++) {
                    float r = vec[d] - primaryCentroid[d];
                    residual[d] = r;
                    residualNormSq += r * r;
                }
                if (residualNormSq < 1e-20f) {
                    continue;
                }
                float invNorm = soarLambda / residualNormSq;

                // Rank candidates by SOAR score into the top-`maxAssign` buffer (ascending score;
                // slot 0 is the best). Simple insertion keeps the buffer sorted — maxAssign is tiny.
                int ranked = 0;
                for (int neighbor : neighbors) {
                    float[] candidate = centroids[neighbor];
                    float dsq = 0f;
                    float proj = 0f;
                    for (int d = 0; d < dim; d++) {
                        float diff = vec[d] - candidate[d];
                        dsq += diff * diff;
                        proj += residual[d] * diff;
                    }
                    // SOAR ranks candidates by a residual-orthogonality-penalized score; the posting,
                    // however, is sorted by the true member distance, so record that separately below.
                    float score = dsq + invNorm * proj * proj;
                    if (ranked < maxAssign) {
                        int p = ranked++;
                        while (p > 0 && rankScore[p - 1] > score) {
                            rankScore[p] = rankScore[p - 1];
                            rankCent[p] = rankCent[p - 1];
                            p--;
                        }
                        rankScore[p] = score;
                        rankCent[p] = neighbor;
                    } else if (score < rankScore[maxAssign - 1]) {
                        int p = maxAssign - 1;
                        while (p > 0 && rankScore[p - 1] > score) {
                            rankScore[p] = rankScore[p - 1];
                            rankCent[p] = rankCent[p - 1];
                            p--;
                        }
                        rankScore[p] = score;
                        rankCent[p] = neighbor;
                    }
                }

                if (ranked == 0) {
                    continue;
                }

                // First (best) secondary keeps the classic flat contract. Recorded distance is
                // squared L2 regardless of space type (posting-list pruning basis) — even though the
                // secondary above is selected using the metric.
                int bestCent = rankCent[0];
                soarAssignments[i] = bestCent;
                soarDistances[i] = VectorMath.squareDistance(vec, centroids[bestCent]);

                // Spills beyond the first: kept only while within the margin of the best score,
                // which collapses interior vectors (large score gap) to primary-only and bounds
                // boundary vectors to `maxAssign` total. Only populated when the cap is > 1.
                if (maxAssign > 1) {
                    float bestScore = rankScore[0];
                    float threshold = bestScore * (1f + spillMargin);
                    int extra = 0;
                    while (extra + 1 < ranked && rankScore[extra + 1] <= threshold) {
                        extra++;
                    }
                    if (extra > 0) {
                        int[] exCent = new int[extra];
                        float[] exDist = new float[extra];
                        for (int e = 0; e < extra; e++) {
                            int c = rankCent[e + 1];
                            exCent[e] = c;
                            exDist[e] = VectorMath.squareDistance(vec, centroids[c]);
                        }
                        extraAssignments[i] = exCent;
                        extraDistances[i] = exDist;
                    } else {
                        extraAssignments[i] = EMPTY_INT;
                        extraDistances[i] = EMPTY_FLOAT;
                    }
                }
            }
        });

        return new SoarResult(soarAssignments, soarDistances, extraAssignments, extraDistances);
    }
}
