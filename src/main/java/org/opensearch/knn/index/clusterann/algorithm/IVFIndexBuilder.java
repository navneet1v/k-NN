/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

import org.opensearch.knn.index.clusterann.*;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.apache.lucene.index.FloatVectorValues;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.io.UncheckedIOException;

/**
 * Builds an IVF index: clustering + SOAR secondary assignments.
 * Takes Lucene's {@link FloatVectorValues} — works with any vector source.
 * Stateless — all configuration passed via parameters.
 */
public final class IVFIndexBuilder {

    private static final int SOAR_CANDIDATE_LIMIT = 10;

    private IVFIndexBuilder() {}

    /**
     * Build IVF clustering with SOAR.
     *
     * @param vectors           input vectors (Lucene type)
     * @param targetClusterSize target vectors per cluster (e.g., 512)
     * @param metric            distance metric
     * @param soarLambda        SOAR parameter (0 to disable)
     * @param initialCentroids  pre-selected centroids for merge (null for flush)
     * @param seed              random seed
     * @param parallel          enable parallel processing
     * @return clustering result with centroids, assignments, and SOAR
     */
    public static ClusteringResult build(
        FloatVectorValues vectors,
        int targetClusterSize,
        DistanceMetric metric,
        float soarLambda,
        float[][] initialCentroids,
        long seed,
        boolean parallel
    ) throws IOException {
        int n = vectors.size();
        if (n == 0) {
            return new ClusteringResult(new float[0][], new int[0], new int[0], 0);
        }

        // Wrap in ClusterANNVectorValues if not already (for k-means compatibility)
        ClusterANNVectorValues clusterVectors;
        if (vectors instanceof ClusterANNVectorValues cv) {
            clusterVectors = cv;
        } else {
            // Shouldn't happen in practice — our factory methods return ClusterANNVectorValues
            throw new IllegalArgumentException("Expected ClusterANNVectorValues, got " + vectors.getClass());
        }

        // Cluster
        HierarchicalKMeans.Config hConfig = HierarchicalKMeans.Config.builder()
            .targetSize(targetClusterSize)
            .metric(metric)
            .seed(seed)
            .parallel(parallel)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(clusterVectors, hConfig, initialCentroids);
        float[][] centroids = result.centroids();
        int[] assignments = result.assignments();
        int numCentroids = result.numCentroids();

        // SOAR secondary assignments
        int[] soarAssignments;
        if (soarLambda > 0 && numCentroids > 1) {
            soarAssignments = computeSOAR(clusterVectors, centroids, assignments, numCentroids, metric, soarLambda);
        } else {
            soarAssignments = new int[n];
            Arrays.fill(soarAssignments, -1);
        }

        return new ClusteringResult(centroids, assignments, soarAssignments, numCentroids);
    }

    // ========== SOAR ==========

    private static int[] computeSOAR(
        ClusterANNVectorValues vectors,
        float[][] centroids,
        int[] assignments,
        int numCentroids,
        DistanceMetric metric,
        float soarLambda
    ) throws IOException {
        int n = vectors.size();
        int dim = vectors.dimension();
        int[] soarAssignments = new int[n];
        Arrays.fill(soarAssignments, -1);

        if (numCentroids <= 1) return soarAssignments;

        // Precompute nearest centroids per centroid
        int candidateLimit = Math.min(numCentroids - 1, SOAR_CANDIDATE_LIMIT);
        int[][] nearestCentroids = new int[numCentroids][candidateLimit];
        for (int c = 0; c < numCentroids; c++) {
            float[] dists = new float[numCentroids];
            int[] idx = new int[numCentroids];
            for (int j = 0; j < numCentroids; j++) {
                dists[j] = (j == c) ? Float.MAX_VALUE : metric.distance(centroids[c], centroids[j]);
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

        // Thread-local buffers to avoid per-vector allocation
        int flatSize = candidateLimit * dim;
        ThreadLocal<float[]> tlFlatCandidates = ThreadLocal.withInitial(() -> new float[flatSize]);
        ThreadLocal<float[]> tlDists = ThreadLocal.withInitial(() -> new float[candidateLimit]);

        // Parallel SOAR computation
        IntStream.range(0, n).parallel().forEach(i -> {
            float[] vec;
            try {
                vec = vectors.vectorValue(i);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            int primaryCent = assignments[i];
            float[] primaryCentroid = centroids[primaryCent];
            int[] neighbors = nearestCentroids[primaryCent];
            int numCandidates = neighbors.length;
            if (numCandidates == 0) return;

            float[] flatCandidates = tlFlatCandidates.get();
            for (int j = 0; j < numCandidates; j++) {
                System.arraycopy(centroids[neighbors[j]], 0, flatCandidates, j * dim, dim);
            }

            float[] dists = tlDists.get();
            try {
                SimdVectorComputeService.bulkSOARDistance(vec, primaryCentroid, flatCandidates, dists, dim, numCandidates, soarLambda);
            } catch (Throwable t) {
                // Fallback: scalar SOAR
                float residualNormSq = 0f;
                for (int d = 0; d < dim; d++) {
                    float r = vec[d] - primaryCentroid[d];
                    residualNormSq += r * r;
                }
                if (residualNormSq < 1e-20f) return;
                float invNorm = soarLambda / residualNormSq;
                for (int j = 0; j < numCandidates; j++) {
                    float dsq = 0f, proj = 0f;
                    for (int d = 0; d < dim; d++) {
                        float diff = vec[d] - centroids[neighbors[j]][d];
                        dsq += diff * diff;
                        proj += (vec[d] - primaryCentroid[d]) * diff;
                    }
                    dists[j] = dsq + invNorm * proj * proj;
                }
            }

            float bestDist = Float.MAX_VALUE;
            int bestCent = -1;
            for (int j = 0; j < numCandidates; j++) {
                if (dists[j] < bestDist) {
                    bestDist = dists[j];
                    bestCent = neighbors[j];
                }
            }
            soarAssignments[i] = bestCent;
        });

        return soarAssignments;
    }
}
