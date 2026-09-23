/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.math;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

/**
 * Vector distance utilities for ClusterANN algorithms.
 *
 * <p>Single-pair methods delegate to Lucene's {@link VectorUtil} which uses Panama SIMD
 * on capable JVMs via Lucene's {@code VectorizationProvider}.
 */
public final class VectorMath {

    private VectorMath() {}

    /**
     * A resolved clustering distance: <b>lower = closer</b> (argmin semantics). Obtained once from
     * a {@link VectorSimilarityFunction} via {@link #distanceFunction} and then invoked directly in
     * hot loops, so the metric switch is not re-evaluated per vector.
     */
    @FunctionalInterface
    public interface DistanceFunction {
        /** Distance between two vectors; lower = closer. SIMD-accelerated. */
        float distance(float[] a, float[] b);
    }

    /**
     * Resolve a Lucene {@link VectorSimilarityFunction} to a clustering {@link DistanceFunction}
     * (lower = closer) — the opposite orientation of Lucene's {@link VectorSimilarityFunction#compare},
     * which returns a normalized higher-is-better similarity. Clustering needs the raw, un-normalized
     * distance to pick nearest centroids, so we map each function to its distance form here rather
     * than calling {@code compare}. Resolve once and reuse.
     *
     * <ul>
     *   <li>{@code EUCLIDEAN} &rarr; squared L2 distance</li>
     *   <li>{@code MAXIMUM_INNER_PRODUCT} &rarr; negated inner product</li>
     *   <li>{@code COSINE} &rarr; {@code 1 - cosine} similarity</li>
     *   <li>{@code DOT_PRODUCT} &rarr; unsupported</li>
     * </ul>
     */
    public static DistanceFunction distanceFunction(VectorSimilarityFunction metric) {
        return switch (metric) {
            case EUCLIDEAN -> VectorMath::squareDistance;
            case MAXIMUM_INNER_PRODUCT -> (a, b) -> -dotProduct(a, b);
            case DOT_PRODUCT -> throw new IllegalArgumentException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
            case COSINE -> (a, b) -> 1f - cosine(a, b);
        };
    }

    /** L2 squared distance. SIMD-accelerated via Lucene. */
    public static float squareDistance(float[] a, float[] b) {
        return VectorUtil.squareDistance(a, b);
    }

    /** Dot product. SIMD-accelerated via Lucene. */
    public static float dotProduct(float[] a, float[] b) {
        return VectorUtil.dotProduct(a, b);
    }

    /** Cosine similarity. SIMD-accelerated via Lucene. */
    public static float cosine(float[] a, float[] b) {
        return VectorUtil.cosine(a, b);
    }

    /**
     * Find the nearest centroid to {@code vector} over a flattened centroid array using L2²,
     * writing each centroid's squared distance into {@code distances}.
     *
     * @param vector       query vector
     * @param flatCentroids centroids laid out as {@code flatCentroids[c * dimension + d]}
     * @param k            number of centroids
     * @param dimension    vector dimension
     * @param distances    scratch buffer of length {@code k} to receive per-centroid distances
     * @return index of the nearest centroid
     */
    public static int findNearestCentroidBulk(float[] vector, float[] flatCentroids, int k, int dimension, float[] distances) {
        int bestIdx = 0;
        float bestDist = Float.MAX_VALUE;
        for (int c = 0; c < k; c++) {
            int off = c * dimension;
            float d = 0f;
            for (int j = 0; j < dimension; j++) {
                float diff = vector[j] - flatCentroids[off + j];
                d += diff * diff;
            }
            distances[c] = d;
            if (d < bestDist) {
                bestDist = d;
                bestIdx = c;
            }
        }
        return bestIdx;
    }

    /** Flatten centroids into a single {@code float[]} for cache-friendly bulk scans. */
    public static float[] flattenCentroids(float[][] centroids) {
        int k = centroids.length;
        int dim = centroids[0].length;
        float[] flat = new float[k * dim];
        for (int c = 0; c < k; c++) {
            System.arraycopy(centroids[c], 0, flat, c * dim, dim);
        }
        return flat;
    }

    /**
     * Return the indices of the {@code limit} centroids nearest to {@code centroids[source]}
     * (excluding {@code source} itself), ordered nearest-first. Uses a partial selection sort,
     * which is cheaper than a full sort when {@code limit} is small relative to {@code k}.
     *
     * @param centroids all centroids
     * @param source    index of the centroid whose neighbors are wanted
     * @param limit     number of neighbors to return (clamped to {@code k - 1})
     * @param distanceFn resolved distance function (lower = closer)
     * @return array of length {@code min(limit, k - 1)} of nearest neighbor indices
     */
    public static int[] nearestCentroids(float[][] centroids, int source, int limit, DistanceFunction distanceFn) {
        int k = centroids.length;
        int resultSize = Math.min(limit, k - 1);
        float[] dists = new float[k];
        int[] idx = new int[k];
        for (int j = 0; j < k; j++) {
            dists[j] = (j == source) ? Float.MAX_VALUE : distanceFn.distance(centroids[source], centroids[j]);
            idx[j] = j;
        }
        for (int i = 0; i < resultSize; i++) {
            int minPos = i;
            for (int j = i + 1; j < k; j++) {
                if (dists[idx[j]] < dists[idx[minPos]]) {
                    minPos = j;
                }
            }
            int tmp = idx[i];
            idx[i] = idx[minPos];
            idx[minPos] = tmp;
        }
        int[] result = new int[resultSize];
        System.arraycopy(idx, 0, result, 0, resultSize);
        return result;
    }
}
