/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.IntroSorter;

/**
 * Sorts an index array in place so its members end up nearest-first — closest to the centroid at the head,
 * so posting-list pruning drops the farthest. The sort key is a squared-L2 distance for every metric
 * ({@code ClusterBuilder} records L2 regardless of similarity), so nearest-first is the natural (ascending)
 * order for L2 but the reverse (descending) for inner product and cosine, whose closest members carry the
 * largest squared-L2 distance — see the combined-assignments design doc. Sorting the index array rather than
 * the parallel arrays lets those be gathered afterwards without boxing.
 */
final class NearestFirstDistanceSorter extends IntroSorter {

    /**
     * Orders two {@code float} sort keys. Used instead of {@link java.util.Comparator}{@code <Float>} because
     * that compares {@code Float} objects, autoboxing every key on each comparison — O(n log n) allocations
     * per sort — which defeats the point of sorting an index array over the primitive {@code float[]}. This
     * takes primitive {@code float}s, so comparison stays boxing-free.
     */
    @FunctionalInterface
    private interface FloatComparator {
        int compare(float a, float b);
    }

    private final int[] order;
    private final float[] distances;
    private final FloatComparator comparator;
    private float pivot;

    private NearestFirstDistanceSorter(final int[] order, final float[] distances, final FloatComparator comparator) {
        this.order = order;
        this.distances = distances;
        this.comparator = comparator;
    }

    /**
     * Sorts {@code order} in place so its entries index into {@code distances} nearest-first for {@code
     * metric} — the closest member to the centroid at the head, so posting-list pruning drops the farthest.
     * L2 orders by ascending distance; inner product and cosine reverse it.
     */
    static void sort(final int[] order, final float[] distances, final VectorSimilarityFunction metric) {
        final FloatComparator comparator = metric == VectorSimilarityFunction.EUCLIDEAN
            ? (a, b) -> Float.compare(a, b)   // nearest-first: ascending distance
            : (a, b) -> Float.compare(b, a);  // nearest-first: descending distance
        new NearestFirstDistanceSorter(order, distances, comparator).sort(0, order.length);
    }

    @Override
    protected void setPivot(int i) {
        pivot = distances[order[i]];
    }

    @Override
    protected int comparePivot(int j) {
        return comparator.compare(pivot, distances[order[j]]);
    }

    @Override
    protected int compare(int i, int j) {
        return comparator.compare(distances[order[i]], distances[order[j]]);
    }

    @Override
    protected void swap(int i, int j) {
        final int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}
