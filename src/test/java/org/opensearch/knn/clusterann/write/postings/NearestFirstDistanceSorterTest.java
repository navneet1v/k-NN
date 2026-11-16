/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies {@link NearestFirstDistanceSorter} orders an index array by its distance in the metric's pruning
 * direction — ascending for L2, descending for inner product, cosine, and dot product — and leaves it a
 * permutation of the original indices. Uses a large, randomly keyed array so the sort exercises the quicksort
 * path (pivot selection), not only the small-range insertion sort.
 */
class NearestFirstDistanceSorterTest {

    @Test
    void sortsAscendingByDistanceForL2() {
        assertSortedByDistance(VectorSimilarityFunction.EUCLIDEAN, true);
    }

    @Test
    void sortsDescendingByDistanceForInnerProduct() {
        assertSortedByDistance(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, false);
    }

    @Test
    void sortsDescendingByDistanceForDotProduct() {
        assertSortedByDistance(VectorSimilarityFunction.DOT_PRODUCT, false);
    }

    @Test
    void sortsDescendingByDistanceForCosine() {
        assertSortedByDistance(VectorSimilarityFunction.COSINE, false);
    }

    @Test
    void handlesEmptyAndSingleElement() {
        // Empty and single-element arrays must not throw and stay valid permutations.
        NearestFirstDistanceSorter.sort(new int[0], new float[0], VectorSimilarityFunction.EUCLIDEAN);

        final int[] single = { 0 };
        NearestFirstDistanceSorter.sort(single, new float[] { 1.0f }, VectorSimilarityFunction.COSINE);
        assertTrue(single[0] == 0, "single element unchanged");
    }

    private static void assertSortedByDistance(final VectorSimilarityFunction metric, final boolean ascending) {
        final float[] distances = randomDistances(200, metric.ordinal() + 1);
        final int[] order = identity(distances.length);

        NearestFirstDistanceSorter.sort(order, distances, metric);

        for (int i = 1; i < distances.length; i++) {
            final float prev = distances[order[i - 1]];
            final float curr = distances[order[i]];
            if (ascending) {
                assertTrue(prev <= curr, "ascending by distance at " + i + " for " + metric);
            } else {
                assertTrue(prev >= curr, "descending by distance at " + i + " for " + metric);
            }
        }
        assertPermutation(order);
    }

    private static float[] randomDistances(final int n, final long seed) {
        final Random rnd = new Random(seed);
        final float[] distances = new float[n];
        for (int i = 0; i < n; i++) {
            distances[i] = rnd.nextFloat();
        }
        return distances;
    }

    private static int[] identity(final int n) {
        final int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        return order;
    }

    private static void assertPermutation(final int[] order) {
        final boolean[] seen = new boolean[order.length];
        for (final int o : order) {
            assertFalse(seen[o], "index " + o + " appears once");
            seen[o] = true;
        }
    }
}
