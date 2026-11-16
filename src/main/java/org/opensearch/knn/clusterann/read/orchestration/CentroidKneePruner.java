/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

/**
 * Inter-cluster pruning: cuts a ranked centroid plan at the knee in its distance curve.
 *
 * <p>Past the knee the centroids are not the query's neighbours — they are the rest of the index — so each costs a
 * posting read while holding nothing competitive. Where it falls is a property of the query, not the field, so it is
 * read off the curve per query. Only a prefix is cut, never a hole, so closest-first order survives.
 */
final class CentroidKneePruner {

    /** Factor for unfiltered plans and filters passing at least {@link #CURVE_START} of the vectors. */
    private static final float BASE_KNEE_FACTOR = 3.0f;

    /** Highest factor, for the most selective filters. */
    private static final float MAX_KNEE_FACTOR = 8.0f;

    /** Selectivity below which the factor starts to rise. */
    private static final double CURVE_START = 0.25;

    /** ln(4/3) / ln(2.5): the factor reaches 4 at s = 0.10. */
    private static final double CURVE_EXPONENT = 0.314;

    private CentroidKneePruner() {}

    /** Knee factor for a selectivity: 3 at s >= 0.25, then min(8, 3 * (0.25 / s)^0.314). */
    static float kneeFactor(final double selectivity) {
        if (selectivity >= CURVE_START) {
            return BASE_KNEE_FACTOR;
        }
        // Non-positive or NaN: the most selective case.
        if (!(selectivity > 0)) {
            return MAX_KNEE_FACTOR;
        }
        return (float) Math.min(MAX_KNEE_FACTOR, BASE_KNEE_FACTOR * Math.pow(CURVE_START / selectivity, CURVE_EXPONENT));
    }

    /**
     * Clusters to keep: the floor, then up to the first tail gap over {@link #kneeFactor(double)} tail mean gaps. A
     * tail of that many gaps or fewer cannot cut.
     *
     * @param distances ranking keys, ascending. A true squared distance only under EUCLIDEAN; elsewhere a negated
     *     similarity, still monotone in cluster quality, which is all the cut relies on
     * @param minProbes the floor the cut may not go below; at least 1
     * @param selectivity fraction of the field's vectors the filter accepts; 1 when unfiltered
     */
    static int prune(final float[] distances, final int minProbes, final double selectivity) {
        final int ranked = distances.length;
        if (ranked <= minProbes) {
            return ranked;
        }

        final int first = minProbes - 1;
        final float rise = distances[ranked - 1] - distances[first];
        if (rise <= 0) {
            // A flat tail: no gap stands out, and the mean would be zero.
            return ranked;
        }

        // Tail mean gap times the factor; dividing first keeps a huge rise finite.
        final float threshold = rise / (ranked - minProbes) * kneeFactor(selectivity);
        for (int i = first; i < ranked - 1; i++) {
            if (distances[i + 1] - distances[i] > threshold) {
                return i + 1;
            }
        }
        return ranked;
    }
}
