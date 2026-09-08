/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.orchestration;

/**
 * The bounds a plan is chosen within.
 *
 * <p>There is deliberately no {@code nprobe} here. A caller does not know how many clusters a query needs — that
 * depends on where the query falls relative to the centroids, which is only knowable once they have been ranked. So
 * the caller supplies a range and the planner decides within it.
 *
 * <p>{@code maxProbes} is applied during the ranking sweep, as the capacity of the selection heap, so the clusters
 * beyond it are never ordered. {@code minProbes} is applied afterwards, as a floor on whatever the pruners left.
 *
 * @param minProbes fewest clusters a plan may name, unless the field has fewer non-empty clusters than this
 * @param maxProbes most clusters a plan may name; also the ranking sweep's heap capacity
 */
public record PlanParams(int minProbes, int maxProbes) {

    /** Multiple of √centroids a plan may probe at most. */
    public static final int NPROBE_MULTIPLIER = 2;

    /** Probes a plan will not go below, however few clusters the field has. */
    public static final int MIN_NPROBE = 10;

    public PlanParams {
        if (minProbes < 1) {
            throw new IllegalArgumentException("minProbes must be at least 1, got: " + minProbes);
        }
        if (maxProbes < minProbes) {
            throw new IllegalArgumentException("maxProbes=" + maxProbes + " is below minProbes=" + minProbes);
        }
    }

    /**
     * Bounds scaled to √centroids, which is the scale the cluster count itself is on: an IVF index holds roughly √N
     * clusters, so √centroids is roughly the fourth root of the vector count and the probe count tracks the field
     * rather than a fixed guess.
     *
     * @param numCentroids the field's cluster count
     */
    public static PlanParams of(int numCentroids) {
        int root = (int) Math.sqrt(numCentroids);
        int maxProbes = Math.max(1, Math.min(NPROBE_MULTIPLIER * root, numCentroids));
        // The floor cannot exceed the ceiling: on a small field there are not ten clusters to insist on.
        int minProbes = Math.min(Math.max(MIN_NPROBE, root), maxProbes);
        return new PlanParams(minProbes, maxProbes);
    }
}
