/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann;

/**
 * Immutable result of IVF clustering: centroids, primary assignments with their member distances,
 * and SOAR secondary assignments with their member distances. Clean data transfer between clustering
 * (Layer 2) and writing (Layer 3).
 *
 * <p>{@code distances[i]} is vector {@code i}'s distance to its assigned centroid
 * {@code assignments[i]} (lower = closer). {@code soarDistances[i]} is the distance to the SOAR
 * secondary centroid {@code soarAssignments[i]} when one was assigned, else {@link Float#NaN}
 * (paired with {@code soarAssignments[i] == -1}). The writer consumes these directly to sort each
 * posting by member distance without recomputing them.
 *
 * <p>{@code soarAssignments}/{@code soarDistances} carry the <em>first</em> (best) SOAR secondary,
 * preserving the classic single-secondary contract. When the spill cap
 * ({@code ClusterBuilderConstants.MAX_SOAR_ASSIGNMENTS}) is {@code > 1}, additional secondaries for
 * a boundary vector are carried in {@code extraSoarAssignments[i]} / {@code extraSoarDistances[i]}
 * (parallel ragged rows, ordered by ascending SOAR score; empty row = no further spill). At the
 * default cap of {@code 1} these two fields are {@code null} and the result is identical to
 * classic SOAR.
 *
 * @param centroids            cluster centroids
 * @param assignments          primary centroid index per vector
 * @param distances            distance from each vector to its primary centroid
 * @param soarAssignments      first SOAR secondary centroid index per vector, or {@code -1} if none
 * @param soarDistances        distance from each vector to its first SOAR secondary, or {@code NaN}
 * @param extraSoarAssignments additional SOAR secondary indices per vector beyond the first, or
 *                             {@code null} when the spill cap is {@code 1}
 * @param extraSoarDistances   distances for {@code extraSoarAssignments}, or {@code null} when the
 *                             spill cap is {@code 1}
 * @param numCentroids         number of centroids
 */
public record ClusteringResult(float[][] centroids, int[] assignments, float[] distances, int[] soarAssignments, float[] soarDistances,
    int[][] extraSoarAssignments, float[][] extraSoarDistances, int numCentroids) {

    /**
     * Convenience constructor for the classic single-secondary result (spill cap 1): the extra-spill
     * fields are {@code null}. Existing callers and tests use this arity unchanged.
     */
    public ClusteringResult(
        float[][] centroids,
        int[] assignments,
        float[] distances,
        int[] soarAssignments,
        float[] soarDistances,
        int numCentroids
    ) {
        this(centroids, assignments, distances, soarAssignments, soarDistances, null, null, numCentroids);
    }
}
