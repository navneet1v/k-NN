/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ClusteringResultTests {

    @Test
    void record_exposesCentroidsAssignmentsAndDistances() {
        float[][] centroids = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };
        int[] assignments = { 0, 1, 0 };
        float[] distances = { 0.1f, 0.2f, 0.3f };
        int[] soar = { 1, -1, 1 };
        float[] soarDistances = { 0.5f, Float.NaN, 0.7f };

        ClusteringResult result = new ClusteringResult(centroids, assignments, distances, soar, soarDistances, 2);

        assertEquals(2, result.numCentroids());
        assertArrayEquals(centroids, result.centroids());
        assertArrayEquals(assignments, result.assignments());
        assertArrayEquals(distances, result.distances(), 1e-6f);
        assertArrayEquals(soar, result.soarAssignments());
        assertArrayEquals(soarDistances, result.soarDistances(), 1e-6f);
    }

    @Test
    void record_carriesExtraSpillRowsWhenPresent() {
        float[][] centroids = { { 0.0f, 0.0f }, { 1.0f, 1.0f }, { 2.0f, 2.0f } };
        int[] assignments = { 0, 1 };
        float[] distances = { 0.1f, 0.2f };
        int[] soar = { 1, 2 };
        float[] soarDistances = { 0.5f, 0.6f };
        int[][] extraSoar = { { 2 }, {} };
        float[][] extraSoarDistances = { { 0.9f }, {} };

        ClusteringResult result = new ClusteringResult(
            centroids,
            assignments,
            distances,
            soar,
            soarDistances,
            extraSoar,
            extraSoarDistances,
            3
        );

        assertArrayEquals(extraSoar, result.extraSoarAssignments());
        assertArrayEquals(extraSoarDistances[0], result.extraSoarDistances()[0], 1e-6f);
        // The 6-arg convenience constructor leaves the extra-spill fields null.
        ClusteringResult classic = new ClusteringResult(centroids, assignments, distances, soar, soarDistances, 3);
        assertNull(classic.extraSoarAssignments());
        assertNull(classic.extraSoarDistances());
    }

    @Test
    void record_supportsEmptyResult() {
        ClusteringResult result = new ClusteringResult(new float[0][], new int[0], new float[0], new int[0], new float[0], 0);
        assertEquals(0, result.numCentroids());
        assertEquals(0, result.assignments().length);
        assertEquals(0, result.distances().length);
        assertEquals(0, result.soarDistances().length);
    }
}
