/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

/**
 * Immutable result of IVF clustering: centroids, primary assignments, and SOAR secondary assignments.
 * Clean data transfer between clustering (Layer 2) and writing (Layer 3).
 */
public record ClusteringResult(float[][] centroids, int[] assignments, int[] soarAssignments, int numCentroids) {

    /** Build primary posting lists: postings[centroidIdx] = sorted vector ordinals. */
    public int[][] primaryPostingLists() {
        return buildPostingLists(assignments, numCentroids);
    }

    /** Build SOAR posting lists: postings[centroidIdx] = sorted vector ordinals. */
    public int[][] soarPostingLists() {
        return buildPostingLists(soarAssignments, numCentroids);
    }

    private static int[][] buildPostingLists(int[] assignments, int numCentroids) {
        int[] counts = new int[numCentroids];
        for (int a : assignments) {
            if (a >= 0 && a < numCentroids) counts[a]++;
        }
        int[][] postings = new int[numCentroids][];
        for (int c = 0; c < numCentroids; c++) {
            postings[c] = new int[counts[c]];
        }
        int[] pos = new int[numCentroids];
        for (int i = 0; i < assignments.length; i++) {
            int a = assignments[i];
            if (a >= 0 && a < numCentroids) {
                postings[a][pos[a]++] = i;
            }
        }
        return postings;
    }
}
