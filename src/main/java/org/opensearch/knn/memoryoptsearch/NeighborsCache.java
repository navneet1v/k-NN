/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class NeighborsCache {

    private final Map<Integer, int[]> neighborsCache0;
    private final Map<Integer, int[]> neighborsCache1;
    private final Map<Integer, int[]> neighborsCache2;

    // This is not a good implementation for prod. But this should help in making the POC easy to test.
    public NeighborsCache(int totalNumberOfVector) {
        // this means that we are storing neighbors of 50K nodes here. In worst case a node can have 32 neighbors
        // (only for level 0)
        // so 50000 * 32 * 4 = ~ 6.2mb. We are using a decay factor of 1/M. M = 16
        // We only need 3 levels since Graphs created with Faiss Lib has maximum of 3 levels.
        neighborsCache0 = new ConcurrentHashMap<>(totalNumberOfVector);
        /**
         * The decay factor of 16 is used since for multi-layer graph structures where higher
         * levels become progressively sparser.
         */
        neighborsCache1 = new ConcurrentHashMap<>((int) Math.ceil((double) totalNumberOfVector / 16));
        neighborsCache2 = new ConcurrentHashMap<>((int) Math.ceil((double) totalNumberOfVector / (16 * 16)));
    }

    public void put(int nodeId, int level, int[] neighborsList, int numberOfNeighbors) {
        int[] arr = new int[numberOfNeighbors];
        System.arraycopy(neighborsList, 0, arr, 0, numberOfNeighbors);
        switch (level) {
            case 0:
                neighborsCache0.put(nodeId, arr);
                break;
            case 1:
                neighborsCache1.put(nodeId, arr);
                break;
            case 2:
                neighborsCache2.put(nodeId, arr);
                break;
        }
    }

    public int[] load(int nodeId, int level) {
        int[] arr = null;
        switch (level) {
            case 0:
                if (neighborsCache0.containsKey(nodeId)) {
                    arr = neighborsCache0.get(nodeId);
                }
                break;
            case 1:
                if (neighborsCache1.containsKey(nodeId)) {
                    arr = neighborsCache1.get(nodeId);
                }
                break;
            case 2:
                if (neighborsCache2.containsKey(nodeId)) {
                    arr = neighborsCache2.get(nodeId);
                }
                break;
        }
        return arr;
    }

    public boolean contains(int nodeId, int level) {
        return switch (level) {
            case 0 -> neighborsCache0.containsKey(nodeId);
            case 1 -> neighborsCache1.containsKey(nodeId);
            case 2 -> neighborsCache2.containsKey(nodeId);
            default -> false;
        };
    }

}
