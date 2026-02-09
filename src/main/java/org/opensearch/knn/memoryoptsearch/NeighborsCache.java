/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.internal.hppc.IntArrayList;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class NeighborsCache {

    private final Map<Integer, IntArrayList> neighborsCache0;
    private final Map<Integer, IntArrayList> neighborsCache1;
    private final Map<Integer, IntArrayList> neighborsCache2;

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

    public void put(int nodeId, int level, IntArrayList neighbors) {
        switch (level) {
            case 0 -> neighborsCache0.putIfAbsent(nodeId, neighbors);
            case 1 -> neighborsCache1.putIfAbsent(nodeId, neighbors);
            case 2 -> neighborsCache2.putIfAbsent(nodeId, neighbors);
        }
    }

    public IntArrayList load(int nodeId, int level) {
        return switch (level) {
            case 0 -> neighborsCache0.get(nodeId);
            case 1 -> neighborsCache1.get(nodeId);
            case 2 -> neighborsCache2.get(nodeId);
            default -> null;
        };
    }

}
