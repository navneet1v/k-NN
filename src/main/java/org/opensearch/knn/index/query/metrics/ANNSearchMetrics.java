/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

/**
 * Mutable accumulator for ANN search metrics.
 * Not thread-safe — single segment = single thread.
 *
 * Fields:
 * - vectorBytesPrefetched: Bytes requested via madvise(MADV_WILLNEED) through PrefetchHelper + single-node score() calls. Includes 128KB grouping padding — may exceed actual vector bytes scored.
 * - vectorBytesRead: Bytes of vector data read to compute distances. Computed as visitedCount × vectorByteLength. Represents actual vector data touched, independent of prefetch grouping.
 * - neighborBytesRead: Bytes read for HNSW neighbor lists from disk/mmap. Computed as sum of (numNeighbors × 4 bytes) per seek.
 * - prefetchGroupCount: Number of prefetch I/O groups issued. Each group is one madvise call (bulk) or one score() call (single node).
 * - vectorsScored: Number of vectors whose distance to the query vector was computed. Captured from KnnCollector.visitedCount().
 * - neighborSeeks: Number of times a node's neighbor list was loaded from disk (graph.seek() calls).
 * - edgesTraversed: Number of individual neighbor IDs read from loaded neighbor lists (graph.nextNeighbor() calls).
 * - earlyTerminated: Whether the search hit the visit limit and stopped before exhausting all candidates.
 * - resultsReturned: Number of results returned at the segment level (before top-k merge across segments).
 */
public class ANNSearchMetrics {
    long vectorBytesPrefetched;
    long vectorBytesInPrefetchGroups;
    long vectorBytesRead;
    long neighborBytesRead;
    int prefetchGroupCount;
    long vectorsInPrefetchGroups;
    long vectorsScored;
    long neighborSeeks;
    long edgesTraversed;
    boolean earlyTerminated;
    int resultsReturned;

    public void addVectorBytesPrefetched(long bytes) {
        vectorBytesPrefetched += bytes;
    }

    public void addVectorBytesInPrefetchGroups(long bytes) {
        vectorBytesInPrefetchGroups += bytes;
    }

    public void addVectorBytesRead(long bytes) {
        vectorBytesRead += bytes;
    }

    public void addNeighborBytesRead(long bytes) {
        neighborBytesRead += bytes;
    }

    public void incrementPrefetchGroupCount() {
        prefetchGroupCount++;
    }

    public void addVectorsInPrefetchGroups(int count) {
        vectorsInPrefetchGroups += count;
    }

    public void setVectorsScored(long count) {
        vectorsScored = count;
    }

    public void addNeighborSeeks(long count) {
        neighborSeeks += count;
    }

    public void addEdgesTraversed(long count) {
        edgesTraversed += count;
    }

    public void setEarlyTerminated(boolean earlyTerminated) {
        this.earlyTerminated = earlyTerminated;
    }

    public void setResultsReturned(int resultsReturned) {
        this.resultsReturned = resultsReturned;
    }

    public long getVectorBytesPrefetched() {
        return vectorBytesPrefetched;
    }

    public long getVectorBytesInPrefetchGroups() {
        return vectorBytesInPrefetchGroups;
    }

    public long getVectorBytesRead() {
        return vectorBytesRead;
    }

    public long getNeighborBytesRead() {
        return neighborBytesRead;
    }

    public int getPrefetchGroupCount() {
        return prefetchGroupCount;
    }

    public long getVectorsInPrefetchGroups() {
        return vectorsInPrefetchGroups;
    }

    public long getVectorsScored() {
        return vectorsScored;
    }

    public long getNeighborSeeks() {
        return neighborSeeks;
    }

    public long getEdgesTraversed() {
        return edgesTraversed;
    }

    public boolean isEarlyTerminated() {
        return earlyTerminated;
    }

    public int getResultsReturned() {
        return resultsReturned;
    }

    public long totalBytesRead() {
        return vectorBytesRead + neighborBytesRead;
    }

    /**
     * Ratio of actual vector bytes within prefetch groups to total prefetched bytes.
     * Values < 1 indicate gaps between vectors within 128KB windows.
     * Returns -1 when no prefetch occurred.
     */
    public float prefetchGroupingEfficiency() {
        if (vectorBytesPrefetched == 0) return -1f;
        return (float) vectorBytesInPrefetchGroups / vectorBytesPrefetched;
    }

    public void merge(ANNSearchMetrics other) {
        this.vectorBytesPrefetched += other.vectorBytesPrefetched;
        this.vectorBytesInPrefetchGroups += other.vectorBytesInPrefetchGroups;
        this.vectorBytesRead += other.vectorBytesRead;
        this.neighborBytesRead += other.neighborBytesRead;
        this.prefetchGroupCount += other.prefetchGroupCount;
        this.vectorsInPrefetchGroups += other.vectorsInPrefetchGroups;
        this.vectorsScored += other.vectorsScored;
        this.neighborSeeks += other.neighborSeeks;
        this.edgesTraversed += other.edgesTraversed;
        this.resultsReturned += other.resultsReturned;
        this.earlyTerminated |= other.earlyTerminated;
    }

    public void reset() {
        vectorBytesPrefetched = 0;
        vectorBytesInPrefetchGroups = 0;
        vectorBytesRead = 0;
        neighborBytesRead = 0;
        prefetchGroupCount = 0;
        vectorsInPrefetchGroups = 0;
        vectorsScored = 0;
        neighborSeeks = 0;
        edgesTraversed = 0;
        earlyTerminated = false;
        resultsReturned = 0;
    }
}
