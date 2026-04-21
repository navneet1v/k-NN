/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

/**
 * Immutable snapshot of metrics for a single KNN search query.
 * Built by accumulating per-segment metrics during search.
 */
public class KNNSearchMetrics {
    // Type A: Data Read

    /** Bytes requested via madvise(MADV_WILLNEED) through PrefetchHelper during HNSW bulk scoring.
     *  Includes padding from 128KB grouping — may be larger than actual vector bytes scored. */
    private final long vectorBytesPrefetched;

    /** Bytes of vector data read to compute distances. Computed as visitedCount × dimension × bytesPerElement.
     *  Represents the actual vector data touched, independent of prefetch. */
    private final long vectorBytesRead;

    /** Bytes read for HNSW neighbor lists from disk/mmap. Computed as sum of (numNeighbors × 4 bytes) per seek. */
    private final long neighborBytesRead;

    /** Number of prefetch I/O groups issued by PrefetchHelper. Each group is one madvise call covering
     *  contiguous vectors within a 128KB window. */
    private final int prefetchGroupCount;

    // Type B: Search Traversal

    /** Number of vectors whose distance to the query vector was computed. Captured from KnnCollector.visitedCount(). */
    private final long vectorsScored;

    /** Number of times a node's neighbor list was loaded from disk (graph.seek() calls). Each seek loads the
     *  entire neighbor list for one node. With M=16, each seek loads up to 16 neighbor IDs.
     *  Relationship: neighborSeeks = number of nodes whose neighbors were expanded. */
    private final long neighborSeeks;

    /** Number of individual neighbor IDs read from loaded neighbor lists (graph.nextNeighbor() calls).
     *  Each call returns one neighbor link. With M=16 and 10 seeks: edgesTraversed ≈ 160.
     *  Relationship: edgesTraversed = sum of neighbor list sizes across all seeks. */
    private final long edgesTraversed;

    /** Whether the search hit the visit limit and stopped before exhausting all candidates. */
    private final boolean earlyTerminated;

    /** Number of results returned at the segment level (before top-k merge across segments). */
    private final int resultsReturned;

    public KNNSearchMetrics(
        long vectorBytesPrefetched,
        long vectorBytesRead,
        long neighborBytesRead,
        int prefetchGroupCount,
        long vectorsScored,
        long neighborSeeks,
        long edgesTraversed,
        boolean earlyTerminated,
        int resultsReturned
    ) {
        this.vectorBytesPrefetched = vectorBytesPrefetched;
        this.vectorBytesRead = vectorBytesRead;
        this.neighborBytesRead = neighborBytesRead;
        this.prefetchGroupCount = prefetchGroupCount;
        this.vectorsScored = vectorsScored;
        this.neighborSeeks = neighborSeeks;
        this.edgesTraversed = edgesTraversed;
        this.earlyTerminated = earlyTerminated;
        this.resultsReturned = resultsReturned;
    }

    public long getVectorBytesPrefetched() {
        return vectorBytesPrefetched;
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
     * Ratio of useful vector bytes (scored) to total prefetched bytes.
     * Lower values indicate more wasted prefetch due to 128KB grouping.
     */
    public float prefetchEfficiency(int dimension, int bytesPerElement) {
        if (vectorBytesPrefetched == 0) return 0f;
        return (float) (vectorsScored * dimension * bytesPerElement) / vectorBytesPrefetched;
    }
}
