/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

/**
 * Mutable accumulator for search metrics within a single segment.
 * One instance per doANNSearch() call. Not thread-safe — single segment = single thread.
 */
public class SegmentSearchMetrics {
    // Type A: Data Read

    /** Bytes requested via madvise(MADV_WILLNEED) through PrefetchHelper during HNSW bulk scoring.
     *  Includes padding from 128KB grouping — may be larger than actual vector bytes scored. */
    long vectorBytesPrefetched;

    /** Bytes of vector data read to compute distances. Computed as visitedCount × dimension × bytesPerElement.
     *  Represents the actual vector data touched, independent of prefetch. */
    long vectorBytesRead;

    /** Bytes read for HNSW neighbor lists from disk/mmap. Computed as sum of (numNeighbors × 4 bytes) per seek. */
    long neighborBytesRead;

    /** Number of prefetch I/O groups issued by PrefetchHelper. Each group is one madvise call covering
     *  contiguous vectors within a 128KB window. */
    int prefetchGroupCount;

    // Type B: Search Traversal

    /** Number of vectors whose distance to the query vector was computed. Captured from KnnCollector.visitedCount(). */
    long vectorsScored;

    /** Number of times a node's neighbor list was loaded from disk (graph.seek() calls). Each seek loads the
     *  entire neighbor list for one node. With M=16, each seek loads up to 16 neighbor IDs.
     *  Relationship: neighborSeeks = number of nodes whose neighbors were expanded. */
    long neighborSeeks;

    /** Number of individual neighbor IDs read from loaded neighbor lists (graph.nextNeighbor() calls).
     *  Each call returns one neighbor link. With M=16 and 10 seeks: edgesTraversed ≈ 160.
     *  Relationship: edgesTraversed = sum of neighbor list sizes across all seeks. */
    long edgesTraversed;

    public void addVectorBytesPrefetched(long bytes) {
        vectorBytesPrefetched += bytes;
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

    public void setVectorsScored(long count) {
        vectorsScored = count;
    }

    public void addNeighborSeeks(long count) {
        neighborSeeks += count;
    }

    public void addEdgesTraversed(long count) {
        edgesTraversed += count;
    }

    /**
     * Merge another segment's metrics into this one.
     */
    public void merge(SegmentSearchMetrics other) {
        this.vectorBytesPrefetched += other.vectorBytesPrefetched;
        this.vectorBytesRead += other.vectorBytesRead;
        this.neighborBytesRead += other.neighborBytesRead;
        this.prefetchGroupCount += other.prefetchGroupCount;
        this.vectorsScored += other.vectorsScored;
        this.neighborSeeks += other.neighborSeeks;
        this.edgesTraversed += other.edgesTraversed;
    }

    /**
     * Freeze into immutable snapshot.
     */
    public KNNSearchMetrics toSearchMetrics(boolean earlyTerminated, int resultsReturned) {
        return new KNNSearchMetrics(
            vectorBytesPrefetched,
            vectorBytesRead,
            neighborBytesRead,
            prefetchGroupCount,
            vectorsScored,
            neighborSeeks,
            edgesTraversed,
            earlyTerminated,
            resultsReturned
        );
    }

    /**
     * Reset all counters for reuse.
     */
    public void reset() {
        vectorBytesPrefetched = 0;
        vectorBytesRead = 0;
        neighborBytesRead = 0;
        prefetchGroupCount = 0;
        vectorsScored = 0;
        neighborSeeks = 0;
        edgesTraversed = 0;
    }
}
