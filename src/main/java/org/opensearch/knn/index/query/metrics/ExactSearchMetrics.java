/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import lombok.Getter;

/**
 * Mutable accumulator for exact search metrics.
 * Tracks vector bytes prefetched, read, and docs scored during brute-force exact search.
 */
@Getter
public class ExactSearchMetrics {
    long vectorBytesPrefetched;
    long vectorBytesRead;
    long docsScored;
    int prefetchGroupCount;

    public void addVectorBytesPrefetched(long bytes) {
        vectorBytesPrefetched += bytes;
    }

    public void addVectorBytesRead(long bytes) {
        vectorBytesRead += bytes;
    }

    public void addDocsScored(long count) {
        docsScored += count;
    }

    public void addPrefetchGroupCount(int count) {
        prefetchGroupCount += count;
    }

    public void merge(ExactSearchMetrics other) {
        this.vectorBytesPrefetched += other.vectorBytesPrefetched;
        this.vectorBytesRead += other.vectorBytesRead;
        this.docsScored += other.docsScored;
        this.prefetchGroupCount += other.prefetchGroupCount;
    }

    public void reset() {
        vectorBytesPrefetched = 0;
        vectorBytesRead = 0;
        docsScored = 0;
        prefetchGroupCount = 0;
    }
}
