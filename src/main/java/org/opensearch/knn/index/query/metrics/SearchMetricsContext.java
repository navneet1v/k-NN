/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

/**
 * Thread-local context for accumulating search metrics within a single segment search.
 * Reset before each segment search, read after search completes.
 */
public class SearchMetricsContext {
    private static final ThreadLocal<SegmentSearchMetrics> CURRENT = ThreadLocal.withInitial(SegmentSearchMetrics::new);

    /**
     * Get the current segment's metrics accumulator.
     */
    public static SegmentSearchMetrics current() {
        return CURRENT.get();
    }

    /**
     * Reset the current segment's metrics for a new search.
     */
    public static void reset() {
        CURRENT.get().reset();
    }
}
