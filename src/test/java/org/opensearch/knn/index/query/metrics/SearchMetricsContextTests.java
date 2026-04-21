/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class SearchMetricsContextTests extends KNNTestCase {

    public void testCurrentReturnsInstance() {
        SegmentSearchMetrics metrics = SearchMetricsContext.current();
        assertNotNull(metrics);
    }

    public void testCurrentReturnsSameInstanceOnSameThread() {
        SegmentSearchMetrics first = SearchMetricsContext.current();
        SegmentSearchMetrics second = SearchMetricsContext.current();
        assertSame(first, second);
    }

    public void testResetClearsMetrics() {
        SegmentSearchMetrics metrics = SearchMetricsContext.current();
        metrics.addVectorBytesPrefetched(5000);
        metrics.addEdgesTraversed(100);

        SearchMetricsContext.reset();

        SegmentSearchMetrics after = SearchMetricsContext.current();
        assertEquals(0, after.vectorBytesPrefetched);
        assertEquals(0, after.edgesTraversed);
    }

    public void testIsolationAcrossThreads() throws Exception {
        SearchMetricsContext.current().addEdgesTraversed(999);

        Thread other = new Thread(() -> {
            SegmentSearchMetrics otherMetrics = SearchMetricsContext.current();
            assertEquals(0, otherMetrics.edgesTraversed);
            otherMetrics.addEdgesTraversed(1);
        });
        other.start();
        other.join();

        // Original thread unaffected
        assertEquals(999, SearchMetricsContext.current().edgesTraversed);

        // Clean up
        SearchMetricsContext.reset();
    }
}
