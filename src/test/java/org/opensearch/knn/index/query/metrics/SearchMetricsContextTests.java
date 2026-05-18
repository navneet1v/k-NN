/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class SearchMetricsContextTests extends KNNTestCase {

    public void testCurrentReturnsInstance() {
        ANNSearchMetrics metrics = SearchMetricsContext.current();
        assertNotNull(metrics);
    }

    public void testCurrentReturnsSameInstanceOnSameThread() {
        ANNSearchMetrics first = SearchMetricsContext.current();
        ANNSearchMetrics second = SearchMetricsContext.current();
        assertSame(first, second);
    }

    public void testResetClearsMetrics() {
        ANNSearchMetrics metrics = SearchMetricsContext.current();
        metrics.addVectorBytesPrefetched(5000);
        metrics.addEdgesTraversed(100);

        SearchMetricsContext.reset();

        ANNSearchMetrics after = SearchMetricsContext.current();
        assertEquals(0, after.getVectorBytesPrefetched());
        assertEquals(0, after.getEdgesTraversed());
    }

    public void testIsolationAcrossThreads() throws Exception {
        SearchMetricsContext.current().addEdgesTraversed(999);

        Thread other = new Thread(() -> {
            ANNSearchMetrics otherMetrics = SearchMetricsContext.current();
            assertEquals(0, otherMetrics.getEdgesTraversed());
            otherMetrics.addEdgesTraversed(1);
        });
        other.start();
        other.join();

        // Original thread unaffected
        assertEquals(999, SearchMetricsContext.current().getEdgesTraversed());

        // Clean up
        SearchMetricsContext.reset();
    }
}
