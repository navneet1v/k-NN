/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class KNNSearchMetricsTests extends KNNTestCase {

    public void testTotalBytesRead() {
        KNNSearchMetrics metrics = new KNNSearchMetrics(4096, 51200, 256, 2, 100, 20, 400, false, 10);
        assertEquals(51456, metrics.totalBytesRead());
    }

    public void testPrefetchEfficiency() {
        // 100 vectors scored, 128 dimensions, 4 bytes each = 51200 useful bytes
        // 131072 bytes prefetched
        KNNSearchMetrics metrics = new KNNSearchMetrics(131072, 51200, 0, 1, 100, 0, 0, false, 10);
        float efficiency = metrics.prefetchEfficiency(128, 4);
        // 51200 / 131072 ≈ 0.39
        assertTrue(efficiency > 0.38f && efficiency < 0.40f);
    }

    public void testPrefetchEfficiency_zeroPrefetch() {
        KNNSearchMetrics metrics = new KNNSearchMetrics(0, 0, 0, 0, 100, 0, 0, false, 10);
        assertEquals(0f, metrics.prefetchEfficiency(128, 4), 0.001f);
    }

    public void testGetters() {
        KNNSearchMetrics metrics = new KNNSearchMetrics(1000, 500, 200, 3, 50, 10, 150, true, 5);
        assertEquals(1000, metrics.getVectorBytesPrefetched());
        assertEquals(500, metrics.getVectorBytesRead());
        assertEquals(200, metrics.getNeighborBytesRead());
        assertEquals(3, metrics.getPrefetchGroupCount());
        assertEquals(50, metrics.getVectorsScored());
        assertEquals(10, metrics.getNeighborSeeks());
        assertEquals(150, metrics.getEdgesTraversed());
        assertTrue(metrics.isEarlyTerminated());
        assertEquals(5, metrics.getResultsReturned());
    }
}
