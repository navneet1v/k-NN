/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class ANNSearchMetricsTests extends KNNTestCase {

    public void testAccumulation() {
        ANNSearchMetrics metrics = new ANNSearchMetrics();
        metrics.addVectorBytesPrefetched(1024);
        metrics.addVectorBytesPrefetched(2048);
        metrics.addVectorBytesRead(512);
        metrics.addNeighborBytesRead(128);
        metrics.incrementPrefetchGroupCount();
        metrics.incrementPrefetchGroupCount();
        metrics.setVectorsScored(50);
        metrics.addNeighborSeeks(10);
        metrics.addEdgesTraversed(200);

        assertEquals(3072, metrics.getVectorBytesPrefetched());
        assertEquals(512, metrics.getVectorBytesRead());
        assertEquals(128, metrics.getNeighborBytesRead());
        assertEquals(2, metrics.getPrefetchGroupCount());
        assertEquals(50, metrics.getVectorsScored());
        assertEquals(10, metrics.getNeighborSeeks());
        assertEquals(200, metrics.getEdgesTraversed());
    }

    public void testMerge() {
        ANNSearchMetrics seg1 = new ANNSearchMetrics();
        seg1.addVectorBytesPrefetched(1000);
        seg1.addVectorBytesRead(800);
        seg1.addNeighborBytesRead(100);
        seg1.incrementPrefetchGroupCount();
        seg1.setVectorsScored(50);
        seg1.addNeighborSeeks(5);
        seg1.addEdgesTraversed(100);

        ANNSearchMetrics seg2 = new ANNSearchMetrics();
        seg2.addVectorBytesPrefetched(2000);
        seg2.addVectorBytesRead(1600);
        seg2.addNeighborBytesRead(200);
        seg2.incrementPrefetchGroupCount();
        seg2.setVectorsScored(75);
        seg2.addNeighborSeeks(8);
        seg2.addEdgesTraversed(150);

        seg1.merge(seg2);

        assertEquals(3000, seg1.getVectorBytesPrefetched());
        assertEquals(2400, seg1.getVectorBytesRead());
        assertEquals(300, seg1.getNeighborBytesRead());
        assertEquals(2, seg1.getPrefetchGroupCount());
        assertEquals(125, seg1.getVectorsScored());
        assertEquals(13, seg1.getNeighborSeeks());
        assertEquals(250, seg1.getEdgesTraversed());
    }

    public void testReset() {
        ANNSearchMetrics metrics = new ANNSearchMetrics();
        metrics.addVectorBytesPrefetched(1024);
        metrics.addVectorBytesRead(512);
        metrics.addNeighborBytesRead(128);
        metrics.incrementPrefetchGroupCount();
        metrics.setVectorsScored(50);
        metrics.addNeighborSeeks(10);
        metrics.addEdgesTraversed(200);
        metrics.setEarlyTerminated(true);
        metrics.setResultsReturned(5);

        metrics.reset();

        assertEquals(0, metrics.getVectorBytesPrefetched());
        assertEquals(0, metrics.getVectorBytesRead());
        assertEquals(0, metrics.getNeighborBytesRead());
        assertEquals(0, metrics.getPrefetchGroupCount());
        assertEquals(0, metrics.getVectorsScored());
        assertEquals(0, metrics.getNeighborSeeks());
        assertEquals(0, metrics.getEdgesTraversed());
        assertFalse(metrics.isEarlyTerminated());
        assertEquals(0, metrics.getResultsReturned());
    }

    public void testTotalBytesRead() {
        ANNSearchMetrics metrics = new ANNSearchMetrics();
        metrics.addVectorBytesRead(51200);
        metrics.addNeighborBytesRead(256);
        assertEquals(51456, metrics.totalBytesRead());
    }

    public void testPrefetchGroupingEfficiency() {
        ANNSearchMetrics metrics = new ANNSearchMetrics();
        metrics.addVectorBytesPrefetched(131072);
        metrics.addVectorBytesInPrefetchGroups(51200);
        float efficiency = metrics.prefetchGroupingEfficiency();
        // 51200 / 131072 ≈ 0.39
        assertTrue(efficiency > 0.38f && efficiency < 0.40f);
    }

    public void testPrefetchGroupingEfficiency_zeroPrefetch() {
        ANNSearchMetrics metrics = new ANNSearchMetrics();
        assertEquals(-1f, metrics.prefetchGroupingEfficiency(), 0.001f);
    }
}
