/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class SegmentSearchMetricsTests extends KNNTestCase {

    public void testAccumulation() {
        SegmentSearchMetrics metrics = new SegmentSearchMetrics();
        metrics.addVectorBytesPrefetched(1024);
        metrics.addVectorBytesPrefetched(2048);
        metrics.addVectorBytesRead(512);
        metrics.addNeighborBytesRead(128);
        metrics.incrementPrefetchGroupCount();
        metrics.incrementPrefetchGroupCount();
        metrics.setVectorsScored(50);
        metrics.addNeighborSeeks(10);
        metrics.addEdgesTraversed(200);

        assertEquals(3072, metrics.vectorBytesPrefetched);
        assertEquals(512, metrics.vectorBytesRead);
        assertEquals(128, metrics.neighborBytesRead);
        assertEquals(2, metrics.prefetchGroupCount);
        assertEquals(50, metrics.vectorsScored);
        assertEquals(10, metrics.neighborSeeks);
        assertEquals(200, metrics.edgesTraversed);
    }

    public void testMerge() {
        SegmentSearchMetrics seg1 = new SegmentSearchMetrics();
        seg1.addVectorBytesPrefetched(1000);
        seg1.addVectorBytesRead(800);
        seg1.addNeighborBytesRead(100);
        seg1.incrementPrefetchGroupCount();
        seg1.setVectorsScored(50);
        seg1.addNeighborSeeks(5);
        seg1.addEdgesTraversed(100);

        SegmentSearchMetrics seg2 = new SegmentSearchMetrics();
        seg2.addVectorBytesPrefetched(2000);
        seg2.addVectorBytesRead(1600);
        seg2.addNeighborBytesRead(200);
        seg2.incrementPrefetchGroupCount();
        seg2.setVectorsScored(75);
        seg2.addNeighborSeeks(8);
        seg2.addEdgesTraversed(150);

        seg1.merge(seg2);

        assertEquals(3000, seg1.vectorBytesPrefetched);
        assertEquals(2400, seg1.vectorBytesRead);
        assertEquals(300, seg1.neighborBytesRead);
        assertEquals(2, seg1.prefetchGroupCount);
        assertEquals(125, seg1.vectorsScored);
        assertEquals(13, seg1.neighborSeeks);
        assertEquals(250, seg1.edgesTraversed);
    }

    public void testReset() {
        SegmentSearchMetrics metrics = new SegmentSearchMetrics();
        metrics.addVectorBytesPrefetched(1024);
        metrics.addVectorBytesRead(512);
        metrics.addNeighborBytesRead(128);
        metrics.incrementPrefetchGroupCount();
        metrics.setVectorsScored(50);
        metrics.addNeighborSeeks(10);
        metrics.addEdgesTraversed(200);

        metrics.reset();

        assertEquals(0, metrics.vectorBytesPrefetched);
        assertEquals(0, metrics.vectorBytesRead);
        assertEquals(0, metrics.neighborBytesRead);
        assertEquals(0, metrics.prefetchGroupCount);
        assertEquals(0, metrics.vectorsScored);
        assertEquals(0, metrics.neighborSeeks);
        assertEquals(0, metrics.edgesTraversed);
    }

    public void testToSearchMetrics() {
        SegmentSearchMetrics metrics = new SegmentSearchMetrics();
        metrics.addVectorBytesPrefetched(4096);
        metrics.addVectorBytesRead(2048);
        metrics.addNeighborBytesRead(256);
        metrics.incrementPrefetchGroupCount();
        metrics.setVectorsScored(100);
        metrics.addNeighborSeeks(20);
        metrics.addEdgesTraversed(400);

        KNNSearchMetrics snapshot = metrics.toSearchMetrics(true, 10);

        assertEquals(4096, snapshot.getVectorBytesPrefetched());
        assertEquals(2048, snapshot.getVectorBytesRead());
        assertEquals(256, snapshot.getNeighborBytesRead());
        assertEquals(1, snapshot.getPrefetchGroupCount());
        assertEquals(100, snapshot.getVectorsScored());
        assertEquals(20, snapshot.getNeighborSeeks());
        assertEquals(400, snapshot.getEdgesTraversed());
        assertTrue(snapshot.isEarlyTerminated());
        assertEquals(10, snapshot.getResultsReturned());
    }
}
