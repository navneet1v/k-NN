/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class ExactSearchMetricsTests extends KNNTestCase {

    public void testAccumulation() {
        ExactSearchMetrics metrics = new ExactSearchMetrics();
        metrics.addDocsScored(50);
        metrics.addVectorBytesRead(25600);
        metrics.addVectorBytesPrefetched(131072);
        metrics.addPrefetchGroupCount(2);
        metrics.addDocsScored(30);
        metrics.addVectorBytesRead(15360);
        metrics.addVectorBytesPrefetched(65536);
        metrics.addPrefetchGroupCount(1);

        assertEquals(80, metrics.getDocsScored());
        assertEquals(40960, metrics.getVectorBytesRead());
        assertEquals(196608, metrics.getVectorBytesPrefetched());
        assertEquals(3, metrics.getPrefetchGroupCount());
    }

    public void testMerge() {
        ExactSearchMetrics seg1 = new ExactSearchMetrics();
        seg1.addDocsScored(10);
        seg1.addVectorBytesRead(5120);
        seg1.addVectorBytesPrefetched(131072);
        seg1.addPrefetchGroupCount(1);

        ExactSearchMetrics seg2 = new ExactSearchMetrics();
        seg2.addDocsScored(20);
        seg2.addVectorBytesRead(10240);
        seg2.addVectorBytesPrefetched(65536);
        seg2.addPrefetchGroupCount(2);

        seg1.merge(seg2);

        assertEquals(30, seg1.getDocsScored());
        assertEquals(15360, seg1.getVectorBytesRead());
        assertEquals(196608, seg1.getVectorBytesPrefetched());
        assertEquals(3, seg1.getPrefetchGroupCount());
    }

    public void testReset() {
        ExactSearchMetrics metrics = new ExactSearchMetrics();
        metrics.addDocsScored(100);
        metrics.addVectorBytesRead(51200);
        metrics.addVectorBytesPrefetched(262144);
        metrics.addPrefetchGroupCount(4);

        metrics.reset();

        assertEquals(0, metrics.getDocsScored());
        assertEquals(0, metrics.getVectorBytesRead());
        assertEquals(0, metrics.getVectorBytesPrefetched());
        assertEquals(0, metrics.getPrefetchGroupCount());
    }
}
