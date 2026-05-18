/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class KNNSearchMetricsEmitterTests extends KNNTestCase {

    public void testEmitSegmentLevelANNSearchMetrics() {
        ANNSearchMetrics metrics = new ANNSearchMetrics();
        metrics.addVectorBytesPrefetched(4096);
        metrics.addVectorBytesRead(51200);
        metrics.addNeighborBytesRead(256);
        metrics.incrementPrefetchGroupCount();
        metrics.incrementPrefetchGroupCount();
        metrics.setVectorsScored(100);
        metrics.addNeighborSeeks(20);
        metrics.addEdgesTraversed(400);
        metrics.setResultsReturned(10);
        KNNSearchMetricsEmitter.emitSegmentLevelANNSearchMetrics(metrics, "test-index", 0, "hnsw", "seg_0", 5000);
    }

    public void testEmitExactSearchSegmentMetrics() {
        ExactSearchMetrics metrics = new ExactSearchMetrics();
        metrics.addDocsScored(50);
        metrics.addVectorBytesPrefetched(131072);
        metrics.addVectorBytesRead(25600);
        metrics.addPrefetchGroupCount(2);
        KNNSearchMetricsEmitter.emitExactSearchSegmentMetrics(metrics, "test-index", 0, "seg_0", 5000);
    }

    public void testEmitShardLevelMetrics() {
        ANNSearchMetrics annMetrics = new ANNSearchMetrics();
        annMetrics.addVectorBytesPrefetched(131072);
        annMetrics.addVectorBytesRead(76800);
        annMetrics.addNeighborBytesRead(9600);
        annMetrics.setVectorsScored(150);
        annMetrics.addNeighborSeeks(150);
        annMetrics.addEdgesTraversed(2400);
        annMetrics.setEarlyTerminated(true);
        annMetrics.setResultsReturned(10);

        ExactSearchMetrics exactMetrics = new ExactSearchMetrics();
        exactMetrics.addDocsScored(30);
        exactMetrics.addVectorBytesPrefetched(65536);
        exactMetrics.addVectorBytesRead(15360);
        exactMetrics.addPrefetchGroupCount(1);

        KNNSearchMetricsEmitter.emitShardLevelMetrics(annMetrics, exactMetrics, "test-index", 0, true, 3, 15000);
    }
}
