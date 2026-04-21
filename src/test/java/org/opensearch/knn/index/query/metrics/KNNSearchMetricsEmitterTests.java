/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.opensearch.knn.KNNTestCase;

public class KNNSearchMetricsEmitterTests extends KNNTestCase {

    public void testEmitDoesNotThrow() {
        KNNSearchMetrics metrics = new KNNSearchMetrics(4096, 51200, 256, 2, 100, 20, 400, false, 10);
        // Should not throw regardless of log level
        KNNSearchMetricsEmitter.emit(metrics, "test-index", 0, "hnsw");
    }

    public void testEmitWithZeroMetrics() {
        KNNSearchMetrics metrics = new KNNSearchMetrics(0, 0, 0, 0, 0, 0, 0, false, 0);
        KNNSearchMetricsEmitter.emit(metrics, "test-index", 0, "cluster");
    }

    public void testEmitWithEarlyTermination() {
        KNNSearchMetrics metrics = new KNNSearchMetrics(131072, 76800, 9600, 8, 150, 150, 2400, true, 10);
        KNNSearchMetricsEmitter.emit(metrics, "my-vectors", 2, "hnsw");
    }
}
