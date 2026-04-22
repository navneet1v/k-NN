/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Emits per-query KNN search metrics via debug logging.
 * When debug is enabled, per-query metrics are logged for local observability.
 * When MetricsRegistry (OTEL) integration is added, this class will also record
 * to counters/histograms for export to observability backends.
 */
public class KNNSearchMetricsEmitter {

    private static final Logger log = LogManager.getLogger(KNNSearchMetricsEmitter.class);

    /**
     * Record per-query search metrics. Emits debug log if enabled.
     *
     * @param metrics   the aggregated per-query metrics
     * @param indexName the index that was searched
     * @param shardId   the shard ID
     * @param algorithm the algorithm used (e.g., "hnsw", "cluster")
     */
    public static void emit(KNNSearchMetrics metrics, String indexName, int shardId, String algorithm) {
        if (log.isDebugEnabled()) {
            log.debug(
                "KNN search metrics: index={}, shard={}, algorithm={}, vectors_scored={}, edges_traversed={}, "
                    + "neighbor_seeks={}, vector_bytes_prefetched={}, vector_bytes_read={}, neighbor_bytes_read={}, "
                    + "total_bytes_read={}, prefetch_groups={}, early_terminated={}, results_returned={}",
                indexName,
                shardId,
                algorithm,
                metrics.getVectorsScored(),
                metrics.getEdgesTraversed(),
                metrics.getNeighborSeeks(),
                metrics.getVectorBytesPrefetched(),
                metrics.getVectorBytesRead(),
                metrics.getNeighborBytesRead(),
                metrics.totalBytesRead(),
                metrics.getPrefetchGroupCount(),
                metrics.isEarlyTerminated(),
                metrics.getResultsReturned()
            );
        }
    }
}
