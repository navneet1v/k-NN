/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.metrics;

import lombok.extern.log4j.Log4j2;

/**
 * Emits per-query KNN search metrics
 */
@Log4j2
public class KNNSearchMetricsEmitter {
    public static void emitSegmentLevelANNSearchMetrics(
        final ANNSearchMetrics metrics,
        final String indexName,
        int shardId,
        final String algorithm,
        String segment,
        int totalDocsInSegment
    ) {
        log.info(
            "KNN segment level search metrics: index={}, shard={}, segment={}, algorithm={}, total_docs={}, "
                + "vectors_scored={}, edges_traversed={}, "
                + "neighbor_seeks={}, vector_bytes_prefetched={}, vector_bytes_in_prefetch_groups={}, vector_bytes_read={}, "
                + "neighbor_bytes_read={}, total_bytes_read={}, prefetch_groups={}, vectors_in_prefetch_groups={}, "
                + "prefetch_grouping_efficiency={}, early_terminated={}, results_returned={}",
            indexName,
            shardId,
            segment,
            algorithm,
            totalDocsInSegment,
            metrics.getVectorsScored(),
            metrics.getEdgesTraversed(),
            metrics.getNeighborSeeks(),
            metrics.getVectorBytesPrefetched(),
            metrics.getVectorBytesInPrefetchGroups(),
            metrics.getVectorBytesRead(),
            metrics.getNeighborBytesRead(),
            metrics.totalBytesRead(),
            metrics.getPrefetchGroupCount(),
            metrics.getVectorsInPrefetchGroups(),
            metrics.prefetchGroupingEfficiency(),
            metrics.isEarlyTerminated(),
            metrics.getResultsReturned()
        );
    }

    public static void emitExactSearchSegmentMetrics(
        final ExactSearchMetrics metrics,
        final String indexName,
        int shardId,
        String segment,
        int totalDocsInSegment
    ) {
        log.info(
            "KNN exact search segment metrics: index={}, shard={}, segment={}, total_docs={}, docs_scored={}, "
                + "vector_bytes_prefetched={}, vector_bytes_read={}, prefetch_groups={}",
            indexName,
            shardId,
            segment,
            totalDocsInSegment,
            metrics.getDocsScored(),
            metrics.getVectorBytesPrefetched(),
            metrics.getVectorBytesRead(),
            metrics.getPrefetchGroupCount()
        );
    }

    /**
     * Emit combined shard-level metrics covering both ANN and exact search.
     *
     * @param annMetrics      aggregated ANN search metrics across all segments
     * @param exactMetrics    aggregated exact search metrics across all segments
     * @param indexName       the index that was searched
     * @param shardId         the shard ID
     * @param isMOS           whether memory-optimized search was used
     * @param numSegments     number of segments searched in this shard
     */
    public static void emitShardLevelMetrics(
        final ANNSearchMetrics annMetrics,
        final ExactSearchMetrics exactMetrics,
        final String indexName,
        int shardId,
        final boolean isMOS,
        int numSegments,
        int totalDocs
    ) {
        log.info(
            "KNN shard search metrics: index={}, shard={}, mos={}, num_segments={}, total_docs={}, "
                + "ann_vectors_scored={}, ann_edges_traversed={}, ann_neighbor_seeks={}, "
                + "ann_vector_bytes_prefetched={}, ann_vector_bytes_read={}, ann_neighbor_bytes_read={}, "
                + "ann_total_bytes_read={}, ann_prefetch_groups={}, ann_results_returned={}, "
                + "exact_docs_scored={}, exact_vector_bytes_prefetched={}, exact_vector_bytes_read={}, exact_prefetch_groups={}",
            indexName,
            shardId,
            isMOS,
            numSegments,
            totalDocs,
            annMetrics.getVectorsScored(),
            annMetrics.getEdgesTraversed(),
            annMetrics.getNeighborSeeks(),
            annMetrics.getVectorBytesPrefetched(),
            annMetrics.getVectorBytesRead(),
            annMetrics.getNeighborBytesRead(),
            annMetrics.totalBytesRead(),
            annMetrics.getPrefetchGroupCount(),
            annMetrics.getResultsReturned(),
            exactMetrics.getDocsScored(),
            exactMetrics.getVectorBytesPrefetched(),
            exactMetrics.getVectorBytesRead(),
            exactMetrics.getPrefetchGroupCount()
        );
    }
}
