/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;

import java.io.IOException;

/**
 * Query-scoped strategy for scanning clusters.
 *
 * <p>A new instance is created per query and serves every cluster in that query, so implementations
 * may hold query-level context and reuse it across clusters rather than re-deriving it per cluster.
 *
 * <p><b>One cluster at a time.</b> The clusters of a single scan are visited sequentially: a caller must drain the
 * scorer it was given before asking for the next cluster's. That lets the per-query context carry mutable scratch
 * sized to the field — the data-blind int8 family reuses one block reader across every probe this way, rather than
 * allocating a block's worth of buffers per cluster — so neither the context nor the scorer need be thread-safe.
 * Concurrency across queries is unaffected, since each query gets its own instance.
 */
public interface ClusterScan {

    /**
     * Produces a scorer for one cluster, wrapping the {@link Cluster#prepareScan} then
     * {@link Cluster#scorer} sequence.
     *
     * <p>The returned scorer holds the per-cluster state, is single-threaded, and is bound to
     * {@code cluster}.
     */
    PostingScorer scorer(Cluster cluster, ScanParams scanParams, Bits acceptedOrds) throws IOException;
}
