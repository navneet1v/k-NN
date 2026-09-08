/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.orchestration;

import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.reader.Cluster;
import org.opensearch.knn.clusterann.reader.PostingScorer;
import org.opensearch.knn.clusterann.reader.ScanParams;

import java.io.IOException;

/**
 * Query-scoped strategy for scanning clusters.
 *
 * <p>A new instance is created per query and serves every cluster in that query, so implementations
 * may hold query-level context and reuse it across clusters rather than re-deriving it per cluster.
 * {@link #scan} may be called concurrently for different clusters, so that context must be
 * thread-safe.
 */
public interface ClusterScan {

    /**
     * Produces a scorer for one cluster, wrapping the {@link Cluster#prepareScan} then
     * {@link Cluster#scorer} sequence.
     *
     * <p>The returned scorer holds the per-cluster state, is single-threaded, and is bound to
     * {@code cluster}.
     */
    PostingScorer scan(Cluster cluster, ScanParams scanParams, Bits acceptedOrds) throws IOException;
}
