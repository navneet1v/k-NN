/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;

import javax.annotation.Nullable;
import java.io.IOException;

/**
 * The format's own {@link ClusterScan}: it holds whatever a scan shares across its clusters and hands the same
 * thing to every cluster it probes.
 *
 * <p>Two things are shared:
 *
 * <ul>
 *   <li>{@link ScanBuffers} — the scratch every cluster reuses, chiefly block buffers sized from the field's
 *       geometry. Handed in at construction, since the field's metadata is all it takes to size them, so a probe
 *       never allocates its own.
 *   <li>The {@link ScanContext}, but only when it reports {@link ScanContext#isQueryScoped()}. A centroid-free scheme
 *       quantizes the query identically for every cluster, so one context serves the whole scan instead of being
 *       rebuilt per probe. A centroid-relative scheme gets a fresh context per cluster, since its context is only
 *       valid for the centroid it was centred on.
 * </ul>
 *
 * <p>A new instance per query, so nothing here crosses queries, and within one query the clusters are visited one
 * at a time — the contract in {@link ClusterScan}. That is what lets the shared buffers be mutable, and why the
 * lazily prepared context below needs no lock.
 */
public final class DefaultClusterScan implements ClusterScan {

    @Nullable
    private final ScanBuffers scanBuffers;

    private ScanContext queryScopedContext;

    /**
     * @param scanBuffers scratch for every cluster of this scan, or {@code null} for a family that keeps none;
     *     from {@link org.opensearch.knn.clusterann.read.ClusterFactory#newScanBuffers()}
     */
    public DefaultClusterScan(@Nullable final ScanBuffers scanBuffers) {
        this.scanBuffers = scanBuffers;
    }

    @Override
    public PostingScorer scorer(final Cluster cluster, final ScanParams scanParams, @Nullable final Bits acceptedOrds) throws IOException {
        return cluster.scorer(scanContext(cluster, scanParams), acceptedOrds, scanBuffers);
    }

    /**
     * The query in the form this cluster scores against: shared across the scan when the context says it can be,
     * else prepared per cluster.
     *
     * <p>The flag is read off the prepared context rather than asked of the cluster beforehand, so it describes the
     * value actually returned. That costs nothing: once a query-scoped context has been kept, no later cluster of
     * the scan prepares one at all.
     */
    private ScanContext scanContext(final Cluster cluster, final ScanParams scanParams) throws IOException {
        if (queryScopedContext != null) {
            return queryScopedContext;
        }
        final ScanContext prepared = cluster.prepareScan(scanParams);
        if (prepared.isQueryScoped()) {
            queryScopedContext = prepared;
        }
        return prepared;
    }
}
