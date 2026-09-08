/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.store.IndexInput;
import org.opensearch.common.Nullable;
import org.opensearch.knn.clusterann.reader.orchestration.ClusterScan;
import org.opensearch.knn.clusterann.reader.orchestration.ScanContext;

import java.io.IOException;

/**
 * The clusters of one field: a persistent, query-independent handle. Intended to be built once per field and
 * reused. It has access to all the information of the clusters and acts as an abstraction to navigate and get
 * the information.
 *
 * <p>Immutable and thread-safe. It holds the {@code .clap} input, the immutable {@link ClusterANNFieldMeta}, and a
 * <em>base</em> transformed-centroid reader over {@code .clac}..
 */
public final class Clusters {

    private final ClusterANNFieldMeta fieldMeta;
    private final ClusterFactory clusterFactory;

    /**
     * @param postings the {@code .clap} file; a cluster's own posting is sliced out of it per {@link #get}
     * @param centroids the {@code .clac} file
     * @param rotation the {@code .clar} file
     * @param fieldMeta this field's entry from {@code .clam}
     */
    public Clusters(
        final IndexInput postings,
        final IndexInput centroids,
        @Nullable final IndexInput rotation,
        final ClusterANNFieldMeta fieldMeta
    ) throws IOException {
        this.fieldMeta = fieldMeta;
        this.clusterFactory = new ClusterFactory(fieldMeta, postings, centroids, rotation);
    }

    /** Number of clusters in this field. */
    public int numClusters() {
        return fieldMeta.centroidCount();
    }

    /** Number of vectors in this field, across all clusters. */
    public int numVectors() {
        return fieldMeta.vectorCount();
    }

    /**
     * Number of vectors in the cluster with this centroid ordinal, primary and SOAR together.
     */
    public int clusterSize(int ordinal) {
        return fieldMeta.clusterSizes()[ordinal];
    }

    /**
     * Returns a scanner for one query.
     *
     * <p>Call this once per query. Then call {@link ClusterScan#scan} once for each cluster you want to
     * search. Each call builds its own {@link ScanContext}, so the scanner holds no state and the calls
     * can run in parallel.
     */
    public ClusterScan scan() {
        return (cluster, scanParams, acceptedOrds) -> {
            ScanContext scanContext = cluster.prepareScan(scanParams);
            return cluster.scorer(scanContext, acceptedOrds);
        };
    }

    /**
     * The cluster with this centroid ordinal. <b>Reads nothing</b> — it looks up layout facts and hands the cluster
     * private cursors to read through when it needs to. Both the posting and the cluster's own centroid are read
     * lazily inside {@link Cluster#scorer}.
     *
     * <p>That laziness is deliberate: it makes {@link Cluster#prefetch} and a walk's skip checks free, so a get
     * costs no I/O.
     */
    public Cluster get(int ordinal) throws IOException {
        return clusterFactory.create(ordinal);
    }
}
