/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;

import java.io.IOException;
import java.util.BitSet;

/**
 * Scans a ranked probe list, scanning each cluster into the collector.
 *
 * <p>Owns the concerns that span clusters — SOAR dedup and filter membership — and nothing else.
 * Storage-agnostic: it only ever consumes a {@link PostingScorer}, so how postings are stored or scored
 * never reaches it. It takes the probe order as given and does not reorder it, since pruning and the
 * collector's competitive threshold both depend on visiting closest-first.
 *
 * <p>Stateless: everything is passed in, and everything it mutates is local to one call.
 */
public final class ClusterSearcher {

    private ClusterSearcher() {}

    /**
     * Scan each cluster in {@code probes} into {@code collector}.
     *
     * @param clusters the field's clusters
     * @param probes centroid ordinals to visit, closest-first
     * @param params the query and its per-query knobs
     * @param collector where hits go, and the source of the competitive threshold that drives pruning
     * @param acceptedOrds ord-space filter, or {@code null} to accept every vector; composed here with
     *     SOAR dedup into the single membership test each cluster applies before scoring
     * @return the number of clusters actually scanned (fewer than {@code probes.length} when some are empty)
     */
    public static int search(Clusters clusters, int[] probes, ScanParams params, KnnCollector collector, Bits acceptedOrds)
        throws IOException {
        // acceptedOrds is deliberately absent: null is the match-all filter, the same meaning it carries in
        // AcceptDocs#bits and LeafReader#getLiveDocs, and it is what an unfiltered query on an undeleted segment
        // arrives with. Rejecting it here would fail the most common query shape.
        if (clusters == null || params == null || probes == null) {
            throw new IllegalArgumentException("clusters, probes and params must be non-null");
        }

        if (clusters.numClusters() == 0 || probes.length == 0) {
            return 0;
        }

        // Dedup across clusters: SOAR places a boundary vector in two of them, so the same ordinal can be reached
        // twice. Local to this scan, since it means nothing outside one query.
        BitSet visited = new BitSet(clusters.numVectors());
        Bits wanted = wanted(acceptedOrds, visited, clusters.numVectors());

        ClusterScan scan = clusters.scan();
        int scanned = 0;
        for (int probe : probes) {
            Cluster cluster = clusters.get(probe);
            if (cluster.size() != 0) {
                scanCluster(scan, cluster, params, wanted, visited, collector);
                scanned++;
            }
        }
        return scanned;
    }

    /**
     * The single membership test a cluster applies before scoring: the caller's filter and this walk's dedup as one
     * {@link Bits}, so a cluster never learns there were two.
     */
    private static Bits wanted(Bits acceptedOrds, BitSet visited, int numOfVectors) {
        return new Bits() {
            @Override
            public boolean get(int ord) {
                return (acceptedOrds == null || acceptedOrds.get(ord)) && !visited.get(ord);
            }

            @Override
            public int length() {
                return acceptedOrds != null ? acceptedOrds.length() : numOfVectors;
            }
        };
    }

    /** Scan one cluster's postings into the collector. */
    private static void scanCluster(
        ClusterScan scan,
        Cluster cluster,
        ScanParams params,
        Bits wanted,
        BitSet visited,
        KnnCollector collector
    ) throws IOException {
        PostingScorer scorer = scan.scorer(cluster, params, wanted);
        while (scorer.advance(collector.minCompetitiveSimilarity())) {
            int ord = scorer.ord();
            visited.set(ord);
            collector.incVisitedCount(1);
            collector.collect(ord, scorer.score());
        }
    }
}
