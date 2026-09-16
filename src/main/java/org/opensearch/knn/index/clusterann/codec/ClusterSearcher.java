/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;

/**
 * Searches {@link Clusters} for a query — the <em>walk</em> over the planner's ranked probe list,
 * scanning each probed cluster into the collector.
 *
 * <p>Stateless: everything the walk needs is passed in, and everything it mutates is local to one call.
 * SOAR dedup in particular is bookkeeping internal to a single walk, so it is created here rather than
 * handed in — a caller has no reason to see it.
 *
 * <p>It also owns the level-1 prefetch <em>policy</em>: it keeps a window of upcoming postings hinted
 * ahead of the cursor, because only the walk knows the probe order. Crucially it hints a posting only if
 * it will actually scan it — a cluster the filter is about to skip is never fetched — and leaves how much
 * of a posting to warm to the cluster. Level-2 (block) prefetch lives inside the block reader.
 *
 * <p>The walk is storage-agnostic: it forwards {@link ScanParams} to each cluster and only ever consumes
 * a {@link PostingScorer}, so nothing about how postings are stored or scored reaches it.
 */
public final class ClusterSearcher {

    /** Clusters kept hinted ahead of the cursor. */
    private static final int PREFETCH_WINDOW = 8;

    /** Below this filter selectivity, skip clusters unlikely to hold a single matching doc. */
    private static final float SELECTIVE_FILTER_THRESHOLD = 0.10f;

    /** Expected matches below this in a cluster aren't worth the scan. */
    private static final float MIN_EXPECTED_MATCHES = 0.5f;

    private ClusterSearcher() {}

    /**
     * Walk {@code probes} — centroid ordinals, ranked closest-first — keeping a window hinted ahead and
     * scanning each cluster into {@code collector}; returns the number of clusters actually scanned.
     *
     * @param acceptedOrds ord-space filter, or {@code null} to accept every vector. Composed here with
     *     this walk's SOAR dedup into the single membership test each cluster applies before scoring.
     */
    public static int search(
        Clusters clusters,
        int[] probes,
        ScanParams params,
        KnnCollector collector,
        Bits acceptedOrds
    ) throws IOException {
        if (clusters.numClusters() == 0) {
            return 0;
        }
        // Dedup across clusters: SOAR places a boundary vector in two of them, so the same ordinal can be
        // reached twice. Local to this walk.
        BitSet visited = new BitSet(clusters.numVectors());
        Bits wanted = new Bits() {
            @Override
            public boolean get(int ord) {
                return (acceptedOrds == null || acceptedOrds.get(ord)) && visited.get(ord) == false;
            }

            @Override
            public int length() {
                return acceptedOrds != null ? acceptedOrds.length() : Integer.MAX_VALUE;
            }
        };

        // One scan for this query over this field: the per-query work many postings share (projecting and
        // quantizing the query against a reference centroid) happens once behind it.
        ClusterScan scan = clusters.scan(params);

        float selectivity = params.filterSelectivity();
        boolean filterActive = selectivity > 0f && selectivity < SELECTIVE_FILTER_THRESHOLD;
        int scanned = 0;
        int hintedUpTo = -1; // high-water mark, so each probe is hinted at most once
        for (int i = 0; i < probes.length; i++) {
            // Keep the window ahead — but only hint clusters we will actually scan, so a filtered-out
            // cluster costs no I/O. Obtaining a cluster reads nothing, so warming one is just the hint
            // itself; partial leaves the bulk of the posting to the scan's own streaming. Handed over as a
            // set so the hints can be issued in file order — this is a burst of `window` at the first
            // probe, then one per step.
            int target = Math.min(i + PREFETCH_WINDOW, probes.length - 1);
            if (hintedUpTo < target) {
                Cluster[] batch = new Cluster[target - hintedUpTo];
                int n = 0;
                while (hintedUpTo < target) {
                    Cluster ahead = clusters.cluster(probes[++hintedUpTo]);
                    if (worthScanning(ahead, filterActive, selectivity)) {
                        batch[n++] = ahead;
                    }
                }
                if (n > 0) {
                    clusters.prefetch(n == batch.length ? batch : Arrays.copyOf(batch, n), true);
                }
            }

            Cluster cluster = clusters.cluster(probes[i]);
            if (!worthScanning(cluster, filterActive, selectivity)) {
                continue;
            }
            scanCluster(scan, cluster, wanted, visited, collector);
            scanned++;
        }
        return scanned;
    }

    /**
     * Whether a cluster is worth touching at all: never if it is empty, and under a selective filter not if
     * its expected match count is negligible. Decided from the {@link Cluster} alone, so it costs no I/O and
     * the same test gates both scanning and hinting.
     */
    private static boolean worthScanning(Cluster cluster, boolean filterActive, float selectivity) {
        if (cluster.size() == 0) {
            return false;
        }
        return !filterActive || cluster.size() * selectivity >= MIN_EXPECTED_MATCHES;
    }

    /** Scan one cluster's postings into the collector; collects by ordinal (collector maps ord→doc). */
    private static void scanCluster(
        ClusterScan scan,
        Cluster cluster,
        Bits wanted,
        BitSet visited,
        KnnCollector collector
    ) throws IOException {
        PostingScorer scorer = scan.scorer(cluster, wanted);
        while (scorer.advance(collector.minCompetitiveSimilarity())) {
            int ord = scorer.ord();          // wanted: passed the filter and unvisited (checked pre-scoring)
            visited.set(ord);
            collector.incVisitedCount(1);      // a distance was actually computed for this ord
            collector.collect(ord, scorer.score()); // collector maps ord → doc
            // Termination hooks (visit-limit / contribution / shared-threshold) deferred.
        }
    }
}
