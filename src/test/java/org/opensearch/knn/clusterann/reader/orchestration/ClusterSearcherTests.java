/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.orchestration;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.reader.Cluster;
import org.opensearch.knn.clusterann.reader.Clusters;
import org.opensearch.knn.clusterann.reader.PostingScorer;
import org.opensearch.knn.clusterann.reader.ScanParams;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Fast, isolated unit tests for {@link ClusterSearcher} (JUnit 5 + Mockito).
 *
 * <p>The searcher holds the walk's own logic — probe order, SOAR dedup, skipping empties, feeding the collector's
 * threshold back into each step — and nothing else, so the
 * clusters underneath it are stubs that hand out fixed ordinals. That keeps these tests about the walk rather than
 * about quantization or block layout, which are covered where they live.
 */
class ClusterSearcherTests {

    private static final float[] QUERY = { 0.1f, 0.2f, 0.3f, 0.4f };

    // ---------------------------------------------------------------- the walk

    /** Every probed cluster is scanned, and every ordinal it yields reaches the collector. */
    @Test
    void testSearch_thenScansEveryProbedClusterInOrder() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 11 }, 1, new int[] { 20 }, 2, new int[] { 30, 31 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 0, 1, 2 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(3, scanned);
        assertEquals(List.of(10, 11, 20, 30, 31), collector.collected);
    }

    /** The probe order is the caller's, and the walk must not reorder it — pruning depends on closest-first. */
    @Test
    void testSearch_thenVisitsProbesInTheOrderGiven() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10 }, 1, new int[] { 20 }, 2, new int[] { 30 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        ClusterSearcher.search(clusters, new int[] { 2, 0, 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(List.of(30, 10, 20), collector.collected);
    }

    /** A probe list need not cover the field, and may name a cluster twice. */
    @Test
    void testSearch_whenOnlySomeClustersAreProbed_thenTheRestAreUntouched() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10 }, 1, new int[] { 20 }, 2, new int[] { 30 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(1, scanned);
        assertEquals(List.of(20), collector.collected);
    }

    @Test
    void testSearch_whenThereAreNoProbes_thenScansNothing() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        int scanned = ClusterSearcher.search(clusters, new int[0], ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(0, scanned);
        assertTrue(collector.collected.isEmpty());
    }

    @Test
    void testSearch_whenTheFieldHasNoClusters_thenScansNothing() throws IOException {
        // given
        Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(0);
        RecordingCollector collector = new RecordingCollector();

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 0, 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(0, scanned, "a field with no clusters cannot be walked at all");
        assertTrue(collector.collected.isEmpty());
    }

    /** An empty cluster is skipped and not counted — the decision costs no I/O, since obtaining it read nothing. */
    @Test
    void testSearch_whenAClusterIsEmpty_thenSkipsItWithoutCountingIt() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10 }, 1, new int[0], 2, new int[] { 30 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 0, 1, 2 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(2, scanned, "the empty cluster is not among the scanned");
        assertEquals(List.of(10, 30), collector.collected);
    }

    // ---------------------------------------------------------------- dedup and filtering

    /**
     * SOAR puts a boundary vector in two clusters, so a walk can reach the same ordinal twice — and must collect it
     * once. The dedup is the walk's own, invisible to the clusters, which see only one membership test.
     */
    @Test
    void testSearch_whenSoarPutsAnOrdinalInTwoClusters_thenCollectsItOnce() throws IOException {
        // given — ordinal 20 is in both clusters, as a SOAR copy
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 20 }, 1, new int[] { 20, 30 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        ClusterSearcher.search(clusters, new int[] { 0, 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(List.of(10, 20, 30), collector.collected, "the second sighting of 20 is dropped");
    }

    /** The caller's filter is honoured, composed with the dedup into the one test a cluster applies. */
    @Test
    void testSearch_whenOrdinalsAreFiltered_thenOnlyAcceptedOnesAreCollected() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 11, 12 }));
        FixedBitSet accepted = new FixedBitSet(64);
        accepted.set(10);
        accepted.set(12);
        RecordingCollector collector = new RecordingCollector();

        // when
        ClusterSearcher.search(clusters, new int[] { 0 }, ScanParams.of(QUERY), collector, accepted);

        // then
        assertEquals(List.of(10, 12), collector.collected);
    }

    /** A filter that accepts nothing still leaves the cluster scanned — it was visited, it just yielded nothing. */
    @Test
    void testSearch_whenTheFilterAcceptsNothing_thenCollectsNothing() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 11 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 0 }, ScanParams.of(QUERY), collector, new FixedBitSet(64));

        // then
        assertEquals(1, scanned);
        assertTrue(collector.collected.isEmpty());
    }

    // ---------------------------------------------------------------- collector contract

    /** One visit counted per distance actually computed, which is what a visit limit is measured against. */
    @Test
    void testSearch_thenCountsOneVisitPerScoredVector() throws IOException {
        // given
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 11 }, 1, new int[] { 20 }));
        RecordingCollector collector = new RecordingCollector();

        // when
        ClusterSearcher.search(clusters, new int[] { 0, 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(3, collector.visited);
    }

    /**
     * The collector's threshold is read on every step rather than once per cluster, so a threshold that rises mid-walk
     * prunes the rest of the walk too — the tail of the cluster in hand included. Reading it once per cluster would let
     * the second cluster run to its end on the stale threshold it was entered with.
     */
    @Test
    void testSearch_whenTheThresholdRisesMidCluster_thenPrunesTheRestOfThatCluster() throws IOException {
        // given — the stub scores by position, so a cluster's ordinals score 1.0, 0.5, 0.33 in turn
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 11 }, 1, new int[] { 20, 21, 22 }));
        // the third hit lifts the threshold past 0.5, which lands mid-way through the second cluster
        RecordingCollector collector = new RecordingCollector(3, 0.75f);

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 0, 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(2, scanned, "both clusters were still entered");
        assertEquals(List.of(10, 11, 20), collector.collected, "21 and 22 fall under the threshold 20 raised");
        assertEquals(3, collector.visited, "a pruned posting costs no distance computation");
        // one per advance() that returned a hit, plus one per advance() that ended a cluster
        assertEquals(collector.collected.size() + scanned, collector.thresholdReads, "the threshold is consulted per step");
    }

    /** Once nothing left can compete, a later probe is entered but nothing in it is scored. */
    @Test
    void testSearch_whenNothingCanCompete_thenLaterProbesAreNotScored() throws IOException {
        // given — a threshold above every score the stub can produce, raised by the very first hit
        Clusters clusters = clusters(Map.of(0, new int[] { 10, 11, 12 }, 1, new int[] { 20, 21 }));
        RecordingCollector collector = new RecordingCollector(1, 2f);

        // when
        int scanned = ClusterSearcher.search(clusters, new int[] { 0, 1 }, ScanParams.of(QUERY), collector, null);

        // then
        assertEquals(2, scanned, "the second cluster is still counted as scanned — it was visited");
        assertEquals(List.of(10), collector.collected, "the walk stops scoring the moment nothing can compete");
        assertEquals(1, collector.visited);
    }

    // ---------------------------------------------------------------- helpers

    /**
     * Clusters whose postings are fixed ordinal lists. Built with Mockito because {@link Clusters} is final and owns
     * file inputs; what matters to these tests is only the ordinals it yields.
     */
    private static Clusters clusters(Map<Integer, int[]> postings) throws IOException {
        Map<Integer, int[]> ordered = new LinkedHashMap<>(postings);
        Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(ordered.size());
        when(clusters.numVectors()).thenReturn(1024);

        for (Map.Entry<Integer, int[]> entry : ordered.entrySet()) {
            Cluster cluster = mock(Cluster.class);
            when(cluster.ordinal()).thenReturn(entry.getKey());
            when(cluster.size()).thenReturn(entry.getValue().length);
            when(clusters.get(entry.getKey())).thenReturn(cluster);
        }

        // The scan resolves each cluster to a scorer over its own ordinals, applying the membership test the walk
        // composed — which is how the dedup and the filter reach the posting.
        when(clusters.scan()).thenReturn((cluster, scanParams, acceptedOrds) -> {
            int[] ordinals = ordered.get(cluster.ordinal());
            return new StubScorer(ordinals, acceptedOrds);
        });
        return clusters;
    }

    /**
     * Hands out the ordinals the filter accepts, scoring them by position so the scores differ — and, since they fall
     * as the position grows, stopping outright at the first one under the threshold it was handed. That standing-in for
     * a real scorer's early termination is what lets these tests observe pruning at all.
     */
    private static final class StubScorer implements PostingScorer {

        private final int[] ordinals;
        private final Bits acceptedOrds;
        private int position = -1;

        private StubScorer(int[] ordinals, Bits acceptedOrds) {
            this.ordinals = ordinals;
            this.acceptedOrds = acceptedOrds;
        }

        @Override
        public boolean advance(float minCompetitiveSimilarity) {
            while (++position < ordinals.length) {
                if (score() < minCompetitiveSimilarity) {
                    return false; // scores only fall from here, so the remainder of this cluster is hopeless
                }
                if (acceptedOrds == null || acceptedOrds.get(ordinals[position])) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public int ord() {
            return ordinals[position];
        }

        @Override
        public float score() {
            return 1f / (1f + position);
        }
    }

    /**
     * Records what the walk collected, in order, plus the bookkeeping calls it made. Its threshold is flat by default,
     * so nothing prunes; {@link #RecordingCollector(int, float)} makes it rise once a given number of hits are in, which
     * is how a real collector behaves once its queue fills.
     */
    private static final class RecordingCollector implements KnnCollector {

        private final List<Integer> collected = new ArrayList<>();
        private final int riseAfter;
        private final float risenTo;
        private int visited;
        private int thresholdReads;

        private RecordingCollector() {
            this(Integer.MAX_VALUE, Float.NEGATIVE_INFINITY);
        }

        private RecordingCollector(int riseAfter, float risenTo) {
            this.riseAfter = riseAfter;
            this.risenTo = risenTo;
        }

        @Override
        public boolean earlyTerminated() {
            return false;
        }

        @Override
        public void incVisitedCount(int count) {
            visited += count;
        }

        @Override
        public long visitedCount() {
            return visited;
        }

        @Override
        public long visitLimit() {
            return Long.MAX_VALUE;
        }

        @Override
        public int k() {
            return 10;
        }

        @Override
        public boolean collect(int docId, float similarity) {
            collected.add(docId);
            return true;
        }

        @Override
        public float minCompetitiveSimilarity() {
            thresholdReads++;
            return collected.size() < riseAfter ? Float.NEGATIVE_INFINITY : risenTo;
        }

        @Override
        public org.apache.lucene.search.TopDocs topDocs() {
            throw new UnsupportedOperationException("not needed by these tests");
        }

        @Override
        public org.apache.lucene.search.knn.KnnSearchStrategy getSearchStrategy() {
            return null; // the walk never consults a strategy
        }
    }
}
