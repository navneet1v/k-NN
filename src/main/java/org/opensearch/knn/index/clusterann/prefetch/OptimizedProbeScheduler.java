/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.clusterann.codec.ClusterANNCentroidScanner;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Wraps a {@link NearestProbeScheduler} to reorder probes by file offset
 * within sliding windows, and issue read-ahead prefetch for upcoming probes.
 *
 * <p>Also enforces a scoring budget: stops probing when enough vectors
 * have been scored and competitive results are found.
 */
public final class OptimizedProbeScheduler implements ProbeScheduler {

    private static final int WINDOW_SIZE = 8;
    private static final int LOOKAHEAD = 8;

    private final ProbeTarget[] probes;
    private final int nprobe;
    private final ClusterANNCentroidScanner scanner;
    private final IndexInput postingsInput;
    private final int[] centroidDocCounts;
    private final int numVectors;
    private final int k;
    private final long filterCost;
    private final int[] filterMatchCounts;

    public OptimizedProbeScheduler(
        NearestProbeScheduler nearest,
        ClusterANNCentroidScanner scanner,
        IndexInput postingsInput,
        int[] centroidDocCounts,
        int numVectors,
        int k,
        long filterCost
    ) {
        this(nearest, scanner, postingsInput, centroidDocCounts, numVectors, k, filterCost, null);
    }

    public OptimizedProbeScheduler(
        NearestProbeScheduler nearest,
        ClusterANNCentroidScanner scanner,
        IndexInput postingsInput,
        int[] centroidDocCounts,
        int numVectors,
        int k,
        long filterCost,
        int[] filterMatchCounts
    ) {
        this.probes = nearest.probes().clone();
        this.nprobe = nearest.nprobe();
        this.scanner = scanner;
        this.postingsInput = postingsInput;
        this.centroidDocCounts = centroidDocCounts;
        this.numVectors = numVectors;
        this.k = k;
        this.filterCost = filterCost;
        this.filterMatchCounts = filterMatchCounts;
    }

    /** Per-query I/O bytes counter (ADC scan portion). */
    private static final ThreadLocal<long[]> QUERY_BYTES = ThreadLocal.withInitial(() -> new long[1]);

    /** Cross-segment score sharing: segments in same shard share this threshold. */
    private static final AtomicLong SHARED_MIN_SCORE = new AtomicLong(Float.floatToIntBits(Float.NEGATIVE_INFINITY));

    /** Reset shared threshold at start of each query. */
    public static void resetSharedThreshold() {
        SHARED_MIN_SCORE.set(Float.floatToIntBits(Float.NEGATIVE_INFINITY));
    }

    /** Get shared min competitive score across all segments. */
    public static float getSharedThreshold() {
        return Float.intBitsToFloat((int) SHARED_MIN_SCORE.get());
    }

    /** Update shared threshold if this segment found better results. */
    public static void updateSharedThreshold(float score) {
        long newBits = Float.floatToIntBits(score);
        SHARED_MIN_SCORE.accumulateAndGet(newBits, (current, update) ->
            Float.intBitsToFloat((int) update) > Float.intBitsToFloat((int) current) ? update : current
        );
    }

    /** Get bytes read for ADC scan in the last query on this thread. */
    public static long getLastQueryAdcBytes() { return QUERY_BYTES.get()[0]; }
    public static void resetQueryAdcBytes() { QUERY_BYTES.get()[0] = 0; }
    public static void addActualBytes(long bytes) { QUERY_BYTES.get()[0] += bytes; }

    @Override
    public int execute(KnnCollector collector) throws IOException {
        
        reorderByOffset(probes, nprobe, WINDOW_SIZE);

        // Hybrid termination: soft budget + contribution-based override
        // Budget = expected vectors in probed clusters (nprobe/numCentroids of segment)
        // Soft budget = half the expected vectors in probed clusters

        float filterSelectivity = numVectors > 0 ? (float) filterCost / numVectors : 1.0f;
        boolean filterActive = filterSelectivity < 0.10f && filterSelectivity > 0;

        // Contribution-based early termination
        int consecutiveEmpty = 0;

        // Prefetch initial window
        int prefetchedUpTo = Math.min(LOOKAHEAD, nprobe - 1);
        for (int i = 0; i <= prefetchedUpTo; i++) {
            issueReadAhead(probes[i]);
        }

        
        long docsScored = 0;
        int totalScored = 0;

        for (int i = 0; i < nprobe; i++) {
            ProbeTarget probe = probes[i];

            if (filterActive) {
                int docCount = centroidDocCounts[probe.centroidIdx()];
                if (docCount * filterSelectivity < 0.5f) {
                    if (prefetchedUpTo + 1 < nprobe) {
                        prefetchedUpTo++;
                        issueReadAhead(probes[prefetchedUpTo]);
                    }
                    continue;
                }
            }

            // Prefetch next probes
            while (prefetchedUpTo + 1 < nprobe && prefetchedUpTo < i + LOOKAHEAD) {
                prefetchedUpTo++;
                issueReadAhead(probes[prefetchedUpTo]);
            }

            float thresholdBefore = collector.minCompetitiveSimilarity();
            // Adaptive scoring precision: exact score sparse clusters, ADC for dense ones
            if (filterMatchCounts != null) {
                int matches = filterMatchCounts[probe.centroidIdx()];
                scanner.setForceExact(matches < k);
            }
            scanner.prepare(probe);
            int scored = scanner.scan(collector);
            
            docsScored += scored;
            totalScored += scored;

            // Cross-segment score sharing: update shared threshold after each cluster
            float currentMin = collector.minCompetitiveSimilarity();
            if (currentMin > Float.NEGATIVE_INFINITY) {
                updateSharedThreshold(currentMin);
            }

            if (collector.earlyTerminated()) break;

            // Filter-aware early termination: stop once we've accumulated enough filter matches
            if (filterMatchCounts != null && docsScored >= k * 3
                    && collector.minCompetitiveSimilarity() != Float.NEGATIVE_INFINITY) {
                break;
            }

            // Contribution-based termination: stop when clusters stop helping
            if (i >= 2 && docsScored >= k * 3) {
                float thresholdAfter = collector.minCompetitiveSimilarity();
                // Also consider shared threshold from other segments
                float shared = getSharedThreshold();
                if (shared > thresholdAfter) thresholdAfter = shared;

                boolean improving = thresholdAfter > thresholdBefore && thresholdBefore != Float.NEGATIVE_INFINITY;
                if (improving) {
                    consecutiveEmpty = 0;
                } else if (thresholdBefore != Float.NEGATIVE_INFINITY || shared > Float.NEGATIVE_INFINITY) {
                    consecutiveEmpty++;
                }

                if (consecutiveEmpty >= 2 && thresholdAfter != Float.NEGATIVE_INFINITY) {
                    break;
                }
            }
        }

        return totalScored;
    }

    private static final int L2_CACHE_THRESHOLD = 2 * 1024 * 1024;

    private void issueReadAhead(ProbeTarget probe) throws IOException {
        long offset = probe.fileOffset();
        long len = probe.postingBytes();
        if (len <= 0 || offset < 0 || offset + len > postingsInput.length()) return;
        // Skip prefetch for oversized postings that would thrash L2 cache
        if (len > L2_CACHE_THRESHOLD) return;
        postingsInput.prefetch(offset, len);
    }

    private static void reorderByOffset(ProbeTarget[] probes, int count, int windowSize) {
        for (int start = 0; start < count; start += windowSize) {
            int end = Math.min(start + windowSize, count);
            Arrays.sort(probes, start, end, (a, b) -> Long.compare(a.fileOffset(), b.fileOffset()));
        }
    }
}
