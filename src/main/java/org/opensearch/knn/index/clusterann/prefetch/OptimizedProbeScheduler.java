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

/**
 * Wraps a {@link NearestProbeScheduler} to reorder probes by file offset
 * within sliding windows, and issue read-ahead prefetch for upcoming probes.
 *
 * <p>Also enforces a scoring budget: stops probing when enough vectors
 * have been scored and competitive results are found.
 */
public final class OptimizedProbeScheduler implements ProbeScheduler {

    private static final int WINDOW_SIZE = 4;
    private static final int LOOKAHEAD = 4;

    private final ProbeTarget[] probes;
    private final int nprobe;
    private final ClusterANNCentroidScanner scanner;
    private final IndexInput postingsInput;
    private final int[] centroidDocCounts;
    private final int numVectors;
    private final int k;
    private final long filterCost;

    public OptimizedProbeScheduler(
        NearestProbeScheduler nearest,
        ClusterANNCentroidScanner scanner,
        IndexInput postingsInput,
        int[] centroidDocCounts,
        int numVectors,
        int k,
        long filterCost
    ) {
        this.probes = nearest.probes().clone();
        this.nprobe = nearest.nprobe();
        this.scanner = scanner;
        this.postingsInput = postingsInput;
        this.centroidDocCounts = centroidDocCounts;
        this.numVectors = numVectors;
        this.k = k;
        this.filterCost = filterCost;
    }

    @Override
    public int execute(KnnCollector collector) throws IOException {
        reorderByOffset(probes, nprobe, WINDOW_SIZE);

        double logN = Math.log10(Math.max(numVectors, 10));
        long budget = Math.max(k * 4L, (long) (2.0 * logN * logN * k));
        float filterSelectivity = numVectors > 0 ? (float) filterCost / numVectors : 1.0f;
        boolean filterActive = filterSelectivity < 0.10f && filterSelectivity > 0;

        // Prefetch initial window
        int prefetchedUpTo = Math.min(LOOKAHEAD, nprobe - 1);
        for (int i = 0; i <= prefetchedUpTo; i++) {
            issueReadAhead(probes[i]);
        }

        long docsExpected = 0;
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

            if (docsScored >= k && docsExpected >= budget && collector.minCompetitiveSimilarity() != Float.NEGATIVE_INFINITY) {
                break;
            }

            // Prefetch next probes
            while (prefetchedUpTo + 1 < nprobe && prefetchedUpTo < i + LOOKAHEAD) {
                prefetchedUpTo++;
                issueReadAhead(probes[prefetchedUpTo]);
            }

            scanner.prepare(probe);
            int scored = scanner.scan(collector);
            docsExpected += centroidDocCounts[probe.centroidIdx()];
            docsScored += scored;
            totalScored += scored;

            if (collector.earlyTerminated()) break;
        }

        return totalScored;
    }

    private static final int L2_CACHE_THRESHOLD = 256 * 1024;

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
