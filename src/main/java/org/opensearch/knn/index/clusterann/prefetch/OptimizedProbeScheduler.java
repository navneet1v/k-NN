/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.clusterann.codec.ClusterANNCentroidScanner;
import org.opensearch.knn.index.clusterann.codec.ClipPruningData;

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

    private static final int WINDOW_SIZE = 8;
    private static final int LOOKAHEAD = 8;

    // Instrumentation for measuring effective segments
    private static final java.util.concurrent.atomic.AtomicInteger TOTAL_SEGMENTS_SEARCHED = new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger TOTAL_CLUSTERS_PROBED = new java.util.concurrent.atomic.AtomicInteger();
    private static final ThreadLocal<Integer> LAST_CLUSTERS_PROBED = ThreadLocal.withInitial(() -> 0);
    private static final ThreadLocal<Integer> LAST_NPROBE_PLANNED = ThreadLocal.withInitial(() -> 0);

    /** Get and reset query-level stats. Returns [segmentsSearched, totalClustersProbed]. */
    public static int[] getAndResetStats() {
        int segs = TOTAL_SEGMENTS_SEARCHED.getAndSet(0);
        int clusters = TOTAL_CLUSTERS_PROBED.getAndSet(0);
        return new int[]{segs, clusters};
    }

    /** Get clusters probed in the last segment search on this thread. */
    public static int lastClustersProbed() {
        return LAST_CLUSTERS_PROBED.get();
    }

    private final ProbeTarget[] probes;
    private final int nprobe;
    private final ClusterANNCentroidScanner scanner;
    private final IndexInput postingsInput;
    private final int[] centroidDocCounts;
    private final float[] centroidNorms;
    private final int numVectors;
    private final int k;
    private final long filterCost;
    private final int[] filterMatchCounts;
    private final ClipPruningData clipData;

    public OptimizedProbeScheduler(
        NearestProbeScheduler nearest,
        ClusterANNCentroidScanner scanner,
        IndexInput postingsInput,
        int[] centroidDocCounts,
        int numVectors,
        int k,
        long filterCost
    ) {
        this(nearest, scanner, postingsInput, centroidDocCounts, null, numVectors, k, filterCost, null, null);
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
        this(nearest, scanner, postingsInput, centroidDocCounts, null, numVectors, k, filterCost, filterMatchCounts, null);
    }

    public OptimizedProbeScheduler(
        NearestProbeScheduler nearest,
        ClusterANNCentroidScanner scanner,
        IndexInput postingsInput,
        int[] centroidDocCounts,
        int numVectors,
        int k,
        long filterCost,
        int[] filterMatchCounts,
        ClipPruningData clipData
    ) {
        this(nearest, scanner, postingsInput, centroidDocCounts, null, numVectors, k, filterCost, filterMatchCounts, clipData);
    }

    public OptimizedProbeScheduler(
        NearestProbeScheduler nearest,
        ClusterANNCentroidScanner scanner,
        IndexInput postingsInput,
        int[] centroidDocCounts,
        float[] centroidNorms,
        int numVectors,
        int k,
        long filterCost,
        int[] filterMatchCounts,
        ClipPruningData clipData
    ) {
        this.probes = nearest.probes().clone();
        this.nprobe = nearest.nprobe();
        this.scanner = scanner;
        this.postingsInput = postingsInput;
        this.centroidDocCounts = centroidDocCounts;
        this.centroidNorms = centroidNorms;
        this.numVectors = numVectors;
        this.k = k;
        this.filterCost = filterCost;
        this.filterMatchCounts = filterMatchCounts;
        this.clipData = clipData;
    }

    /** Per-query I/O bytes counter (ADC scan portion). */
    private static final ThreadLocal<long[]> QUERY_BYTES = ThreadLocal.withInitial(() -> new long[1]);

    /** Cross-segment score sharing: per-query, keyed by collector identity.
     *  Same KnnCollector instance = same query = share threshold.
     *  Different collectors = different queries = independent thresholds.
     *  Thread-safe via volatile (threshold only goes up). */
    private static final java.util.Map<KnnCollector, float[]> COLLECTOR_THRESHOLDS =
        java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /** Reset shared threshold for a new query (called before segment iteration). */
    public static void resetSharedThreshold() {
        // No-op: threshold is created per collector on first access
    }

    /** Initialize or get threshold for this collector (query). */
    private static float[] getOrCreateThreshold(KnnCollector collector) {
        return COLLECTOR_THRESHOLDS.computeIfAbsent(collector, k -> new float[]{Float.NEGATIVE_INFINITY});
    }

    /** Get shared min competitive score for this query's collector. */
    public static float getSharedThreshold(KnnCollector collector) {
        float[] t = COLLECTOR_THRESHOLDS.get(collector);
        return t != null ? t[0] : Float.NEGATIVE_INFINITY;
    }

    /** Update shared threshold if this segment found better results. */
    public static void updateSharedThreshold(KnnCollector collector, float score) {
        float[] t = getOrCreateThreshold(collector);
        if (score > t[0]) {
            t[0] = score;
        }
    }

    /** Get bytes read for ADC scan in the last query on this thread. */
    public static long getLastQueryAdcBytes() {
        return QUERY_BYTES.get()[0];
    }

    public static void resetQueryAdcBytes() {
        QUERY_BYTES.get()[0] = 0;
    }

    public static void addActualBytes(long bytes) {
        QUERY_BYTES.get()[0] += bytes;
    }

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
        int clustersActuallyProbed = 0;

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

            // CLIP inter-cluster pruning: provably skip clusters whose upper bound
            // on max IP (or lower bound on min L2²) cannot beat the current threshold.
            // Falls back to distance-budget heuristic when CLIP data is unavailable.
            // Only apply after scoring at least one cluster in this segment.
            if (clustersActuallyProbed >= 1 && docsScored >= k
                && collector.minCompetitiveSimilarity() > Float.NEGATIVE_INFINITY) {
                float threshold = collector.minCompetitiveSimilarity();
                // Also consider shared threshold from other segments
                float shared = getSharedThreshold(collector);
                if (shared > threshold) threshold = shared;

                if (clipData != null) {
                    // Convert stored centroidDist to L2² for CLIP formula.
                    // NearestProbeScheduler stores:
                    //   L2 metric: ‖q-c‖² directly (= ‖q‖² + ‖c‖² - 2·dot)
                    //   IP metric: -dot(q,c)
                    // For IP with non-unit centroids:
                    //   ‖q-c‖² = ‖q‖² + ‖c‖² - 2·dot = 1 + ‖c‖² + 2·rawDist
                    //   (since ‖q‖²=1 for normalized queries, rawDist=-dot)
                    float rawDist = probe.centroidDist();
                    float distQC_sq;
                    if (centroidNorms != null && rawDist <= 0) {
                        // IP metric: use centroid norm for exact conversion
                        float cNormSq = centroidNorms[probe.centroidIdx()];
                        distQC_sq = 1.0f + cNormSq + 2.0f * rawDist;
                    } else if (rawDist <= 0) {
                        // IP without norms: approximate with unit centroids
                        distQC_sq = 2.0f + 2.0f * rawDist;
                    } else {
                        // L2: already ‖q-c‖²
                        distQC_sq = rawDist;
                    }
                    if (distQC_sq < 0) distQC_sq = 0; // numerical safety

                    float ubIP = clipData.upperBoundIP(probe.centroidIdx(), distQC_sq);
                    // Transform raw IP upper bound to Lucene similarity score space.
                    // MAXIMUM_INNER_PRODUCT: score = dot >= 0 ? dot+1 : 1/(1-dot)
                    // DOT_PRODUCT (cosine): score = (1+dot)/2
                    // EUCLIDEAN: score = 1/(1+l2²) — for L2 we'd use lowerBoundL2Sq instead
                    float ubScore;
                    if (ubIP >= 0) {
                        ubScore = ubIP + 1.0f; // MAXIMUM_INNER_PRODUCT transform
                    } else {
                        ubScore = 1.0f / (1.0f - ubIP);
                    }
                    if (ubScore < threshold) {
                        if (prefetchedUpTo + 1 < nprobe) {
                            prefetchedUpTo++;
                            issueReadAhead(probes[prefetchedUpTo]);
                        }
                        continue;
                    }
                } else if (i >= 3 && docsScored >= k * 2
                    && probe.centroidDist() > probes[0].centroidDist() * 4.0f) {
                    // Fallback heuristic: skip clusters 4x farther than closest
                    if (prefetchedUpTo + 1 < nprobe) {
                        prefetchedUpTo++;
                        issueReadAhead(probes[prefetchedUpTo]);
                    }
                    consecutiveEmpty++;
                    if (consecutiveEmpty >= 2) break;
                    continue;
                }
            }

            scanner.prepare(probe);
            int scored = scanner.scan(collector);
            clustersActuallyProbed++;

            docsScored += scored;
            totalScored += scored;

            // Cross-segment score sharing: update shared threshold after each cluster
            float currentMin = collector.minCompetitiveSimilarity();
            if (currentMin > Float.NEGATIVE_INFINITY) {
                updateSharedThreshold(collector, currentMin);
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
                float shared = getSharedThreshold(collector);
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

        // Instrumentation: track clusters probed per segment
        LAST_CLUSTERS_PROBED.set(clustersActuallyProbed);
        LAST_NPROBE_PLANNED.set(nprobe);
        TOTAL_SEGMENTS_SEARCHED.incrementAndGet();
        TOTAL_CLUSTERS_PROBED.addAndGet(clustersActuallyProbed);

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
