/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.extern.log4j.Log4j2;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-query instrumentation that measures <b>read amplification</b> for a single vector search — the
 * bytes the device faults in (whole 32&nbsp;KB pages) versus the bytes actually needed (the visited
 * vectors' records).
 *
 * <p>For one query (one segment traversal) it accumulates the set of distinct physical pages touched and
 * the set of distinct vectors read, then logs {@code dataReadBytes (X) = distinctPages*32KB},
 * {@code dataUsedBytes (Y) = distinctVectors*recordSize}, and {@code readAmplification = X/Y}. Comparing
 * X/Y between a locality-reordered index and a baseline measures whether reordering actually cuts wasted
 * I/O — the distinct-page count is the runtime analogue of {@code compute_pages_touched} in
 * {@code scripts/locality_simulation.py} (the query-UNION of pages), which the per-call grouping count in
 * {@link PrefetchHelper} cannot show.
 *
 * <p><b>How it is fed.</b> {@link #recordBatch} is called from {@link PrefetchHelper#prefetch} with the
 * <em>physical</em> ordinals about to be read (both the reordered {@code .veqlo} path and the baseline SQ
 * path route through there with {@code baseOffset == 0}), so a page is
 * {@code (ord * recordSize) / 32768}. The reordered store uses 112-byte records and the baseline 96-byte
 * records, so {@code recordSize} is logged too and lets you tell the two runs apart.
 *
 * <p><b>Scope / lifecycle.</b> One instance per thread ({@link ThreadLocal}). A segment search brackets
 * its traversal with {@link #begin()} / {@link #end(String)} (see
 * {@code FaissMemoryOptimizedSearcher}). After a force-merge to one segment this is one bracket per user
 * query; pre-merge it is one bracket per segment (log lines are per-segment).
 *
 * <p><b>Enabling / cost.</b> Off by default. The easy runtime toggle (no restart, reaches the node) is to
 * set this class's logger to {@code DEBUG} via the cluster settings API:
 * <pre>
 * PUT _cluster/settings
 * {"persistent":{"logger.org.opensearch.knn.index.codec.scorer.PageTouchTracker":"DEBUG"}}
 * </pre>
 * A {@code -Dknn.pageTouch.enabled=true} JVM arg is also honored if you can plumb it to the node JVM, but
 * a bare {@code -D} on the gradle command line reaches only the gradle JVM, not the forked OpenSearch node.
 * The gate is re-read per query in {@link #begin()}, so toggling the level takes effect on the next query.
 * When off every method is a cheap no-op. The per-query summary itself is logged at INFO so it stays
 * visible once tracking is on. Single-vector {@code score()} reads (e.g. isolated entry-point scoring) do
 * not pass through {@link PrefetchHelper} and are therefore not counted; the neighbor-list bulk reads that
 * dominate a traversal are.
 */
@Log4j2
public final class PageTouchTracker {

    // Two page granularities to compare read amplification at. 32 KB matches PrefetchHelper's grouping
    // window; 8 KB models a finer OS/device page (less wasted data per fault, but more distinct faults).
    private static final long PAGE_BYTES_32K = 32L * 1024;
    private static final long PAGE_BYTES_8K = 8L * 1024;

    /**
     * Optional JVM-arg override, read once at class load. Reaching the OpenSearch node JVM with a {@code -D}
     * flag is fiddly (it must be plumbed through testClusters, not the gradle command line), so the primary
     * gate is the logger level below; this is only honored if it happens to be set.
     */
    private static final boolean ENABLED_BY_PROP = readEnabledProp();

    /** Monotonic query id across all threads, purely so log lines can be correlated. */
    private static final AtomicLong QUERY_SEQ = new AtomicLong();

    private static final ThreadLocal<PageTouchTracker> THREAD_LOCAL = ThreadLocal.withInitial(PageTouchTracker::new);

    // Registry of per-(segment,field) reordering permutations (ordToPhysicalOrdMap), published by the
    // locality reader when it loads .vemlo. Lets the rescore path compute a "shadow" reordered-.vec
    // read amplification — what the .vec reads WOULD touch if .vec were reordered by the same map —
    // without actually reordering the file. Instrumentation-only; entries are bounded (per field/segment).
    private static final Map<String, int[]> PERMUTATIONS = new ConcurrentHashMap<>();

    // Distinct pages touched at each granularity (a record straddling a boundary counts both pages).
    private final Set<Long> pages32 = new HashSet<>();
    private final Set<Long> pages8 = new HashSet<>();
    // Shadow: pages that would be touched if reads went to a .vec REORDERED by the permutation.
    private final Set<Long> pages32Shadow = new HashSet<>();
    private final Set<Long> pages8Shadow = new HashSet<>();
    private int[] shadowPermutation;   // ordToPhysicalOrdMap for the shadow stream; null = no shadow
    // Distinct physical ordinals actually read this query (deduped across neighbor lists). This is the
    // "useful" set — a vector only needs to be paged in once — and is the denominator for read amplification.
    private final Set<Integer> distinctOrds = new HashSet<>();
    private long ordsScored;
    private long prefetchCalls;
    private long recordSize;
    private boolean active;
    // Fixed record stride for the direct-read (full-precision rescore) bracket; 0 in the prefetch bracket.
    private long directRecordSize;

    private PageTouchTracker() {}

    private static boolean readEnabledProp() {
        try {
            return Boolean.parseBoolean(System.getProperty("knn.pageTouch.enabled", "false"));
        } catch (final SecurityException e) {
            return false;
        }
    }

    /** Returns the calling thread's tracker. */
    public static PageTouchTracker get() {
        return THREAD_LOCAL.get();
    }

    /** Publishes a field's reordering permutation (ordToPhysicalOrdMap) so the rescore path can compute
     *  the shadow reordered-.vec amplification. Called by the locality reader on load. */
    public static void registerPermutation(final String segment, final String field, final int[] ordToPhysical) {
        PERMUTATIONS.put(segment + "|" + field, ordToPhysical);
    }

    /** Looks up the reordering permutation for a (segment, field), or null if none. */
    public static int[] permutationFor(final String segment, final String field) {
        return PERMUTATIONS.get(segment + "|" + field);
    }

    /**
     * Whether tracking is currently on. Enabled either by {@code -Dknn.pageTouch.enabled=true} OR — the easy
     * runtime toggle, no restart — by setting this class's logger to {@code DEBUG}:
     * <pre>
     * PUT _cluster/settings
     * {"persistent":{"logger.org.opensearch.knn.index.codec.scorer.PageTouchTracker":"DEBUG"}}
     * </pre>
     */
    public static boolean isEnabled() {
        return ENABLED_BY_PROP || log.isDebugEnabled();
    }

    /** Starts accounting for one per-segment query traversal, clearing any prior state. */
    public void begin() {
        active = isEnabled();
        if (!active) {
            return;
        }
        pages32.clear();
        pages8.clear();
        pages32Shadow.clear();
        pages8Shadow.clear();
        distinctOrds.clear();
        ordsScored = 0;
        prefetchCalls = 0;
        recordSize = 0;
        directRecordSize = 0;
        shadowPermutation = null;
    }

    /**
     * Starts accounting for a direct-read (non-prefetch) phase — the full-precision rescore reads in
     * {@code ExactSearcher}, which read {@code .vec} vectors one at a time via {@code vectorValue(docId)}
     * with no prefetch. Sets the fixed record stride so {@link #recordOrd(int)} can account each read.
     * This is a separate bracket from the quantized ANN {@link #begin()} window, so it logs its own line
     * with its own {@code recordSize} (e.g. dim*4 for float) — the script groups lines by record size.
     *
     * @param recordBytes on-disk stride of one full-precision record (e.g. dimension * 4 for float32)
     * @param shadowPerm  reordering permutation (ordToPhysicalOrdMap) to also account a hypothetical
     *                    reordered-.vec stream, or null for none
     */
    public void beginFullPrecision(final long recordBytes, final int[] shadowPerm) {
        begin();
        if (active) {
            directRecordSize = recordBytes;
            shadowPermutation = shadowPerm;
        }
    }

    /**
     * Records one direct full-precision read at {@code ord} (the {@code .vec} ordinal == docId, since the
     * raw store is kept in doc-id order). No-op unless a {@link #beginFullPrecision(long)} bracket is
     * active, so it is safe to call unconditionally from the exact-search score loop.
     */
    public void recordOrd(final int ord) {
        if (!active || directRecordSize <= 0) {
            return;
        }
        final long start = (long) ord * directRecordSize;
        final long end = start + directRecordSize - 1;
        for (long p = start / PAGE_BYTES_32K; p <= end / PAGE_BYTES_32K; p++) {
            pages32.add(p);
        }
        for (long p = start / PAGE_BYTES_8K; p <= end / PAGE_BYTES_8K; p++) {
            pages8.add(p);
        }
        // Shadow: where this read would land if .vec were reordered by the permutation.
        if (shadowPermutation != null && ord >= 0 && ord < shadowPermutation.length) {
            final long sStart = (long) shadowPermutation[ord] * directRecordSize;
            final long sEnd = sStart + directRecordSize - 1;
            for (long p = sStart / PAGE_BYTES_32K; p <= sEnd / PAGE_BYTES_32K; p++) {
                pages32Shadow.add(p);
            }
            for (long p = sStart / PAGE_BYTES_8K; p <= sEnd / PAGE_BYTES_8K; p++) {
                pages8Shadow.add(p);
            }
        }
        if (distinctOrds.add(ord)) {
            recordSize = directRecordSize;
            ordsScored++;
        }
    }

    /**
     * Records every distinct physical page the given batch of ordinals touches. A record spanning a page
     * boundary counts both pages. No-op unless enabled and inside a {@link #begin()}/{@link #end} bracket.
     *
     * @param baseOffset        byte offset where records start in the slice (0 for the paths we track)
     * @param oneVectorByteSize bytes per record (112 reordered, 96 baseline)
     * @param ords              physical ordinals about to be read
     * @param numOrds           number of valid entries in {@code ords}
     */
    public void recordBatch(final long baseOffset, final long oneVectorByteSize, final int[] ords, final int numOrds) {
        if (!active) {
            return;
        }
        recordSize = oneVectorByteSize;
        prefetchCalls++;
        for (int i = 0; i < numOrds; i++) {
            final long start = baseOffset + (long) ords[i] * oneVectorByteSize;
            final long end = start + oneVectorByteSize - 1;
            for (long p = start / PAGE_BYTES_32K; p <= end / PAGE_BYTES_32K; p++) {
                pages32.add(p);
            }
            for (long p = start / PAGE_BYTES_8K; p <= end / PAGE_BYTES_8K; p++) {
                pages8.add(p);
            }
            distinctOrds.add(ords[i]);
            ordsScored++;
        }
    }

    /**
     * Logs the read-amplification metrics for this query, at BOTH 32&nbsp;KB and 8&nbsp;KB page sizes,
     * and stops accounting.
     *
     * <p><b>Read amplification</b> is the storage cost of scattered layout: the OS/device faults in a
     * whole page for every page a query touches, but only the visited vectors' records on it are useful.
     * {@code dataUsedBytes (Y) = distinctVectors * recordSize} is page-size independent; for each page
     * size we report {@code dataReadBytes (X) = distinctPages * pageSize} and {@code readAmplification =
     * X / Y} (bytes moved per useful byte; 1.0x = perfect). A smaller page wastes less per fault but the
     * query touches more distinct pages, so 8&nbsp;KB usually shows lower amplification than 32&nbsp;KB
     * for a scattered access pattern — the gap shrinks as locality reordering packs more visited vectors
     * together. Also logged: {@code distinctPages}, {@code vecsPerPage} (useful records per touched page),
     * and {@code pageUtilPct = 100 / readAmp}. {@code recordSize} (112) attributes the run.
     *
     * @param context short label for the log line (e.g. the field / vector count)
     */
    public void end(final String context) {
        if (!active) {
            return;
        }
        active = false;
        final long distinctVectors = distinctOrds.size();
        final long usefulBytes = distinctVectors * recordSize;

        final long pages32Count = pages32.size();
        final long read32 = pages32Count * PAGE_BYTES_32K;
        final double amp32 = usefulBytes == 0 ? 0.0 : (double) read32 / usefulBytes;
        final double util32 = read32 == 0 ? 0.0 : 100.0 * usefulBytes / read32;
        final double vpp32 = pages32Count == 0 ? 0.0 : (double) distinctVectors / pages32Count;

        final long pages8Count = pages8.size();
        final long read8 = pages8Count * PAGE_BYTES_8K;
        final double amp8 = usefulBytes == 0 ? 0.0 : (double) read8 / usefulBytes;
        final double util8 = read8 == 0 ? 0.0 : 100.0 * usefulBytes / read8;
        final double vpp8 = pages8Count == 0 ? 0.0 : (double) distinctVectors / pages8Count;

        // Shadow (only for the full-precision rescore stream with a permutation): what the SAME reads
        // would touch if .vec were reordered by the permutation. This is the "if we reorder .vec too"
        // number, measured without actually reordering the file.
        String shadowSuffix = "";
        if (shadowPermutation != null) {
            final long sPages32 = pages32Shadow.size();
            final long sPages8 = pages8Shadow.size();
            final long sRead32 = sPages32 * PAGE_BYTES_32K;
            final long sRead8 = sPages8 * PAGE_BYTES_8K;
            final double sAmp32 = usefulBytes == 0 ? 0.0 : (double) sRead32 / usefulBytes;
            final double sAmp8 = usefulBytes == 0 ? 0.0 : (double) sRead8 / usefulBytes;
            shadowSuffix = String.format(
                " | REORDERED32: dataReadBytes=[%d] readAmplification=[%.1fx] distinctPages=[%d]"
                    + " | REORDERED8: dataReadBytes=[%d] readAmplification=[%.1fx] distinctPages=[%d]",
                sRead32,
                sAmp32,
                sPages32,
                sRead8,
                sAmp8,
                sPages8
            );
        }

        // Headline per page size: dataReadBytes (X) = whole pages faulted in; dataUsedBytes (Y) = visited
        // vectors' records; readAmplification = X / Y (bytes moved per useful byte, 1.0x = perfect).
        log.info(
            "PageTouch query#[{}] context=[{}] dataUsedBytes=[{}] distinctVectors=[{}] recordSize=[{}] "
                + "| 32KB: dataReadBytes=[{}] readAmplification=[{}x] distinctPages=[{}] vecsPerPage=[{}] pageUtilPct=[{}] "
                + "| 8KB: dataReadBytes=[{}] readAmplification=[{}x] distinctPages=[{}] vecsPerPage=[{}] pageUtilPct=[{}] "
                + "| ordsScored=[{}] prefetchCalls=[{}]{}",
            QUERY_SEQ.incrementAndGet(),
            context,
            usefulBytes,
            distinctVectors,
            recordSize,
            read32,
            String.format("%.1f", amp32),
            pages32Count,
            String.format("%.2f", vpp32),
            String.format("%.2f", util32),
            read8,
            String.format("%.1f", amp8),
            pages8Count,
            String.format("%.2f", vpp8),
            String.format("%.2f", util8),
            ordsScored,
            prefetchCalls,
            shadowSuffix
        );
        pages32.clear();
        pages8.clear();
        pages32Shadow.clear();
        pages8Shadow.clear();
        distinctOrds.clear();
    }
}
