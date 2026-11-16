package org.opensearch.knn.clusterann.stats;

import java.util.concurrent.atomic.LongAdder;

/**
 * In-memory, process-lifetime accumulators for ClusterANN (IVF) codec metrics.
 *
 * <p>Each constant holds two {@link LongAdder}s: a running {@code total} and an {@code occurrences}
 * count. A caller {@link #record(long)}s a per-event value (e.g. a phase time in ms, a centroid count,
 * a probe count) which adds to {@code total} and bumps {@code occurrences}; {@link #increment()} is the
 * degenerate case for pure event counters (adds 1 to both). The pair lets a downstream collector compute
 * a per-interval average as {@code delta(total) / delta(occurrences)}, matching how the Juno
 * {@code NodeStatsCollector} / {@code KNNScheduledMetricCollector} already treat the HNSW latency/
 * graph-build metrics.
 *
 * <p>{@link LongAdder} (not {@link java.util.concurrent.atomic.AtomicLong}) is deliberate: the access
 * pattern is write-heavy and read-rare — the hot flush/merge/search paths call the mutators from many
 * threads, while the collector reads {@link #getTotal()}/{@link #getCount()} once per ~30s interval.
 * {@code LongAdder} stripes writes across per-thread cells to avoid the single-hot-cache-line CAS
 * contention an {@code AtomicLong} would suffer under that load, trading a slightly more expensive
 * {@code sum()} on the rare read — exactly the right trade here.
 *
 * <p>These values are <em>not</em> persisted. They live on the JVM heap for the life of the worker
 * process; durability and history come from the Juno collector sampling them (and the OpenSearch
 * {@code _stats} API reading them) and shipping deltas to CloudWatch. Each worker holds its own copy;
 * aggregation across workers happens downstream. Recording is O(1) amortized, and safe to call
 * from any thread on the hot flush/merge/search paths.
 *
 * <p>Fractions are stored as basis points ({@code _BPS}, value &times; 10000) so the accumulators stay
 * integer-only; the consumer divides by 10000 to recover the ratio.
 */
public enum ClusterANNStat {

    // ---- A. Flush (write path; executor == null) ----
    FLUSH_OPERATIONS("clusterann_flush_operations"),
    FLUSH_TIME_MS("clusterann_flush_time_ms"),
    FLUSH_DOCS("clusterann_flush_docs"),
    FLUSH_EMPTY_FIELDS("clusterann_flush_empty_fields"),
    FLUSH_ERRORS("clusterann_flush_errors"),

    // ---- B. Merge (write path; executor != null) ----
    MERGE_CURRENT_OPERATIONS("clusterann_merge_current_operations"),
    MERGE_OPERATIONS("clusterann_merge_operations"),
    MERGE_TIME_MS("clusterann_merge_time_ms"),
    MERGE_DOCS("clusterann_merge_docs"),
    MERGE_ERRORS("clusterann_merge_errors"),
    MERGE_ABORTED("clusterann_merge_aborted"),

    // ---- C. Build phases (cluster build — NOT graph) ----
    FLUSH_CLUSTER_BUILD_TIME_MS("clusterann_flush_cluster_build_time_ms"),
    MERGE_CLUSTER_BUILD_TIME_MS("clusterann_merge_cluster_build_time_ms"),
    FLUSH_ENCODE_TIME_MS("clusterann_flush_encode_time_ms"),
    MERGE_ENCODE_TIME_MS("clusterann_merge_encode_time_ms"),
    ROTATION_TIME_MS("clusterann_rotation_time_ms"),
    CENTROID_WRITE_TIME_MS("clusterann_centroid_write_time_ms"),

    // ---- D. Clustering structure ----
    CENTROID_COUNT("clusterann_centroid_count"),
    AVG_CLUSTER_SIZE("clusterann_avg_cluster_size"),
    MAX_CLUSTER_SIZE("clusterann_max_cluster_size"),
    MIN_CLUSTER_SIZE("clusterann_min_cluster_size"),
    EMPTY_CLUSTERS("clusterann_empty_clusters"),
    CLUSTER_SIZE_SKEW_BPS("clusterann_cluster_size_skew_bps"),
    LLOYD_ITERATIONS("clusterann_lloyd_iterations"),

    // ---- E. SOAR ----
    SOAR_ASSIGNMENTS("clusterann_soar_assignments"),
    SOAR_SPILL_VECTORS("clusterann_soar_spill_vectors"),
    SOAR_SPILL_RATE_BPS("clusterann_soar_spill_rate_bps"),

    // ---- F. Search — query ----
    QUERY_REQUESTS("clusterann_query_requests"),
    QUERY_ERRORS("clusterann_query_errors"),
    QUERY_LATENCY_US("clusterann_query_latency_us"),
    PROBES_PLANNED("clusterann_probes_planned"),
    CLUSTERS_SCANNED("clusterann_clusters_scanned"),
    EMPTY_PROBES_SKIPPED("clusterann_empty_probes_skipped"),
    /**
     * Positions whose score was actually computed: every accepted position of every block the scan decoded,
     * counted where {@code scoreBlock} runs. One observation per scored block, so the average is the mean
     * number of positions a scored block contributed.
     *
     * <p>This is the scan's real CPU cost, and it is strictly larger than
     * {@link #VECTORS_COLLECTED_PER_QUERY}: a block whose best score cannot beat the collector's threshold is
     * scored in full and then dropped without a single position reaching the collector.
     */
    VECTORS_SCORED("clusterann_vectors_scored"),
    /**
     * Distinct ordinals handed to the collector, one observation per query. Not a scoring count — it is what
     * survived the competitive threshold, after SOAR dedup and the filter. Compare against
     * {@link #VECTORS_SCORED} to see what the per-block pruning is worth.
     */
    VECTORS_COLLECTED_PER_QUERY("clusterann_vectors_collected_per_query"),
    /** Blocks whose codes were fetched and decoded, which is the scan's IO. */
    BLOCKS_FETCHED("clusterann_blocks_fetched"),
    EXACT_FALLBACK("clusterann_exact_fallback"),
    CENTROID_KNEE_EARLY_STOP("clusterann_centroid_knee_early_stop"),

    // ---- G. Search — filtered ----
    QUERY_WITH_FILTER_REQUESTS("clusterann_query_with_filter_requests"),
    FILTER_SELECTIVITY_BPS("clusterann_filter_selectivity_bps"),
    FILTER_PROBE_MULTIPLIER_BPS("clusterann_filter_probe_multiplier_bps"),
    FILTER_ELIGIBLE_CLUSTERS("clusterann_filter_eligible_clusters"),
    FILTER_EXTENDED_PAST_MAX_PROBES("clusterann_filter_extended_past_max_probes"),

    // ---- H. Quantization / encoding ----
    INT8_ENCODE_COUNT("clusterann_int8_encode_count"),
    QUANTIZATION_ERRORS("clusterann_quantization_errors");

    private final String name;
    private final LongAdder total;
    private final LongAdder occurrences;
    /** Per-metric log2 distribution, for p50/p90/p99 of per-event values (query cost, phase times, sizes). */
    private final ClusterANNHistogram histogram;

    ClusterANNStat(final String name) {
        this.name = name;
        this.total = new LongAdder();
        this.occurrences = new LongAdder();
        this.histogram = new ClusterANNHistogram();
    }

    /** Metric name as surfaced to {@code _stats} and used to derive the Juno/EMF name. */
    public String getName() {
        return name;
    }

    /** Cumulative sum of all recorded values since process start. */
    public long getTotal() {
        return total.sum();
    }

    /** Cumulative number of record/increment events since process start. */
    public long getCount() {
        return occurrences.sum();
    }

    /** Mean recorded value ({@code total / occurrences}), or 0 when nothing has been recorded. */
    public long getAverage() {
        final long c = occurrences.sum();
        return c == 0 ? 0L : total.sum() / c;
    }

    /**
     * Approximate percentile ({@code p} in 0&ndash;100) of the per-event value distribution over the process's
     * <em>entire lifetime</em>. Useful for a one-shot {@code _stats} read, but it is cumulative and never windowed,
     * so on a long-running worker a recent spike is diluted by all prior history. For a recent-window tail signal,
     * a collector should diff two {@link #histogramSnapshot()}s and call {@link #intervalPercentile(long[], long[],
     * double)} instead — the holder cannot window on its own because it does not know the collection interval.
     */
    public long getPercentile(final double p) {
        return histogram.percentile(p);
    }

    /** Convenience: lifetime-cumulative median of the per-event values (see {@link #getPercentile(double)}). */
    public long getP50() {
        return histogram.percentile(50);
    }

    /** Convenience: lifetime-cumulative p90 of the per-event values (see {@link #getPercentile(double)}). */
    public long getP90() {
        return histogram.percentile(90);
    }

    /** Convenience: lifetime-cumulative p99 of the per-event values (see {@link #getPercentile(double)}). */
    public long getP99() {
        return histogram.percentile(99);
    }

    /** Immutable per-bucket snapshot of the distribution, for interval diffing by a metrics collector. */
    public long[] histogramSnapshot() {
        return histogram.snapshot();
    }

    /**
     * Percentile over a collection interval: computes the {@code p}-th percentile of the observations recorded
     * <em>between</em> two {@link #histogramSnapshot()}s (bucket-wise {@code cur - prev}). A collector keeps the
     * previous snapshot, takes a new one each interval, and calls this to emit a recent-window p50/p90/p99 — the
     * recommended way to consume the distribution so a recent tail-cost spike is not diluted by lifetime history.
     */
    public static long intervalPercentile(final long[] prev, final long[] cur, final double p) {
        final long[] delta = new long[cur.length];
        for (int b = 0; b < cur.length; b++) {
            final long d = cur[b] - (prev != null && b < prev.length ? prev[b] : 0L);
            delta[b] = d > 0 ? d : 0L;
        }
        return ClusterANNHistogram.percentileOf(delta, p);
    }

    /** Records one observation of {@code value}: adds to the total, bumps the count, and buckets it. */
    public void record(final long value) {
        total.add(value);
        occurrences.increment();
        histogram.record(value);
    }

    /**
     * Records a single unit event (counter/gauge increment). Unlike {@link #record(long)} this does NOT feed the
     * percentile histogram: an {@code increment()}/{@link #decrement()} gauge has no meaningful per-event value
     * distribution, and feeding it would leave the histogram unable to un-count on decrement. Percentiles
     * ({@link #getP50()} etc.) are therefore only meaningful for {@code record()}/{@code incrementBy()} metrics.
     */
    public void increment() {
        total.increment();
        occurrences.increment();
    }

    /** Adds {@code delta} to the total as a single event (one occurrence), bucketing the delta. */
    public void incrementBy(final long delta) {
        total.add(delta);
        occurrences.increment();
        histogram.record(delta);
    }

    /**
     * Decrements the total and occurrence count by one; for in-flight gauges such as current merges. Paired with
     * {@link #increment()}, it does not touch the histogram (see that method) — so a gauge's total/count stay
     * consistent and its percentiles remain undefined rather than drifting.
     */
    public void decrement() {
        total.decrement();
        occurrences.decrement();
    }
}
