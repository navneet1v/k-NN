package org.opensearch.knn.clusterann.stats;

import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Bounded, lock-free latency/count distribution for a single {@link ClusterANNStat}.
 *
 * <p>Values are bucketed on a log<sub>2</sub> scale: bucket {@code b} holds observations in
 * {@code [2^(b-1), 2^b)} (bucket 0 holds {@code 0}). With {@link #BUCKETS} buckets this covers up to
 * {@code 2^62}, which is every count or millisecond value the codec can produce, in a fixed
 * {@code BUCKETS * 8} bytes per metric — no unbounded sample list, so memory never grows with query
 * volume. {@link #record(long)} is a single {@link AtomicLongArray#incrementAndGet(int)}, so it is
 * cheap and safe on the hot flush/merge/search paths.
 *
 * <p>Percentiles are approximate to the bucket's upper edge, which is the standard trade for a
 * fixed-memory histogram and is what the downstream collector needs to emit {@code P50/P90/P99}. The
 * value is a per-worker, process-lifetime running distribution; a reader computes an interval
 * distribution by diffing two {@link #snapshot()}s, exactly as the Juno collector diffs the running
 * totals.
 */
final class ClusterANNHistogram {

    /** 63 log2 buckets cover [0, 2^62); plenty for ms timings and per-query counts. */
    static final int BUCKETS = 63;

    private final AtomicLongArray counts = new AtomicLongArray(BUCKETS);

    /** Records one observation, clamped to {@code >= 0}, into its log2 bucket. */
    void record(final long value) {
        final long v = value < 0 ? 0 : value;
        final int bucket = v == 0 ? 0 : (64 - Long.numberOfLeadingZeros(v));
        counts.incrementAndGet(Math.min(bucket, BUCKETS - 1));
    }

    /** Total number of observations recorded. */
    long count() {
        long total = 0;
        for (int i = 0; i < BUCKETS; i++) {
            total += counts.get(i);
        }
        return total;
    }

    /**
     * Approximate {@code p}-th percentile (0&ndash;100) of a per-bucket count array as the upper edge of the
     * bucket in which the {@code p}-th observation falls, or 0 when the array is empty. Static so a collector can
     * compute the percentile over an <em>interval delta</em> — {@code cur[b] - prev[b]} of two {@link #snapshot()}s
     * — which is what makes the percentile a recent-window tail signal rather than a lifetime-cumulative value.
     */
    static long percentileOf(final long[] buckets, final double p) {
        long total = 0;
        for (final long c : buckets) {
            total += c;
        }
        if (total <= 0) {
            return 0L;
        }
        final long rank = (long) Math.ceil(p / 100.0 * total);
        long cumulative = 0;
        for (int b = 0; b < buckets.length; b++) {
            cumulative += buckets[b];
            if (cumulative >= rank) {
                return b == 0 ? 0L : (1L << b) - 1;
            }
        }
        return (1L << (buckets.length - 1)) - 1;
    }

    /**
     * Approximate {@code p}-th percentile (0&ndash;100) as the upper edge of the bucket in which the
     * {@code p}-th observation falls, or 0 when nothing has been recorded. {@code percentile(50)} is the
     * median; {@code percentile(100)} the max bucket's upper edge. This is the <em>lifetime-cumulative</em>
     * percentile over all history; for a recent-window value diff two {@link #snapshot()}s and call
     * {@link #percentileOf(long[], double)}.
     */
    long percentile(final double p) {
        return percentileOf(snapshot(), p);
    }

    /** Immutable copy of the current per-bucket counts, for interval diffing by a collector. */
    long[] snapshot() {
        final long[] out = new long[BUCKETS];
        for (int i = 0; i < BUCKETS; i++) {
            out[i] = counts.get(i);
        }
        return out;
    }
}
