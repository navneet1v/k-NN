/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.stats;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link ClusterANNStat} (JUnit 5).
 *
 * <p>The constants are process-global singletons that only ever accumulate, so every assertion is written against a
 * <em>delta</em> (snapshot before, snapshot after) rather than an absolute value. That keeps the tests independent of
 * execution order and of any recording the real codec may have done earlier in the JVM.
 */
class ClusterANNStatTests {

    /** {@code record(v)} adds the value to the total and bumps the occurrence count by one. */
    @Test
    void testRecord_thenTotalAndCountAdvanceByValueAndOne() {
        final ClusterANNStat stat = ClusterANNStat.FLUSH_TIME_MS;
        final long total0 = stat.getTotal();
        final long count0 = stat.getCount();

        stat.record(42);
        stat.record(8);

        assertEquals(50, stat.getTotal() - total0, "total should accumulate the recorded values");
        assertEquals(2, stat.getCount() - count0, "count should advance once per record");
    }

    /** {@code increment()} is the unit-event case: total and count both move by one. */
    @Test
    void testIncrement_thenTotalAndCountEachAdvanceByOne() {
        final ClusterANNStat stat = ClusterANNStat.QUERY_REQUESTS;
        final long total0 = stat.getTotal();
        final long count0 = stat.getCount();

        stat.increment();
        stat.increment();
        stat.increment();

        assertEquals(3, stat.getTotal() - total0);
        assertEquals(3, stat.getCount() - count0);
    }

    /** {@code incrementBy(delta)} adds the delta to the total but counts as a single occurrence. */
    @Test
    void testIncrementBy_thenTotalAdvancesByDeltaAndCountByOne() {
        final ClusterANNStat stat = ClusterANNStat.EMPTY_CLUSTERS;
        final long total0 = stat.getTotal();
        final long count0 = stat.getCount();

        stat.incrementBy(7);

        assertEquals(7, stat.getTotal() - total0, "incrementBy adds the whole delta to the total");
        assertEquals(1, stat.getCount() - count0, "incrementBy is a single occurrence");
    }

    /** {@code decrement()} reverses a prior unit event, so an inc/dec pair nets to zero on both counters. */
    @Test
    void testIncrementThenDecrement_thenNetsToZero() {
        final ClusterANNStat stat = ClusterANNStat.MERGE_CURRENT_OPERATIONS;
        final long total0 = stat.getTotal();
        final long count0 = stat.getCount();

        stat.increment();
        stat.decrement();

        assertEquals(0, stat.getTotal() - total0, "an inc/dec pair leaves the gauge where it started");
        assertEquals(0, stat.getCount() - count0);
    }

    /** {@code getAverage()} is total/occurrences over the recorded window; verified via a freshly isolated metric. */
    @Test
    void testGetAverage_thenReturnsMeanOfRecordedValues() {
        // ROTATION_TIME_MS is not touched by any other test here, so its running average is meaningful on a fresh JVM.
        // Guard against cross-test contamination by only asserting the relationship, not an absolute mean.
        final ClusterANNStat stat = ClusterANNStat.ROTATION_TIME_MS;
        final long total0 = stat.getTotal();
        final long count0 = stat.getCount();

        stat.record(10);
        stat.record(20);
        stat.record(30);

        final long windowTotal = stat.getTotal() - total0;
        final long windowCount = stat.getCount() - count0;
        assertEquals(60, windowTotal);
        assertEquals(3, windowCount);
        // getAverage is defined over the lifetime total/count; assert it is consistent with those lifetime values.
        assertEquals(stat.getTotal() / stat.getCount(), stat.getAverage());
    }

    /** Each constant exposes a stable, non-blank metric name. */
    @Test
    void testGetName_thenEveryConstantHasAClusterAnnName() {
        for (final ClusterANNStat stat : ClusterANNStat.values()) {
            assertNotNull(stat.getName());
            assertTrue(stat.getName().startsWith("clusterann_"), "metric name should be namespaced: " + stat.getName());
        }
    }

    /** Recording into one constant does not disturb another: the enum constants hold independent accumulators. */
    @Test
    void testRecord_thenOtherConstantsAreUnaffected() {
        final ClusterANNStat a = ClusterANNStat.CENTROID_COUNT;
        final ClusterANNStat b = ClusterANNStat.CLUSTERS_SCANNED;
        final long bTotal0 = b.getTotal();
        final long bCount0 = b.getCount();

        a.record(999);

        assertEquals(0, b.getTotal() - bTotal0, "an unrelated metric must not move");
        assertEquals(0, b.getCount() - bCount0);
    }

    // ---------------------------------------------------------------- every metric

    /**
     * Every single metric is independently recordable: for each constant, a {@code record(v)} advances only that
     * constant's own total and count by the expected amount. Parameterizing over {@link ClusterANNStat#values()}
     * guarantees a newly added metric is covered automatically.
     */
    @ParameterizedTest
    @EnumSource(ClusterANNStat.class)
    void testEveryMetric_isIndependentlyRecordable(final ClusterANNStat stat) {
        final long total0 = stat.getTotal();
        final long count0 = stat.getCount();

        stat.record(5);
        assertEquals(5, stat.getTotal() - total0, stat + " total should advance by the recorded value");
        assertEquals(1, stat.getCount() - count0, stat + " count should advance by one");

        stat.incrementBy(3);
        assertEquals(8, stat.getTotal() - total0, stat + " total should accumulate incrementBy");
        assertEquals(2, stat.getCount() - count0, stat + " incrementBy is one occurrence");
    }

    /** Every metric name is unique across the enum, so no two series collide in {@code _stats}/EMF. */
    @Test
    void testNames_areUniqueAcrossAllMetrics() {
        final java.util.Set<String> seen = new java.util.HashSet<>();
        for (final ClusterANNStat stat : ClusterANNStat.values()) {
            assertTrue(seen.add(stat.getName()), "duplicate metric name: " + stat.getName());
        }
    }

    // ---------------------------------------------------------------- distribution / percentiles

    /** Recording a spread of values yields monotonically ordered percentiles that bracket the data. */
    @Test
    void testPercentiles_thenOrderedAndBracketTheData() {
        // Record a known spread into a FRESH histogram rather than the shared CENTROID_WRITE_TIME_MS enum constant:
        // the production codec records into that constant on every field write, so a prior codec run or component
        // test in the same JVM would pollute its lifetime-cumulative histogram and make these absolute bounds flaky.
        final ClusterANNHistogram h = new ClusterANNHistogram();
        // Values 1..100; with log2 bucketing the percentile is the bucket's upper edge (power-of-two minus one).
        for (int v = 1; v <= 100; v++) {
            h.record(v);
        }
        final long p50 = h.percentile(50);
        final long p90 = h.percentile(90);
        final long p99 = h.percentile(99);

        assertTrue(p50 > 0, "p50 should be positive after recording positive values");
        assertTrue(p50 <= p90, "percentiles must be non-decreasing: p50 <= p90");
        assertTrue(p90 <= p99, "percentiles must be non-decreasing: p90 <= p99");
        assertTrue(p99 >= 64 && p99 <= 127, "p99 of values up to 100 falls in the [64,128) bucket, got " + p99);
    }

    /** With no observations the percentile is zero rather than throwing. */
    @Test
    void testPercentiles_whenEmpty_thenZero() {
        // Use a metric this test owns exclusively (nothing else records MIN_CLUSTER_SIZE here before this runs
        // in isolation); still assert the empty-contract via the histogram directly to avoid cross-test coupling.
        final ClusterANNHistogram empty = new ClusterANNHistogram();
        assertEquals(0L, empty.percentile(50));
        assertEquals(0L, empty.percentile(99));
        assertEquals(0L, empty.count());
    }

    /** The histogram buckets a value into [2^(b-1), 2^b) and reports that bucket's upper edge. */
    @Test
    void testHistogram_bucketsByLog2() {
        final ClusterANNHistogram h = new ClusterANNHistogram();
        h.record(0);     // bucket 0 -> edge 0
        h.record(1);     // [1,2) -> edge 1
        h.record(1000);  // [512,1024) -> edge 1023
        assertEquals(3L, h.count());
        assertEquals(1023L, h.percentile(100), "max observation sits in the [512,1024) bucket, edge 1023");
    }

    /** A snapshot is an independent copy: later records do not mutate an earlier snapshot. */
    @Test
    void testHistogramSnapshot_isIndependentCopy() {
        final ClusterANNHistogram h = new ClusterANNHistogram();
        h.record(5);
        final long[] snap = h.snapshot();
        h.record(5);
        final long sumSnap = java.util.Arrays.stream(snap).sum();
        assertEquals(1L, sumSnap, "snapshot captured one observation and is not mutated by later records");
        assertEquals(2L, h.count(), "the live histogram still advanced");
    }
}
