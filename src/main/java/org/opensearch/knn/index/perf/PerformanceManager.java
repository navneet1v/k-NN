/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.perf;

import com.google.common.collect.ImmutableMap;
import com.google.common.math.Quantiles;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.util.DoubleSummaryStatistics;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.stream.DoubleStream;

public class PerformanceManager {
    private static PerformanceManager INSTANCE;

    private final Queue<Double> nmslibNativeCodeLatencyQueue;
    private final Queue<Double> nmslibLatencyQueue;

    private static final int MAX_QUERIES = 10000; // this should come from cluster settings

    public static synchronized PerformanceManager getInstance() {
        if(INSTANCE == null) {
            INSTANCE = new PerformanceManager();
        }
        return INSTANCE;
    }

    private PerformanceManager() {
        nmslibNativeCodeLatencyQueue = new ConcurrentLinkedQueue<>();
        nmslibLatencyQueue = new ConcurrentLinkedQueue<>();
    }

    public Map<String, Object> getPerformanceStats() {
        KNNCounter.KNN_PERF_STATS.increment();
        return ImmutableMap.of(KNNCounter.KNN_PERF_STATS.getName(),
                KNNCounter.KNN_PERF_STATS.getCount().toString(), "nmslibNativeStats(nanoSec)",
                createNMSLibNativePerfStats(), "nmslibStats(nanoSec)",
                createNMSLibPerfStats());
    }

    private Map<String, Double> createNMSLibNativePerfStats() {
        final Map<String, Double> nsmLibStats = new LinkedHashMap<>();
        if(nmslibNativeCodeLatencyQueue.size() > 0) {
            DoubleStream doubleStream = nmslibNativeCodeLatencyQueue.stream().mapToDouble(v -> v);
            DoubleSummaryStatistics doubleSummaryStatistics = doubleStream.summaryStatistics();
            nsmLibStats.put("total_requests", (double) doubleSummaryStatistics.getCount());
            nsmLibStats.put("max", doubleSummaryStatistics.getMax());
            nsmLibStats.put("min", doubleSummaryStatistics.getMin());
            nsmLibStats.put("average", doubleSummaryStatistics.getAverage());

            final Quantiles.Scale percentiles = Quantiles.percentiles();
            nsmLibStats.put("p50", percentiles.index(50).compute(nmslibNativeCodeLatencyQueue));
            nsmLibStats.put("p90", percentiles.index(90).compute(nmslibNativeCodeLatencyQueue));
            nsmLibStats.put("p99", percentiles.index(99).compute(nmslibNativeCodeLatencyQueue));
        }
        return nsmLibStats;
    }

    private Map<String, Double> createNMSLibPerfStats() {
        final Map<String, Double> nsmLibStats = new LinkedHashMap<>();
        if(nmslibLatencyQueue.size() > 0) {
            DoubleStream doubleStream = nmslibLatencyQueue.stream().mapToDouble(v -> v);
            DoubleSummaryStatistics doubleSummaryStatistics = doubleStream.summaryStatistics();
            nsmLibStats.put("total_requests", (double) doubleSummaryStatistics.getCount());
            nsmLibStats.put("max", doubleSummaryStatistics.getMax());
            nsmLibStats.put("min", doubleSummaryStatistics.getMin());
            nsmLibStats.put("average", doubleSummaryStatistics.getAverage());

            final Quantiles.Scale percentiles = Quantiles.percentiles();
            nsmLibStats.put("p50", percentiles.index(50).compute(nmslibLatencyQueue));
            nsmLibStats.put("p90", percentiles.index(90).compute(nmslibLatencyQueue));
            nsmLibStats.put("p99", percentiles.index(99).compute(nmslibLatencyQueue));
        }
        return nsmLibStats;
    }

    public void addNMSLibNativeLatency(double latency) {
        resizeQueue(nmslibNativeCodeLatencyQueue);
        nmslibNativeCodeLatencyQueue.add(latency);
    }

    public void addNMSLibLatency(double latency) {
        resizeQueue(nmslibLatencyQueue);
        nmslibLatencyQueue.add(latency);
    }

    private void resizeQueue(final Queue<Double> queue) {
        while(queue.size() >= MAX_QUERIES) {
            queue.poll();
        }
    }

}
