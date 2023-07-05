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

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.stream.DoubleStream;

public enum PerformanceStats {
    NMSLIB_LATENCY("nmslib_latency", "nanoSec"),
    NMSLIB_JNI_LATENCY("nmslib_jni_latency", "nanoSec");

    private static final int MAX_QUERIES = 10000;
    private final Queue<Double> latencyValuesQueue;
    private final String metricName;
    private final String metricUnit;

    PerformanceStats(final String metricName, final String metricUnit) {
        this.metricName = metricName;
        this.metricUnit = metricUnit;
        this.latencyValuesQueue = new ConcurrentLinkedQueue<>();
    }

    public void addLatency(double latency) {
        resizeQueue(latencyValuesQueue);
        latencyValuesQueue.add(latency);
    }

    public List<Double> getLatencies() {
        return new ArrayList<>(latencyValuesQueue);
    }

    public Map<String, Object> getStats() {
        return ImmutableMap.of(getStatsHeaderName(), getLatencyStats());
    }

    public String getStatsHeaderName() {
        return metricName + "(" + metricUnit + ")";
    }

    private Map<String, Double> getLatencyStats() {
        final Map<String, Double> latencyStats = new LinkedHashMap<>();
        if (latencyValuesQueue.size() > 0) {
            DoubleStream doubleStream = latencyValuesQueue.stream().mapToDouble(v -> v);
            DoubleSummaryStatistics doubleSummaryStatistics = doubleStream.summaryStatistics();
            latencyStats.put("total_requests", (double) doubleSummaryStatistics.getCount());
            latencyStats.put("max", doubleSummaryStatistics.getMax());
            latencyStats.put("min", doubleSummaryStatistics.getMin());
            latencyStats.put("average", doubleSummaryStatistics.getAverage());

            final Quantiles.Scale percentiles = Quantiles.percentiles();
            latencyStats.put("p50", percentiles.index(50).compute(latencyValuesQueue));
            latencyStats.put("p90", percentiles.index(90).compute(latencyValuesQueue));
            latencyStats.put("p99", percentiles.index(99).compute(latencyValuesQueue));
        }
        return latencyStats;
    }

    private void resizeQueue(final Queue<Double> queue) {
        while (queue.size() >= MAX_QUERIES) {
            queue.poll();
        }
    }

}
