/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

/**
 * Per-query I/O instrumentation for ClusterANN, kept thread-local (one query per thread) and read by the
 * native query layer for its {@code [ClusterANN-IO]} log line. Currently just the ADC-scan byte counter,
 * a placeholder — the block reader doesn't account bytes yet, so it reads 0. Split out of the former
 * {@code OptimizedProbeScheduler} so the instrumentation survives its removal.
 */
public final class ClusterANNQueryStats {

    private ClusterANNQueryStats() {}

    private static final ThreadLocal<long[]> ADC_BYTES = ThreadLocal.withInitial(() -> new long[1]);

    /** Reset the ADC-scan byte counter at the start of a query. */
    public static void resetQueryAdcBytes() {
        ADC_BYTES.get()[0] = 0;
    }

    /** Add to the current query's ADC-scan byte counter (TODO: wire from the block reader). */
    public static void addAdcBytes(long bytes) {
        ADC_BYTES.get()[0] += bytes;
    }

    /** Bytes read for the ADC scan in the current query on this thread. */
    public static long getLastQueryAdcBytes() {
        return ADC_BYTES.get()[0];
    }
}
