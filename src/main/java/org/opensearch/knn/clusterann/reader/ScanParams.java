/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

/**
 * Everything a scan varies per query: the query itself plus the scoring knobs a {@link Cluster} should
 * honour. Carrying them as data is what keeps {@link Cluster#scorer} storage-agnostic — a family ignores
 * the knobs that do not apply to it, and adding one does not change the interface.
 *
 * <p>Built once per query and shared across every cluster the scan visits, so {@code query} must not be
 * mutated while a scan is in flight.
 *
 * @param query the query vector, not copied.
 * @param queryBits query-side quantisation width. Can be used for SDC vs ADC scoring. The width a cluster
 *     actually supports is the cluster's business, so it is not constrained here — a scorer rejects a width
 *     it has no kernel for.
 */
public record ScanParams(float[] query, int queryBits) {

    /** Query-side width used unless a query asks otherwise: precise enough that ADC bias stays small. */
    public static final int DEFAULT_QUERY_BITS = 4;

    /** Scan with the default query width — the form to use unless a query asks for a different one. */
    public static ScanParams of(float[] query) {
        return new ScanParams(query, DEFAULT_QUERY_BITS);
    }
}
