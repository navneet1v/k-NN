/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * What a query asks of a posting scan: the prepared query and the per-query facts that shape how it is
 * scanned.
 *
 * <p>Deliberately plain data. It is what the walk carries and forwards to {@link Cluster#scorer}, so a
 * query can influence scoring without the search path naming — or being able to reach — anything about
 * how a posting is stored. How these are honoured is entirely the cluster's business.
 *
 * <p>{@code queryBits} is the query-side quantization width, and it is the axis along which the scoring
 * math varies over identical stored codes:
 * <ul>
 *   <li>{@code queryBits > docBits} — asymmetric (ADC): a higher-precision query against coarse doc
 *       codes. This is the default and the accurate end of the trade.</li>
 *   <li>{@code queryBits == docBits} — symmetric (SDC): cheapest kernel (e.g. 1-bit query against 1-bit
 *       docs reduces to popcount), at coarser scores.</li>
 * </ul>
 * Lowering it also degrades the pruning bounds — see {@code ScalarQuantizedCluster}, which picks pruners
 * knowing this width.
 *
 * <p>{@code filterSelectivity} is the fraction of the field's vectors the query's filter accepts
 * ({@code 1.0} when unfiltered). The walk uses it to skip clusters unlikely to hold a single match.
 *
 * <p>Note on byte vectors: the query is held as {@code float[]} because scoring quantizes it relative to
 * a centroid, and that residual arithmetic is float. A {@code byte[]}-valued field would therefore widen
 * its query when preparing it, keeping the byte form a boundary concern. If a storage family ever scores
 * stored bytes directly against a byte query, this record is where that second form would live.
 */
public record ScanParams(float[] query, int queryBits, float filterSelectivity) {

    /** Query-side width used unless a query asks otherwise: precise enough that ADC bias stays small. */
    public static final int DEFAULT_QUERY_BITS = 4;

    public ScanParams {
        if (queryBits != 1 && queryBits != 2 && queryBits != 4) {
            throw new IllegalArgumentException("queryBits must be 1, 2, or 4, got: " + queryBits);
        }
        if (filterSelectivity < 0f || filterSelectivity > 1f) {
            throw new IllegalArgumentException("filterSelectivity must be in [0, 1], got: " + filterSelectivity);
        }
    }

    /** Scan with the default query width, unfiltered. */
    public static ScanParams of(float[] query) {
        return new ScanParams(query, DEFAULT_QUERY_BITS, 1.0f);
    }

    /** Scan with the default query width, under a filter of the given selectivity. */
    public static ScanParams of(float[] query, float filterSelectivity) {
        return new ScanParams(query, DEFAULT_QUERY_BITS, filterSelectivity);
    }
}
