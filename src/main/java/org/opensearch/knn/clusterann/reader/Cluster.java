/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * One IVF cluster. Everything scoped to it stays behind it: its centroid geometry, its membership, and
 * how its vectors are stored.
 *
 * <p>It hands out a {@link PostingScorer} over its own vectors — a cursor that scores as it advances.
 * Layout and scoring both live inside that scorer because they are not separable: the score is computed
 * from the stored form. So a caller drives any storage family without learning anything about it, and
 * whatever a query may vary arrives as data in {@link ScanParams}.
 *
 * <p>All reads are deferred to {@link #scorer}, including this cluster's own centroid. One instance per
 * query; not thread-safe.
 */
public interface Cluster {

    /** This cluster's centroid ordinal within the field (identity; useful for bookkeeping/logging). */
    int ordinal();

    /** Number of vectors in this cluster (primary + SOAR). */
    int size();

    /**
     * Hint that this cluster's posting will be read soon, so it can be warmed before the scan
     * arrives. Hinting is free since obtaining a cluster reads nothing. This belongs on the cluster
     * because only it knows how its postings are laid out.
     *
     * @param partial if true, hint only the prefix the scan is near-certain to read, letting the scan
     *     stream the rest — this avoids over-fetching a posting. If false, hint the whole posting,
     *     which is better when a near-full scan is expected since one large sequential hint beats many small ones.
     */
    void prefetch(boolean partial) throws IOException;

    /**
     * A scorer over this cluster's postings — a cursor that scores as it advances.
     *
     * @param params the prepared query plus the per-query scoring knobs this cluster should honour
     *     (see {@link ScanParams}).
     * @param acceptedOrds the ordinals the caller wants scored, or {@code null} for all. Read-only, and
     *     honoured <em>before</em> scoring rather than filtered afterwards.
     */
    PostingScorer scorer(ScanParams params, Bits acceptedOrds) throws IOException;
}
