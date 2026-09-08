/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.reader.orchestration.ScanContext;

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
 * <p>Scanning is two steps: {@link #prepareScan} turns a query into the form this cluster scores against,
 * and {@link #scorer} walks the posting with it. Splitting them lets the prepared form be reused across
 * scorers and lets a scan be abandoned before paying for preparation. Nothing is read until one of the
 * two is called. One instance per query; not thread-safe.
 *
 * <p>{@link Accountable} because a scan holds one of these per cluster it may visit, and what a cluster costs
 * depends on whether it was scanned — so the total is only knowable by asking each one.
 */
public interface Cluster extends Accountable {

    /** This cluster's centroid ordinal within the field (identity; useful for bookkeeping/logging). */
    int ordinal();

    /** Number of vectors in this cluster (primary + SOAR). */
    int size();

    /**
     * Hint that this posting will be read soon so it can be warmed first.
     *
     * @param partial if true, hint only the prefix the scan is near-certain to read and stream the rest;
     *     if false, hint the whole posting (better when a near-full scan is expected).
     */
    void prefetch(boolean partial) throws IOException;

    /**
     * A cursor over this cluster's postings that scores as it advances.
     *
     * @param scanContext a query prepared by this cluster's own {@link #prepareScan}; a context from
     *     another cluster family is rejected rather than misread.
     * @param acceptedOrds ordinals to score, or {@code null} for all; honoured before scoring, not after.
     */
    PostingScorer scorer(ScanContext scanContext, Bits acceptedOrds) throws IOException;

    /**
     * Turn a query into the form this cluster scores against.
     *
     * @param scanParams the query and per-query knobs to honour
     * @return a context for {@link #scorer}, valid only for this cluster
     */
    ScanContext prepareScan(ScanParams scanParams) throws IOException;
}
