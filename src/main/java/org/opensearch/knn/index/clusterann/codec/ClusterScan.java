/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * One query's pass over one field's clusters — everything a scan needs that {@link Clusters} cannot hold.
 *
 * <p>{@link Clusters} is segment-scoped and shared by concurrent searches, so it can carry nothing that
 * varies per query. Yet a scan has real per-query state to amortise: the query has to be projected and
 * quantized against a reference before any code can be scored, and that work is shared by every posting
 * quantized against the same reference. This is where that state lives — created once per query per
 * field, used by every posting it touches, then discarded.
 *
 * <p>Deliberately opaque. What the state actually <em>is</em> depends entirely on how the field was
 * written, so a caller only ever asks for a {@link PostingScorer} and never learns that a query was
 * quantized at all. One instance per query per field; <b>not</b> thread-safe.
 */
public interface ClusterScan {

    /**
     * A cursor over {@code cluster}'s posting, scoring against this scan's query.
     *
     * <p>This is where the posting is first read. A {@link Cluster} reads nothing and neither does obtaining
     * the scan, so a query that ends up touching three of four thousand clusters pays for three.
     *
     * @param wanted the ordinals the caller wants scored, or {@code null} for all. Read-only, and honoured
     *     <em>before</em> scoring rather than filtered afterwards.
     */
    PostingScorer scorer(Cluster cluster, Bits wanted) throws IOException;
}
