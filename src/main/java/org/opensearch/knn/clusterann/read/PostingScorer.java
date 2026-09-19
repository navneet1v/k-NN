/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import java.io.IOException;

/**
 * A cursor over one {@link Cluster}'s postings, bound to a query, that scores as it advances — the same
 * iterate-and-score fusion as Lucene's {@code Scorer}.
 *
 * <p>{@link #advance} moves to the next competitive posting; {@link #ord()} and {@link #score()} expose the
 * current one. How the vectors are laid out, and whether scoring happens per-vector or a batch at a time,
 * stays inside — so a different storage family means a different scorer and an unchanged search path.
 *
 * <pre>
 * PostingScorer it = cluster.scorer(params, wanted);
 * while (it.advance(collector.minCompetitiveSimilarity())) {
 *     int ord = it.ord();
 *     collector.collect(ordToDoc(ord), it.score());   // ord → doc + collect is the searcher's hop
 * }
 * </pre>
 *
 * <p>Not thread-safe; one instance per cluster per query.
 */
public interface PostingScorer {

    /**
     * Advance to the next posting that may be competitive at {@code minCompetitiveSimilarity}. Skips
     * non-competitive work and early-terminates when the remainder is provably hopeless (both hidden
     * inside). Returns {@code false} when the cluster is exhausted or the rest cannot compete.
     */
    boolean advance(float minCompetitiveSimilarity) throws IOException;

    /** Global vector ordinal of the current posting ({@code pos → ord}). */
    int ord();

    /** Similarity of the current posting to the query (higher = closer). */
    float score();
}
