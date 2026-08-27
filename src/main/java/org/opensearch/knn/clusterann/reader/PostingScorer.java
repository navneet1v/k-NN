/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import java.io.IOException;

/**
 * A scored iterator over one {@link Cluster}'s postings, bound to a query — "iterate the postings,
 * scoring while iterating" (the same iterate-and-score fusion as Lucene's {@code Scorer}).
 *
 * <p>{@link #advance} moves to the next competitive posting; {@link #ord()}/{@link #score()} expose
 * the current one. All storage detail is hidden behind it: whether vectors are laid out block-columnar
 * or row-major, and whether scoring is bulk (a whole block at once) or per-vector, is the
 * implementation's detail — so a different {@link Cluster} storage produces a different iterator and
 * the search path is unchanged.
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
