/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;

/**
 * Hands Lucene a scorer that scores a segment's candidates against the stored full-precision vectors.
 *
 * <p>Used for both exact passes a nested query makes, because they are the same pass over different sets:
 *
 * <ul>
 *   <li><b>rescore</b> — the candidates the scan surfaced, with a parents filter, so each parent contributes only its
 *       nearest child and the result is k parents rather than k children
 *   <li><b>expand</b> — the siblings of whatever survived, with <em>no</em> parents filter, because here every child is
 *       wanted on its own score and deduping is exactly the wrong thing
 * </ul>
 *
 * One argument apart, they are identical — which is why they are one class.
 *
 * <p><b>Nothing is collected.</b> The scorer is returned rather than drained, so Lucene's collector does the top-k and in
 * return pushes its competitive threshold back through {@link Scorer#setMinCompetitiveScore}. That feedback is the point:
 * the bulk scorer reads a batch, finds its best score already below the bar, and skips the batch without reading a vector
 * in it. Draining it here would forfeit that, since only the collector knows what the bar is at any moment.
 */
final class ExactScoringWeight extends Weight {

    private final String field;
    private final float[] queryVector;
    private final CandidateSource candidates;

    /** Set for the rescore pass, {@code null} for the expand pass. See the class javadoc. */
    @Nullable
    private final BitSetProducer parentsFilter;

    private final ExactSearcher exactSearcher;
    private final float boost;

    ExactScoringWeight(
        final Query query,
        final String field,
        final float[] queryVector,
        final CandidateSource candidates,
        @Nullable final BitSetProducer parentsFilter,
        final ExactSearcher exactSearcher,
        final float boost
    ) {
        super(query);
        this.field = field;
        this.queryVector = queryVector;
        this.candidates = candidates;
        this.parentsFilter = parentsFilter;
        this.exactSearcher = exactSearcher;
        this.boost = boost;
    }

    /**
     * @return {@code null} when this segment holds no candidate, so it is never opened. After a global cap that is most
     *     segments, which is what keeps the exact work proportional to the candidate count rather than the segment count.
     */
    @Override
    @Nullable
    public ScorerSupplier scorerSupplier(final LeafReaderContext context) throws IOException {
        final int size = candidates.size(context);
        if (size == 0) {
            return null;
        }
        return new ScorerSupplier() {
            @Override
            public Scorer get(final long leadCost) throws IOException {
                // Boosted here rather than inside the exact searcher: the boost belongs to this query, and the searcher is
                // shared with callers that have none.
                return BoostedScorer.boosted(exactSearcher.exactSearchScorer(context, searcherContext(context)), boost);
            }

            @Override
            public long cost() {
                return size;
            }
        };
    }

    /**
     * {@code useQuantizedVectorsForSearch(false)} is what makes this exact rather than a repeat of the scan: it takes the
     * full-precision vectors, which is the only reason correcting the approximate ordering is worth anything.
     *
     * <p>When {@code parentsFilter} is set the searcher wraps the scorer so that it yields one best child per parent, and
     * consumes the candidate iterator itself — so the iterator is handed over for that purpose, not merely as a filter.
     */
    private ExactSearcher.ExactSearcherContext searcherContext(final LeafReaderContext context) throws IOException {
        final int size = candidates.size(context);
        return ExactSearcher.ExactSearcherContext.builder()
            .field(field)
            .floatQueryVector(queryVector)
            .matchedDocsIterator(candidates.iterator(context))
            .numberOfMatchedDocs(size)
            .parentsFilter(parentsFilter)
            .useQuantizedVectorsForSearch(false)
            .k(size)
            .build();
    }

    @Override
    public Explanation explain(final LeafReaderContext context, final int doc) throws IOException {
        final ScorerSupplier supplier = scorerSupplier(context);
        if (supplier == null) {
            return Explanation.noMatch("no candidate for this query in this segment");
        }
        final Scorer scorer = supplier.get(Long.MAX_VALUE);
        if (scorer.iterator().advance(doc) == doc) {
            return Explanation.match(scorer.score(), "exact score against the stored full-precision vector");
        }
        return Explanation.noMatch("not among this segment's candidates");
    }

    /**
     * Exact and free — the candidate set is fixed before this weight exists — so a collector that only needs a count can
     * skip scoring the segment entirely.
     *
     * <p>Except with a parents filter, where the answer is the number of <em>parents</em> represented and the candidate
     * count is only an upper bound. Reporting a wrong count is worse than reporting none.
     */
    @Override
    public int count(final LeafReaderContext context) throws IOException {
        return parentsFilter == null ? candidates.size(context) : super.count(context);
    }

    /**
     * Never cacheable: the entry would be keyed on a query vector that never repeats, so it could not hit, and filling it
     * would cost another pass of exact scoring.
     */
    @Override
    public boolean isCacheable(final LeafReaderContext context) {
        return false;
    }
}
