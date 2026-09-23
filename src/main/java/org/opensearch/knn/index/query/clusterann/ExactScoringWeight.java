/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.KNNScorer;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Scores a segment's candidates against the stored full-precision vectors.
 *
 * <p>One segment at a time, and only the segments that hold a candidate — after a global cap that is usually a small
 * minority of them, which is what keeps the exact work proportional to the candidate count rather than the segment count.
 *
 * <p><b>The exact pass runs to completion per segment and its result is replayed as a scorer.</b> The reference
 * implementation hands Lucene a scorer that scores lazily, so the collector's competitive threshold can feed back through
 * {@link Scorer#setMinCompetitiveScore} and whole batches of vectors go unread. That needs a bulk, threshold-aware
 * {@code ExactSearcher} which this branch does not have — {@link ExactSearcher#searchLeaf} returns {@link TopDocs}. So
 * every candidate in a surviving segment is scored, and the pruning is left on the table rather than faked. The candidate
 * set is already capped globally, so the cost is bounded by the first-pass k either way.
 *
 * <p>With {@code parentsFilter} set the searcher yields one best child per parent, which is what keeps a rescore from
 * undoing the dedup the scan's collector did: two children of one parent can both out-score another parent's best.
 */
final class ExactScoringWeight extends Weight {

    private final String field;
    private final float[] queryVector;
    private final PerLeafDocIds candidates;

    /** Set when the field is nested, {@code null} otherwise. */
    @Nullable
    private final BitSetProducer parentsFilter;

    private final ExactSearcher exactSearcher;
    private final float boost;

    /** Each segment's exact scores, kept after the first pass — Lucene may ask for a segment's scorer more than once. */
    private final Map<Integer, TopDocs> perLeafRescored = new ConcurrentHashMap<>();

    ExactScoringWeight(
        final Query query,
        final String field,
        final float[] queryVector,
        final PerLeafDocIds candidates,
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
     * @return {@code null} when this segment holds no candidate, so it is never opened.
     */
    @Override
    @Nullable
    public ScorerSupplier scorerSupplier(final LeafReaderContext context) {
        final int size = candidates.size(context);
        if (size == 0) {
            return null;
        }
        return new ScorerSupplier() {
            @Override
            public Scorer get(final long leadCost) throws IOException {
                final TopDocs rescored = rescore(context);
                if (rescored.scoreDocs.length == 0) {
                    return KNNScorer.emptyScorer();
                }
                // Boosted here rather than inside the exact searcher: the boost belongs to this query, and the searcher is
                // shared with callers that have none.
                return new KNNScorer(rescored, boost);
            }

            @Override
            public long cost() {
                return size;
            }
        };
    }

    /** One segment's candidates, scored against the vectors as written. Segment-local doc ids, as they went in. */
    private TopDocs rescore(final LeafReaderContext context) throws IOException {
        final TopDocs cached = perLeafRescored.get(context.ord);
        if (cached != null) {
            return cached;
        }
        final TopDocs results = exactSearcher.searchLeaf(context, searcherContext(context));
        perLeafRescored.put(context.ord, results);
        return results;
    }

    /**
     * {@code useQuantizedVectorsForSearch(false)} is what makes this exact rather than a repeat of the scan: it takes the
     * full-precision vectors, which is the only reason correcting the approximate ordering is worth anything.
     *
     * <p>{@code k} is the candidate count, so nothing is dropped here — reducing to the user's k is the collector's, and
     * with a parents filter the count of parents is not known until the children have been scored anyway.
     */
    private ExactSearcher.ExactSearcherContext searcherContext(final LeafReaderContext context) {
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
        if (candidates.size(context) == 0) {
            return Explanation.noMatch("no candidate for this query in this segment");
        }
        for (final ScoreDoc scoreDoc : rescore(context).scoreDocs) {
            if (scoreDoc.doc == doc) {
                return Explanation.match(scoreDoc.score * boost, "exact score against the stored full-precision vector");
            }
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
