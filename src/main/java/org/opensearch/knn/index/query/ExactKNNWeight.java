/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

public class ExactKNNWeight extends Weight {

    private final ExactSearchQuery query;
    private final float boost;
    private final Weight filterWeight;
    private final ExactSearcher exactSearcher;

    public ExactKNNWeight(final ExactSearchQuery query, float boost, final Weight filterWeight) {
        super(query);
        this.boost = boost;
        this.filterWeight = filterWeight;
        this.query = query;
        this.exactSearcher = ExactSearcher.getInstance();
    }

    public ExactKNNWeight(final ExactSearchQuery query, final float boost) {
        this(query, boost, null);
    }

    /**
     * {@inheritDoc}
     *
     * @param context the readers context to create the {@link Explanation} for.
     * @param doc     the document's id relative to the given context's reader
     * @return an Explanation for the score
     * @throws IOException if an {@link IOException} occurs
     */
    @Override
    public Explanation explain(LeafReaderContext context, int doc) throws IOException {
        return Explanation.match(1.0f, "No Explanation");
    }

    /**
     * Returns a {@link Scorer} which can iterate in order over all matching documents and assign them
     * a score.
     *
     * <p><b>NOTE:</b> null can be returned if no documents will be scored by this query.
     *
     * <p><b>NOTE</b>: The returned {@link Scorer} does not have {@link LeafReader#getLiveDocs()}
     * applied, they need to be checked on top.
     *
     * @param context the {@link LeafReaderContext} for which to return the
     *                {@link Scorer}.
     * @return a {@link Scorer} which scores documents in/out-of order.
     * @throws IOException if there is a low-level I/O error
     */
    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
        final BitSet matchedDocsBitset;
        if (filterWeight != null) {
            matchedDocsBitset = getFilteredDocsBitSet(context);
        } else {
            final LeafReader leafReader = context.reader();
            final int maxDoc = leafReader.maxDoc();
            final Bits liveDocs = leafReader.getLiveDocs();
            final DocIdSetIterator matchAllIterator = DocIdSetIterator.all(maxDoc);
            final DocIdSetIterator liveDocsIterator = new FilteredDocIdSetIterator(matchAllIterator) {
                @Override
                protected boolean match(int doc) throws IOException {
                    return liveDocs == null || liveDocs.get(doc);
                }
            };
            matchedDocsBitset = BitSet.of(liveDocsIterator, maxDoc);
        }
        final Map<Integer, Float> docIdToScoreMap = exactSearcher.searchLeaf(
            context,
            matchedDocsBitset,
            ExactSearcher.ExactSearcherContext.buildExactSearcherContextFromExactSearchQuery(query),
            query.getK(),
            true
        );
        if (docIdToScoreMap.isEmpty()) {
            return KNNScorer.emptyScorer(this);
        }
        final int maxDoc = Collections.max(docIdToScoreMap.keySet()) + 1;
        return new KNNScorer(this, ResultUtil.resultMapToDocIds(docIdToScoreMap, maxDoc), docIdToScoreMap, boost);
    }

    private BitSet getFilteredDocsBitSet(final LeafReaderContext ctx) throws IOException {
        if (this.filterWeight == null) {
            return new FixedBitSet(0);
        }

        final Bits liveDocs = ctx.reader().getLiveDocs();
        final int maxDoc = ctx.reader().maxDoc();

        final Scorer scorer = filterWeight.scorer(ctx);
        if (scorer == null) {
            return new FixedBitSet(0);
        }

        return createBitSet(scorer.iterator(), liveDocs, maxDoc);
    }

    private BitSet createBitSet(final DocIdSetIterator filteredDocIdsIterator, final Bits liveDocs, int maxDoc) throws IOException {
        if (liveDocs == null && filteredDocIdsIterator instanceof BitSetIterator) {
            // If we already have a BitSet and no deletions, reuse the BitSet
            return ((BitSetIterator) filteredDocIdsIterator).getBitSet();
        }
        // Create a new BitSet from matching and live docs
        FilteredDocIdSetIterator filterIterator = new FilteredDocIdSetIterator(filteredDocIdsIterator) {
            @Override
            protected boolean match(int doc) {
                return liveDocs == null || liveDocs.get(doc);
            }
        };
        return BitSet.of(filterIterator, maxDoc);
    }

    /**
     * @param ctx {@link LeafReaderContext}
     * @return {@code true} if the object can be cached against a given leaf
     */
    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
        return true;
    }
}
