/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import lombok.Setter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;

/**
 * A {@link BulkScorer} that uses {@link ExactSearcher} (backed by Lucene's {@link VectorScorer.Bulk})
 * to batch-prefetch and score vector documents, reducing I/O from one random read per document to
 * one grouped prefetch per batch of 64.
 *
 * <h2>Scoring flow</h2>
 * <ol>
 *   <li>Collect all matching docIDs from the sub-query's BulkScorer into a {@link FixedBitSet}.</li>
 *   <li>Pass those docIDs to {@link ExactSearcher#exactSearchScorer}, which internally creates a
 *       {@link VectorScorer.Bulk} that prefetches vector data in batches before computing SIMD
 *       distances.</li>
 *   <li>Iterate the scored results, apply {@code boost}, and emit to the collector.</li>
 *   <li>Emit any matched docs that lack vectors with a score of {@code 0.0f}, preserving parity
 *       with the per-document {@link KNNScoreScript} path.</li>
 * </ol>
 *
 * <h2>Fallback behavior</h2>
 * If the segment has no vector data for the target field ({@code exactSearchScorer} returns null),
 * all matched docs are emitted with score {@code 0.0f}.
 */
class KNNScriptScoreBulkScorer extends BulkScorer {

    private final BulkScorer subQueryBulkScorer;
    private final float boost;
    private final long cost;
    private final ExactSearcher exactSearcher;
    private final LeafReaderContext context;
    private final KNNScoreScriptLeafFactory.KNNScoreScriptBulkScorerContext knnScoreScriptBulkScorerContext;

    /**
     * @param context                        the leaf reader context for this segment
     * @param subQueryBulkScorer             the sub-query's BulkScorer that identifies matching documents
     * @param boost                          the boost factor to apply to all scores
     * @param knnScoreScriptBulkScorerContext encapsulates the query vector, field name, and space type
     */
    KNNScriptScoreBulkScorer(
        final LeafReaderContext context,
        final BulkScorer subQueryBulkScorer,
        final float boost,
        final KNNScoreScriptLeafFactory.KNNScoreScriptBulkScorerContext knnScoreScriptBulkScorerContext
    ) {
        this.subQueryBulkScorer = subQueryBulkScorer;
        this.boost = boost;
        this.cost = this.subQueryBulkScorer.cost();
        // make exactSearcher singleton.
        this.exactSearcher = new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
        this.knnScoreScriptBulkScorerContext = knnScoreScriptBulkScorerContext;
        this.context = context;
    }

    /**
     * Scores documents in the range {@code [min, max)} by:
     * <ol>
     *   <li>Collecting matched docIDs from the sub-query into a bitset</li>
     *   <li>Batch-scoring vector docs via {@link ExactSearcher} (prefetch + SIMD distance)</li>
     *   <li>Emitting docs without vectors with score {@code 0.0f}</li>
     * </ol>
     *
     * @return an estimate of the next matching doc at or after {@code max},
     *         or {@link DocIdSetIterator#NO_MORE_DOCS} if none remain
     */
    @Override
    public int score(final LeafCollector collector, final Bits acceptDocs, int min, int max) throws IOException {
        final BitSet bitSet = new FixedBitSet(max);
        // Collect the docIds by running the bulkScorer of the Subquery.
        final int nextDoc = subQueryBulkScorer.score(new LeafCollector() {
            @Override
            public void setScorer(Scorable scorer) {}

            @Override
            public void collect(int doc) {
                bitSet.set(doc);
            }
        }, acceptDocs, min, max);

        final DocIdSetIterator docIdSetIterator = new BitSetIterator(bitSet, bitSet.cardinality());
        final ExactSearcher.ExactSearcherContext.ExactSearcherContextBuilder exactSearcherContextBuilder =
            ExactSearcher.ExactSearcherContext.builder()
                .field(knnScoreScriptBulkScorerContext.fieldName())
                .matchedDocsIterator(docIdSetIterator)
                .spaceType(knnScoreScriptBulkScorerContext.spaceType());

        if (knnScoreScriptBulkScorerContext.queryVector() instanceof float[] floatQueryVector) {
            exactSearcherContextBuilder.floatQueryVector(floatQueryVector);
        } else if (knnScoreScriptBulkScorerContext.queryVector() instanceof byte[] byteQueryVector) {
            exactSearcherContextBuilder.byteQueryVector(byteQueryVector);
        } else {
            throw new IllegalArgumentException("Query Vector should be a float[] or byte[]");
        }

        final Scorer exactSearchScorer = exactSearcher.exactSearchScorer(context, exactSearcherContextBuilder.build());
        // This Scorable is the placeholder for storing the scores
        final ExactSearchScorable randomScorable = new ExactSearchScorable();
        collector.setScorer(randomScorable);
        // No vectors in this segment — emit all matched docs with score 0.0
        if (exactSearchScorer == null) {
            randomScorable.setScore(0.0f);
            for (int docId = bitSet.nextSetBit(0); docId != DocIdSetIterator.NO_MORE_DOCS; docId = bitSet.nextSetBit(docId + 1)) {
                collector.collect(docId);
            }
        } else {
            final DocIdSetIterator scoredDocIds = exactSearchScorer.iterator();
            for (int docId = scoredDocIds.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = scoredDocIds.nextDoc()) {
                randomScorable.setScore(exactSearchScorer.score() * boost);
                collector.collect(docId);
                bitSet.clear(docId);
            }
            // Remaining set bits are docs without vectors — emit with score 0.0
            randomScorable.setScore(0.0f);
            for (int docId = bitSet.nextSetBit(0); docId != DocIdSetIterator.NO_MORE_DOCS; docId = bitSet.nextSetBit(docId + 1)) {
                collector.collect(docId);
            }
        }
        return nextDoc;
    }

    @Override
    public long cost() {
        return cost;
    }

    private static final class ExactSearchScorable extends Scorable {
        @Setter
        private float score;

        @Override
        public float score() {
            return score;
        }
    }
}
