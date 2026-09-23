/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.KNNScorer;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Scans one segment per {@link #scorerSupplier}, rather than every segment up front.
 *
 * <p>Lazy because a segment's scan is only worth paying for when someone asks for that segment's hits, and because the
 * caller is better placed to parallelise than this class is: a rescorer already fans out per segment, so doing it here
 * as well would mean two fan-outs and an intermediate merge that the rescorer immediately takes apart again.
 *
 * <p>All the segments share one competitive threshold through the collector manager, which is what makes the
 * parallelism pay for itself — a segment that starts late learns the score it must beat from the ones already done, and
 * prunes from its first cluster instead of rediscovering the same bar.
 */
final class ClusterANNWeight extends Weight {

    private final String field;
    private final float[] queryVector;
    private final int candidateK;
    private final float boost;

    @Nullable
    private final Weight filterWeight;

    private final TimeLimitingKnnCollectorManager collectorManager;

    /**
     * Each segment's hits, kept after the first scan.
     *
     * <p>Lucene may ask for a segment's scorer more than once — {@code count()}, {@code explain()}, a caching layer —
     * and a scan is far too expensive to repeat for an answer that cannot have changed.
     */
    private final Map<Integer, TopDocs> perLeafResults = new ConcurrentHashMap<>();

    ClusterANNWeight(
        final Query query,
        final String field,
        final float[] queryVector,
        final int candidateK,
        @Nullable final Weight filterWeight,
        final TimeLimitingKnnCollectorManager collectorManager,
        final float boost
    ) {
        super(query);
        this.field = field;
        this.queryVector = queryVector;
        this.candidateK = candidateK;
        this.filterWeight = filterWeight;
        this.collectorManager = collectorManager;
        this.boost = boost;
    }

    @Override
    public ScorerSupplier scorerSupplier(final LeafReaderContext context) {
        return new ScorerSupplier() {
            @Override
            public Scorer get(final long leadCost) throws IOException {
                final TopDocs topDocs = searchLeaf(context);
                if (topDocs.scoreDocs.length == 0) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNScorer(topDocs, boost);
            }

            /**
             * Before the scan there is nothing to count, so this is what the scan is allowed to return rather than what
             * it will. Afterwards the real number is known and worth reporting, since a rescorer sizes its queue from
             * it.
             */
            @Override
            public long cost() {
                final TopDocs done = perLeafResults.get(context.ord);
                return done == null ? candidateK : done.scoreDocs.length;
            }
        };
    }

    /**
     * The approximate hits of one segment, in that segment's own doc space.
     *
     * <p>Doc ids are left segment-local. Whoever merges across segments shifts them, and doing it here would mean the
     * memoised results could not be handed out twice.
     */
    TopDocs searchLeaf(final LeafReaderContext context) throws IOException {
        final TopDocs cached = perLeafResults.get(context.ord);
        if (cached != null) {
            return cached;
        }

        final AcceptDocs acceptDocs = acceptDocs(context);
        TopDocs results = TopDocsCollector.EMPTY_TOPDOCS;
        if (acceptDocs.cost() > 0) {
            // TODO: short-circuit to exact search when the filter is selective enough that scanning postings cannot pay
            // for itself. The threshold belongs to the probed clusters' sizes, which the codec knows and this layer does
            // not, so it is left out rather than approximated with a bound that means nothing for a cluster scan.

            // No visit limit: nothing in the scan path reads earlyTerminated() yet, so a limit here would be quietly
            // ignored. Said outright rather than passed as a number that does nothing.
            // Null for a nested field's segment that holds no parents: there are no children to score there either, so
            // this is "nothing here", not "nothing matched".
            final KnnCollector collector = collectorManager.newCollector(Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT, context);
            if (collector != null) {
                context.reader().searchNearestVectors(field, queryVector, collector, acceptDocs);
                results = collector.topDocs();
            }
        }

        perLeafResults.put(context.ord, results);
        return results;
    }

    /**
     * What this segment may return: the filter's matches if there is one, otherwise its live documents.
     *
     * <p>With no filter and no deletions the bits come back {@code null} — Lucene's own way of saying "everything" —
     * which is what lets the codec skip the per-vector membership test entirely instead of probing a set that would
     * answer true every time.
     */
    private AcceptDocs acceptDocs(final LeafReaderContext context) throws IOException {
        if (filterWeight == null) {
            return AcceptDocs.fromLiveDocs(context.reader().getLiveDocs(), context.reader().maxDoc());
        }
        return AcceptDocs.fromIteratorSupplier(() -> {
            final Scorer scorer = filterWeight.scorer(context);
            return scorer == null ? DocIdSetIterator.empty() : scorer.iterator();
        }, context.reader().getLiveDocs(), context.reader().maxDoc());
    }

    @Override
    public Explanation explain(final LeafReaderContext context, final int doc) throws IOException {
        final TopDocs topDocs = searchLeaf(context);
        for (final var scoreDoc : topDocs.scoreDocs) {
            if (scoreDoc.doc == doc) {
                return Explanation.match(scoreDoc.score * boost, "within top " + candidateK + " of the ClusterANN scan");
            }
        }
        return Explanation.noMatch("not among the top " + candidateK + " of the ClusterANN scan");
    }

    /**
     * Never cacheable.
     *
     * <p>A cache entry is keyed on the query, and a query vector is effectively unique — so caching would never hit,
     * while filling the cache costs an extra call into {@link #scorerSupplier} and therefore an extra segment scan.
     */
    @Override
    public boolean isCacheable(final LeafReaderContext context) {
        return false;
    }
}
