/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Rescores an approximate result against the stored full-precision vectors.
 *
 * <p>The scan ranks by quantized score, which is an estimate — good enough to find the right neighbourhood, not to order
 * it. Scoring the survivors against the vectors as written recovers the ordering.
 *
 * <h2>One phase here; the second is Lucene's</h2>
 *
 * <pre>
 * 1. approximate   every segment, in parallel  → merge globally to firstPassK   ← this class
 * 2. rescore       a Scorer per surviving segment, exact scores                 ← Lucene collects
 * </pre>
 *
 * <p><b>The exact pass returns a scorer rather than results.</b> Lucene's collector does the top-k, and in return feeds
 * its competitive threshold back into the scorer through {@code setMinCompetitiveScore} — so a batch whose best possible
 * score is already below the bar is skipped without a single vector in it being read. Collecting here would forfeit
 * that, because only the collector knows what the bar is at any moment.
 *
 * <p><b>The cap is global, and that is the whole reason for two phases.</b> {@code firstPassK} is how many candidates the
 * <em>index</em> contributes, not how many each segment does — so it can only be applied once every segment has reported,
 * which means phase 2 cannot begin until phase 1 is complete. Rescoring each segment's own candidates as they arrive
 * would be one pass instead of two, but it would rescore {@code segments × firstPassK} vectors and make the amount of
 * exact work depend on how the index happens to be segmented.
 *
 * <p>What the global cap buys beyond predictability: after the merge, most segments usually hold no surviving candidate
 * at all, and those are skipped outright in phase 2. The exact scoring concentrates where the neighbours actually are.
 *
 * <p>The exact scoring itself is {@code ExactSearcher}'s, asked for as a scorer rather than as results, so the batched
 * vector reads and the threshold-driven batch skipping are the ones already written rather than a second copy.
 */
@Log4j2
public class ClusterANNRescoreQuery extends Query {

    private final Query innerQuery;
    private final String field;

    /** Candidates the whole index contributes to the rescore. */
    private final int firstPassK;

    /** Hits returned after rescoring. */
    private final int k;

    private final float[] queryVector;

    /**
     * Set when the field is nested, so the exact pass keeps one best child per parent. Without it the dedup the scan's
     * collector did would be undone here: two children of one parent can both out-score another parent's best.
     */
    @Nullable
    private final BitSetProducer parentsFilter;

    private final ExactSearcher exactSearcher;

    public ClusterANNRescoreQuery(
        final Query innerQuery,
        final String field,
        final int firstPassK,
        final int k,
        final float[] queryVector,
        @Nullable final BitSetProducer parentsFilter
    ) {
        this(
            innerQuery,
            field,
            firstPassK,
            k,
            queryVector,
            parentsFilter,
            new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance())
        );
    }

    ClusterANNRescoreQuery(
        final Query innerQuery,
        final String field,
        final int firstPassK,
        final int k,
        final float[] queryVector,
        @Nullable final BitSetProducer parentsFilter,
        final ExactSearcher exactSearcher
    ) {
        if (firstPassK < k) {
            throw new IllegalArgumentException("firstPassK=" + firstPassK + " must be at least k=" + k);
        }
        this.innerQuery = innerQuery;
        this.field = field;
        this.firstPassK = firstPassK;
        this.k = k;
        this.queryVector = queryVector;
        this.parentsFilter = parentsFilter;
        this.exactSearcher = exactSearcher;
    }

    /** A no-op, like the inner query's: the work waits for {@link #createWeight}, so a second rewrite costs nothing. */
    @Override
    public Query rewrite(final IndexSearcher indexSearcher) {
        return this;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        final Query rewritten = searcher.rewrite(innerQuery);
        final Weight innerWeight = searcher.createWeight(rewritten, ScoreMode.COMPLETE, 1.0f);
        final List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();

        final TopDocs candidates = LeafHits.collect(searcher, leaves, innerWeight, firstPassK);
        if (candidates.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }

        // The candidate set is fixed from here: capped globally and split back per segment. What remains is scoring it,
        // and that is left to whoever collects.
        return new ExactScoringWeight(
            this,
            field,
            queryVector,
            new PerLeafDocIds(LeafHits.groupByLeaf(candidates, leaves)),
            parentsFilter,
            exactSearcher,
            boost
        );
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(final String f) {
        return getClass().getSimpleName()
            + "[innerQuery="
            + innerQuery
            + ", field="
            + field
            + ", firstPassK="
            + firstPassK
            + ", k="
            + k
            + "]";
    }

    @Override
    public boolean equals(final Object other) {
        if (!sameClassAs(other)) {
            return false;
        }
        final ClusterANNRescoreQuery o = (ClusterANNRescoreQuery) other;
        return k == o.k
            && firstPassK == o.firstPassK
            && Objects.equals(innerQuery, o.innerQuery)
            && Objects.equals(field, o.field)
            && Arrays.equals(queryVector, o.queryVector)
            && Objects.equals(parentsFilter, o.parentsFilter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(innerQuery, field, firstPassK, k, Arrays.hashCode(queryVector), parentsFilter);
    }
}
