/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Returns every child of the parents its inner query settled on, each scored on its own.
 *
 * <p>The stages before this one answer "which parents", by picking each parent's nearest child. This one answers "and
 * what is in them", which is a different question and needs a second pass over different documents: the siblings of those
 * winning children, deduped by nothing, because every child is wanted.
 *
 * <h2>The last stage, so it does not collect</h2>
 *
 * <pre>
 * inner query   → top-k children, one per parent   ← collected here, because the nests come from them
 * this query    → a Scorer over their siblings     ← Lucene collects
 * </pre>
 *
 * <p>Handing back a scorer rather than results is what keeps the expansion affordable. A nest can be large, and every
 * sibling of every winning parent is a candidate — but the caller usually wants far fewer than that. As Lucene's collector
 * fills, it pushes its competitive score back into the scorer, which then skips whole batches of siblings without reading
 * a vector in them. Collecting every sibling and merging them, which is what nested expansion has done until now, throws
 * that away: ten parents with thousand-child nests is ten thousand documents materialised so that a handful can be kept.
 *
 * <p><b>The inner query still has to complete.</b> A nest is derived from a surviving child, and which children survive is
 * decided globally — so this stage cannot begin until the one before it has finished and merged. That barrier is the
 * reason this is a separate query rather than a mode of the one it wraps.
 */
@Log4j2
public class ClusterANNExpandQuery extends Query {

    private final Query innerQuery;
    private final String field;

    /** Parents to expand — as many as the inner query was asked for, each represented by its nearest child. */
    private final int k;

    private final float[] queryVector;
    private final BitSetProducer parentsFilter;

    /**
     * The query's filter, needed again here rather than inherited.
     *
     * <p>The inner stages used it to decide which documents could be candidates; this one uses it to decide which
     * siblings belong in the answer. Same filter, a different question, so it is applied twice.
     */
    @Nullable
    private final Query filter;

    private final ExactSearcher exactSearcher;
    private final QueryUtils queryUtils;

    public ClusterANNExpandQuery(
        final Query innerQuery,
        final String field,
        final int k,
        final float[] queryVector,
        final BitSetProducer parentsFilter,
        @Nullable final Query filter
    ) {
        this(
            innerQuery,
            field,
            k,
            queryVector,
            parentsFilter,
            filter,
            new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance()),
            QueryUtils.getInstance()
        );
    }

    ClusterANNExpandQuery(
        final Query innerQuery,
        final String field,
        final int k,
        final float[] queryVector,
        final BitSetProducer parentsFilter,
        @Nullable final Query filter,
        final ExactSearcher exactSearcher,
        final QueryUtils queryUtils
    ) {
        this.innerQuery = innerQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.parentsFilter = Objects.requireNonNull(parentsFilter, "expanding needs a parents filter to find a nest");
        this.filter = filter;
        this.exactSearcher = exactSearcher;
        this.queryUtils = queryUtils;
    }

    /** A no-op, like every stage's: the work waits for {@link #createWeight}, so a second rewrite costs nothing. */
    @Override
    public Query rewrite(final IndexSearcher indexSearcher) {
        return this;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        final Query rewritten = searcher.rewrite(innerQuery);
        final Weight innerWeight = searcher.createWeight(rewritten, ScoreMode.COMPLETE, 1.0f);
        final List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();

        final TopDocs winners = LeafHits.collect(searcher, leaves, innerWeight, k);
        if (winners.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }

        // No parents filter on the exact pass: the stage before used one to keep a single child per parent, and undoing
        // that is the entire point of expanding.
        return new ExactScoringWeight(
            this,
            field,
            queryVector,
            new SiblingCandidates(LeafHits.groupByLeaf(winners, leaves), parentsFilter, filterWeight(searcher), queryUtils),
            null,
            exactSearcher,
            boost
        );
    }

    /**
     * The filter, conjoined with "this field has a vector", so a document without one is never mistaken for a sibling
     * worth scoring.
     */
    @Nullable
    private Weight filterWeight(final IndexSearcher searcher) throws IOException {
        if (filter == null) {
            return null;
        }
        final BooleanQuery booleanQuery = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
            .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
            .build();
        final Query rewritten = searcher.rewrite(booleanQuery);
        return searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(final String f) {
        return getClass().getSimpleName() + "[innerQuery=" + innerQuery + ", field=" + field + ", k=" + k + "]";
    }

    @Override
    public boolean equals(final Object other) {
        if (!sameClassAs(other)) {
            return false;
        }
        final ClusterANNExpandQuery o = (ClusterANNExpandQuery) other;
        return k == o.k
            && Objects.equals(innerQuery, o.innerQuery)
            && Objects.equals(field, o.field)
            && Arrays.equals(queryVector, o.queryVector)
            && Objects.equals(parentsFilter, o.parentsFilter)
            && Objects.equals(filter, o.filter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(innerQuery, field, k, Arrays.hashCode(queryVector), parentsFilter, filter);
    }
}
