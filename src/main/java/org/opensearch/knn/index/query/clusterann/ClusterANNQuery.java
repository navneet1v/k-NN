/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.common.Nullable;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Searches a ClusterANN field, one segment at a time, through the codec's own reader.
 *
 * <p><b>Not a {@code KnnFloatVectorQuery} subclass, on purpose.</b> Lucene's {@code AbstractKnnVectorQuery} brings four
 * behaviours, of which this wants one and wants it differently:
 *
 * <ul>
 *   <li>an optimistic second pass re-entering promising segments — sound when the cost is graph nodes visited, wrong
 *       when the cost is postings read, because re-entering a segment means re-reading them
 *   <li>{@code perLeafTopK}, scaling k by a segment's share of the index — worth borrowing later, but as a knob rather
 *       than as an inherited default
 *   <li>a fall back to exact search when the graph "visited too many nodes", which has no analogue in a cluster scan
 *   <li>a short-circuit to exact search when the filter is selective — the one worth keeping, but its threshold should
 *       come from the probed clusters' sizes, not from a graph's minimum work
 * </ul>
 *
 * Inheriting all four and overriding three away reads as a list of things switched off; the loop written out says what
 * it does and is no longer.
 *
 * <p><b>Nothing is searched until a scorer is asked for.</b> OpenSearch rewrites a query again during the fetch phase,
 * so work done in {@link #rewrite} is paid for twice — hence {@code rewrite} returns {@code this}. And a rescorer
 * already fans out per segment, so searching every segment in {@link #createWeight} would mean two fan-outs and an
 * intermediate merge the rescorer immediately takes apart.
 *
 * <p>{@code candidateK} is what each segment collects. When a rescore follows it is already oversampled, and reducing
 * to the user's k stays the rescorer's business — trimming here would discard the candidates it was given extra of.
 */
@Log4j2
public class ClusterANNQuery extends Query {

    private final String field;
    private final float[] queryVector;
    private final int candidateK;

    @Nullable
    private final Query filter;

    /**
     * Identifies parent documents when the field is nested, {@code null} otherwise.
     *
     * <p>Its only effect here is on what a segment keeps: with parents, one best child each. Everything about the scan is
     * unchanged, because a nested field's vectors are ordinary child documents to the codec.
     */
    @Nullable
    private final BitSetProducer parentsFilter;

    /** Only for logging; a query has no business knowing which shard it is on. */
    private final int shardId;

    public ClusterANNQuery(
        final String field,
        final float[] queryVector,
        final int candidateK,
        @Nullable final Query filter,
        @Nullable final BitSetProducer parentsFilter,
        final int shardId
    ) {
        this.field = field;
        this.queryVector = queryVector;
        this.candidateK = candidateK;
        this.filter = filter;
        this.parentsFilter = parentsFilter;
        this.shardId = shardId;
    }

    /**
     * Deliberately a no-op. The scan happens when a scorer is asked for, so however many times this is rewritten it
     * costs nothing.
     */
    @Override
    public Query rewrite(final IndexSearcher indexSearcher) {
        return this;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        // One manager for the whole query, so every segment's collector shares a competitive threshold. Wrapped for
        // timeouts, which is how a cancelled query stops mid-scan rather than running to completion.
        final TimeLimitingKnnCollectorManager collectorManager = new TimeLimitingKnnCollectorManager(
            new ClusterANNCollectorManager(candidateK, parentsFilter, searcher),
            searcher.getTimeout()
        );
        return new ClusterANNWeight(this, field, queryVector, candidateK, filterWeight(searcher), collectorManager, boost);
    }

    /**
     * The filter, conjoined with "this field has a vector".
     *
     * <p>The {@link FieldExistsQuery} is about the count, not the matching: without it {@code cost()} includes documents
     * that hold no vector at all, and every decision taken from that count is measured against the wrong denominator.
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

    String getField() {
        return field;
    }

    @Nullable
    BitSetProducer getParentsFilter() {
        return parentsFilter;
    }

    float[] getQueryVector() {
        return queryVector;
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(final String f) {
        return getClass().getSimpleName()
            + "[field="
            + field
            + ", candidateK="
            + candidateK
            + ", filter="
            + filter
            + ", shardId="
            + shardId
            + "]";
    }

    @Override
    public boolean equals(final Object other) {
        if (!sameClassAs(other)) {
            return false;
        }
        final ClusterANNQuery o = (ClusterANNQuery) other;
        return candidateK == o.candidateK
            && Objects.equals(field, o.field)
            && Arrays.equals(queryVector, o.queryVector)
            && Objects.equals(filter, o.filter)
            && Objects.equals(parentsFilter, o.parentsFilter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, Arrays.hashCode(queryVector), candidateK, filter, parentsFilter);
    }
}
