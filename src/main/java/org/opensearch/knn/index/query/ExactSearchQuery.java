/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Builder;
import lombok.Getter;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;

@Builder
@Getter
public class ExactSearchQuery extends Query {

    private String field;
    private float[] queryVector;
    private byte[] byteQueryVector;
    private int k;
    private String indexName;
    private VectorDataType vectorDataType;

    private Query filterQuery;
    private BitSetProducer parentsFilter;

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        if (!KNNSettings.isKNNPluginEnabled()) {
            throw new IllegalStateException("KNN plugin is disabled. To enable update knn.plugin.enabled to true");
        }
        final Weight filterWeight = getFilterWeight(searcher);
        if (filterWeight != null) {
            return new ExactKNNWeight(this, boost, filterWeight);
        }
        return new ExactKNNWeight(this, boost);
    }

    // this is duplicate
    private Weight getFilterWeight(IndexSearcher searcher) throws IOException {
        if (this.getFilterQuery() != null) {
            // Run the filter query
            final BooleanQuery booleanQuery = new BooleanQuery.Builder().add(this.getFilterQuery(), BooleanClause.Occur.FILTER)
                .add(new FieldExistsQuery(this.getField()), BooleanClause.Occur.FILTER)
                .build();
            final Query rewritten = searcher.rewrite(booleanQuery);
            return searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
        }
        return null;
    }

    /**
     * {@inheritDoc}
     *
     * @param field
     */
    @Override
    public String toString(String field) {
        return "";
    }

    /**
     * {@inheritDoc}
     *
     * @param visitor a QueryVisitor to be called by each query in the tree
     */
    @Override
    public void visit(QueryVisitor visitor) {}

    /**
     * {@inheritDoc}
     *
     * @param obj
     * @see #sameClassAs(Object)
     * @see #classHash()
     */
    @Override
    public boolean equals(Object obj) {
        return false;
    }

    /**
     * {@inheritDoc}
     *
     * @see #equals(Object)
     */
    @Override
    public int hashCode() {
        return 0;
    }
}
