/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.simpleapis.query;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.fieldvisitor.IdOnlyFieldVisitor;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.simpleapis.model.QueryRequest;
import org.opensearch.knn.simpleapis.model.SimpleQueryResults;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Log4j2
@AllArgsConstructor
@Getter
public class KNNIndexQueryShard {

    private IndexShard indexShard;

    public String getIndexName() {
        return indexShard.shardId().getIndexName();
    }

    public List<SimpleQueryResults.SimpleQueryResult> query(QueryRequest queryRequest) {
        KNNQuery query = new KNNQuery(
            queryRequest.getVectorFieldName(),
            queryRequest.getVector(),
            queryRequest.getK(),
            queryRequest.getIndexName()
        );
        List<SimpleQueryResults.SimpleQueryResult> actualDocIds = new ArrayList<>();
        try (Engine.Searcher searcher = indexShard.acquireSearcher("knn-simple-query")) {
            for (LeafReaderContext context : searcher.getLeafContexts()) {
                KNNWeight knnWeight = new KNNWeight(query, 0);
                Scorer scorer = knnWeight.scorer(context);
                final DocIdSetIterator docIdSetIterator = scorer.iterator();
                final IdOnlyFieldVisitor idOnlyFieldVisitor = new IdOnlyFieldVisitor();
                while (docIdSetIterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                    context.reader().storedFields().document(scorer.docID(), idOnlyFieldVisitor);
                    actualDocIds.add(new SimpleQueryResults.SimpleQueryResult(idOnlyFieldVisitor.getId(), scorer.score()));
                    idOnlyFieldVisitor.reset();
                }
            }
        } catch (IOException e) {
            log.error("Error while doing the query {}", queryRequest, e);
            throw new RuntimeException(e);
        }
        return actualDocIds;
    }

}
