/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.Query;

public class ExactSearchQueryFactory extends BaseQueryFactory {

    public static Query create(final CreateQueryRequest createQueryRequest) {
        final ExactSearchQuery.ExactSearchQueryBuilder exactSearchQueryBuilder = ExactSearchQuery.builder()
             .k(createQueryRequest.getK())
            .indexName(createQueryRequest.getIndexName())
            .queryVector(createQueryRequest.getVector())
            .field(createQueryRequest.getFieldName())
            .filterQuery(getFilterQuery(createQueryRequest))
            .parentsFilter(getParentFilter(createQueryRequest))
            .vectorDataType(createQueryRequest.getVectorDataType());
        return exactSearchQueryBuilder.build();
    }

}
