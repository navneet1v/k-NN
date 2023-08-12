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

package org.opensearch.knn.simpleapis.rest;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.common.xcontent.XContentParserUtils;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.simpleapis.model.QueryActionRequest;
import org.opensearch.knn.simpleapis.model.QueryRequest;
import org.opensearch.rest.RestRequest;

import java.io.IOException;
import java.util.List;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class SimpleQueryHandlerHelper {
    public static QueryActionRequest createQueryRequest(final RestRequest request, long startTime) throws IOException {
        final QueryRequest.QueryRequestBuilder queryRequestBuilder = QueryRequest.builder();
        queryRequestBuilder.indexName(request.param("index"));
        XContentParser parser = request.contentParser();
        XContentParserUtils.ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        long initialNetworkDelay = -1;
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();
            if ("k".equals(fieldName)) {
                queryRequestBuilder.k(parser.intValue());
            } else if ("vector".equals(fieldName)) {
                final List<Object> vector = parser.list();
                float[] floatArray = new float[vector.size()];
                for (int i = 0; i < vector.size(); i++) {
                    floatArray[i] = ((Double) vector.get(i)).floatValue();
                }
                queryRequestBuilder.vector(floatArray);
            } else if ("vectorFieldName".equals(fieldName)) {
                queryRequestBuilder.vectorFieldName(parser.text());
            } else if ("initialNetworkDelay".equals(fieldName)) {
                initialNetworkDelay = System.currentTimeMillis() - parser.longValue();
            }
        }
        QueryActionRequest queryActionRequest = new QueryActionRequest(queryRequestBuilder.build());
        queryActionRequest.setStartTimeInNano(startTime);
        queryActionRequest.setInitialNetworkDelay(initialNetworkDelay);
        return queryActionRequest;
    }
}
