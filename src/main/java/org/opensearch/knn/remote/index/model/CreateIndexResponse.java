/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.model;

import lombok.Builder;
import lombok.Value;
import org.opensearch.core.ParseField;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;

@Value
@Builder
public class CreateIndexResponse {
    private static final ParseField INDEX_CREATION_REQUEST_ID = new ParseField("indexCreationRequestId");
    private static final ParseField STATUS = new ParseField("status");
    String indexCreationRequestId;
    String status;

    public static CreateIndexResponse fromXContent(XContentParser parser) throws IOException {
        final CreateIndexResponseBuilder builder = new CreateIndexResponseBuilder();
        XContentParser.Token token = parser.nextToken();
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IOException("Invalid response format, was expecting a " + XContentParser.Token.START_OBJECT);
        }
        String currentFieldName = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (INDEX_CREATION_REQUEST_ID.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.indexCreationRequestId(parser.text());
                } else if (STATUS.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.status(parser.text());
                } else {
                    throw new IOException("Invalid response format, unknown field: " + currentFieldName);
                }
            }
        }
        return builder.build();
    }
}
