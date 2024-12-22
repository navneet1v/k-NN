/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.model;

import lombok.Builder;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.opensearch.core.ParseField;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;

/**
 * Sample response:
 *
 * {
 * 	"error": null,
 * 	"result": {
 * 		"bucketName": "remote-knn-index-build-navneet",
 * 		"graphFileLocation": "xypqqYqLS1urE9UFS9x2Ww__0_location.s3vec.faiss.cpu",
 * 		"stats": {
 * 			"create_index": {
 * 				"indexTime": 0.0005696250009350479,
 * 				"totalTime": 0.0014507080013572704,
 * 				"unit": "seconds",
 * 				"writeIndexTime": 0.0008810830004222225
 * 			            },
 * 			"download_stats": {
 * 				"time": 0.22056070799953886,
 * 				"unit": "seconds"
 *            },
 * 			"upload_stats": {
 * 				"time": 0.41053095899951586,
 * 				"unit": "seconds"
 *            }        * 		}
 * 	},
 * 	"status": "completed"
 * }
 */
@Value
@Builder
@Log4j2
public class GetJobResponse {
    private static final ParseField STATUS = new ParseField("status");
    private static final ParseField RESULT = new ParseField("result");
    String status;
    Result result;

    public static GetJobResponse fromXContent(XContentParser parser) throws IOException {
        final GetJobResponseBuilder builder = new GetJobResponseBuilder();
        XContentParser.Token token = parser.nextToken();
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IOException("Invalid response format, was expecting a " + XContentParser.Token.START_OBJECT);
        }
        String currentFieldName = null;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (STATUS.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.status(parser.text());
                }
            } else if (token == XContentParser.Token.START_OBJECT) {
                if (RESULT.match(currentFieldName, parser.getDeprecationHandler())) {
                    builder.result(Result.fromXContent(parser));
                }
            }
        }
        return builder.build();
    }

    @Value
    @Builder
    public static class Result {
        String bucketName;
        String graphFileLocation;
        private static final ParseField BUCKET_NAME = new ParseField("bucketName");
        private static final ParseField GRAPH_FILE_LOCATION = new ParseField("graphFileLocation");
        private static final ParseField STATS = new ParseField("stats");

        public static Result fromXContent(XContentParser parser) throws IOException {
            final ResultBuilder builder = new ResultBuilder();
            XContentParser.Token token = null;
            String currentFieldName = null;
            while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
                if (token == XContentParser.Token.FIELD_NAME) {
                    currentFieldName = parser.currentName();
                    if (STATS.match(currentFieldName, parser.getDeprecationHandler())) {
                        // as we already know that in stats there are 3 object so lets skip them
                        parser.skipChildren();
                    }
                } else if (token.isValue()) {
                    if (BUCKET_NAME.match(currentFieldName, parser.getDeprecationHandler())) {
                        builder.bucketName(parser.text());
                    } else if (GRAPH_FILE_LOCATION.match(currentFieldName, parser.getDeprecationHandler())) {
                        builder.graphFileLocation(parser.text());
                    }
                } else {
                    // skipping values we don't want to parse
                    if (token == XContentParser.Token.START_OBJECT || token == XContentParser.Token.START_ARRAY) {
                        parser.skipChildren();
                    }
                }
            }
            return builder.build();
        }
    }
}
