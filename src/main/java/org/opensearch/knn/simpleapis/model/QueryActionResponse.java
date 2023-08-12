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

package org.opensearch.knn.simpleapis.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.opensearch.action.support.broadcast.BroadcastResponse;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

@NoArgsConstructor
@AllArgsConstructor
public class QueryActionResponse extends BroadcastResponse implements ToXContentObject {

    @Getter
    @Setter
    private QueryResponse queryResponse;

    private long requestStartTimeInNano;

    private long initialNetworkDelay;

    public QueryActionResponse(StreamInput streamInput) throws IOException {
        super(streamInput);
    }

    /**
     * Write this into the {@linkplain StreamOutput}.
     *
     * @param out
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.startArray("hits");
        for (SimpleQueryResults queryResults : queryResponse.getSimpleQueryResults()) {
            for (SimpleQueryResults.SimpleQueryResult result : queryResults.getQueryResultList()) {
                result.toXContent(builder, params);
            }
        }
        builder.endArray();
        builder.field("timeInNano", System.nanoTime() - requestStartTimeInNano);
        builder.field("initialNetworkDelayInMillis", initialNetworkDelay);
        builder.endObject();
        return builder;
    }
}
