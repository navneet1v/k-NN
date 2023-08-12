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

import lombok.Getter;
import lombok.Setter;
import org.opensearch.action.support.broadcast.BroadcastRequest;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

public class QueryActionRequest extends BroadcastRequest<QueryActionRequest> {
    @Getter
    private final QueryRequest queryRequest;

    @Getter
    @Setter
    private long startTimeInNano;

    @Getter
    @Setter
    private long initialNetworkDelay;

    public QueryActionRequest(final QueryRequest queryRequest) {
        super(queryRequest.getIndexName());
        this.queryRequest = queryRequest;
    }

    public QueryActionRequest(final QueryRequest queryRequest, long startTimeInNano) {
        super(queryRequest.getIndexName());
        this.queryRequest = queryRequest;
        this.startTimeInNano = startTimeInNano;
    }

    public QueryActionRequest(StreamInput in) throws IOException {
        super(in);
        queryRequest = in.readNamedWriteable(QueryRequest.class);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeNamedWriteable(queryRequest);
    }
}
