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

import com.google.common.collect.ImmutableList;
import lombok.extern.log4j.Log4j2;
import org.opensearch.client.node.NodeClient;
import org.opensearch.knn.simpleapis.model.QueryActionRequest;
import org.opensearch.knn.simpleapis.model.QueryRequest;
import org.opensearch.knn.simpleapis.transport.SimpleQueryAction;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.rest.action.admin.cluster.RestNodesUsageAction;

import java.io.IOException;
import java.util.List;

/**
 * /vector_search/my-index/_search
 * {
 * "k": 10,
 * "vector": [1.3, 4.2, 10.2],
 * "vectorFieldName": "name of the field where "
 * }
 */
@Log4j2
public class SimpleQueryHandler extends BaseRestHandler {

    public static final String NAME = "simple_query_api";

    /**
     * @return the name of this handler. The name should be human readable and
     * should describe the action that will performed when this API is
     * called. This name is used in the response to the
     * {@link RestNodesUsageAction}.
     */
    @Override
    public String getName() {
        return NAME;
    }

    /**
     * Prepare the request for execution. Implementations should consume all request params before
     * returning the runnable for actual execution. Unconsumed params will immediately terminate
     * execution of the request. However, some params are only used in processing the response;
     * implementations can override {@link BaseRestHandler#responseParams()} to indicate such
     * params.
     *
     * @param request the request to execute
     * @param client  client for executing actions on the local node
     * @return the action to execute
     * @throws IOException if an I/O exception occurred parsing the request and preparing for
     *                     execution
     */
    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        long stime = System.nanoTime();
        QueryActionRequest queryActionRequest = SimpleQueryHandlerHelper.createQueryRequest(request, stime);
        return restChannel -> client.execute(
            SimpleQueryAction.INSTANCE,
            queryActionRequest,
            new RestToXContentListener<>(restChannel)
        );
    }

    /**
     * The list of {@link Route}s that this RestHandler is responsible for handling.
     */
    @Override
    public List<Route> routes() {
        return ImmutableList.of(new Route(RestRequest.Method.POST, "/vector_search/{index}/_query"));
    }
}
