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

package org.opensearch.knn.simpleapis.transport;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.TransportAction;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.simpleapis.model.QueryActionRequest;
import org.opensearch.knn.simpleapis.model.QueryActionResponse;
import org.opensearch.knn.simpleapis.model.QueryResponse;
import org.opensearch.knn.simpleapis.model.SimpleQueryResults;
import org.opensearch.knn.simpleapis.query.KNNIndexQueryShard;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Log4j2
public class SimpleQueryTransportAction2 extends TransportAction<QueryActionRequest, QueryActionResponse> {

    private final TransportService transportService;
    private final ClusterService clusterService;

    private final IndicesService indicesService;

    private final IndexNameExpressionResolver indexNameExpressionResolver;

    @Inject
    public SimpleQueryTransportAction2(
        ClusterService clusterService,
        TransportService transportService,
        IndicesService indicesService,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(SimpleQueryAction2.NAME, actionFilters, transportService.getTaskManager());
        this.clusterService = clusterService;
        this.transportService = transportService;
        this.indexNameExpressionResolver = indexNameExpressionResolver;
        this.indicesService = indicesService;
    }

    @Override
    protected void doExecute(Task task, QueryActionRequest request, ActionListener<QueryActionResponse> listener) {
        String[] indices = indexNameExpressionResolver.concreteIndexNames(clusterService.state(), request);
        ShardsIterator shardIt = clusterService.state().routingTable().allShards(indices);
        List<SimpleQueryResults> simpleQueryResults = new ArrayList<>();
        for (ShardRouting shardRouting : shardIt) {
            KNNIndexQueryShard knnIndexQueryShard = new KNNIndexQueryShard(
                indicesService.indexServiceSafe(shardRouting.shardId().getIndex()).getShard(shardRouting.shardId().id())
            );
            List<SimpleQueryResults.SimpleQueryResult> results = knnIndexQueryShard.query(request.getQueryRequest());
            results.sort(Comparator.comparing(SimpleQueryResults.SimpleQueryResult::getScore, Comparator.reverseOrder()));
            simpleQueryResults.add(new SimpleQueryResults(results));
        }
        listener.onResponse(
            new QueryActionResponse(
                QueryResponse.builder().simpleQueryResults(simpleQueryResults).build(), request.getStartTimeInNano(),
                    request.getInitialNetworkDelay()
            )
        );
    }
}
