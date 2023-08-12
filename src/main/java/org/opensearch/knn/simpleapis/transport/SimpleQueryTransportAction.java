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
import org.opensearch.action.support.DefaultShardOperationFailedException;
import org.opensearch.action.support.broadcast.node.TransportBroadcastByNodeAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.simpleapis.model.QueryActionRequest;
import org.opensearch.knn.simpleapis.model.QueryActionResponse;
import org.opensearch.knn.simpleapis.model.QueryResponse;
import org.opensearch.knn.simpleapis.model.SimpleQueryResults;
import org.opensearch.knn.simpleapis.query.KNNIndexQueryShard;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;

@Log4j2
public class SimpleQueryTransportAction extends TransportBroadcastByNodeAction<
    QueryActionRequest,
    QueryActionResponse,
    SimpleQueryResults> {

    private final TransportService transportService;
    private final ClusterService clusterService;

    private final IndicesService indicesService;

    private final IndexNameExpressionResolver indexNameExpressionResolver;

    @Inject
    public SimpleQueryTransportAction(
        ClusterService clusterService,
        TransportService transportService,
        IndicesService indicesService,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(
            SimpleQueryAction.NAME,
            clusterService,
            transportService,
            actionFilters,
            indexNameExpressionResolver,
            QueryActionRequest::new,
            ThreadPool.Names.SEARCH
        );
        this.clusterService = clusterService;
        this.transportService = transportService;
        this.indexNameExpressionResolver = indexNameExpressionResolver;
        this.indicesService = indicesService;
    }

    @Override
    protected void doExecute(Task task, QueryActionRequest request, ActionListener<QueryActionResponse> listener) {
        long stime = System.nanoTime();
        log.info("doExecute Start time : {}", stime);
        ActionListener<QueryActionResponse> modifiedActionListener = ActionListener.wrap(res -> {
            listener.onResponse(res);
            log.info("action listener completing : {}", System.nanoTime() - stime);
        }, listener::onFailure);

        super.doExecute(task, request, modifiedActionListener);
    }

    /**
     * Deserialize a shard-level result from an input stream
     *
     * @param in input stream
     * @return a deserialized shard-level result
     */
    @Override
    protected SimpleQueryResults readShardResult(StreamInput in) throws IOException {
        return new SimpleQueryResults(in);
    }

    /**
     * Creates a new response to the underlying request.
     *
     * @param request            the underlying request
     * @param totalShards        the total number of shards considered for execution of the operation
     * @param successfulShards   the total number of shards for which execution of the operation was successful
     * @param failedShards       the total number of shards for which execution of the operation failed
     * @param simpleQueryResults the per-node aggregated shard-level results
     * @param shardFailures      the exceptions corresponding to shard operation failures
     * @param clusterState       the cluster state
     * @return the response
     */
    @Override
    protected QueryActionResponse newResponse(
        QueryActionRequest request,
        int totalShards,
        int successfulShards,
        int failedShards,
        List<SimpleQueryResults> simpleQueryResults,
        List<DefaultShardOperationFailedException> shardFailures,
        ClusterState clusterState
    ) {
        log.info("TotalShards {}, Successful : {}", totalShards, successfulShards);
        return new QueryActionResponse(QueryResponse.builder().simpleQueryResults(simpleQueryResults).build(),
                request.getStartTimeInNano(), request.getInitialNetworkDelay());
    }

    /**
     * Deserialize a request from an input stream
     *
     * @param in input stream
     * @return a de-serialized request
     */
    @Override
    protected QueryActionRequest readRequestFrom(StreamInput in) throws IOException {
        return new QueryActionRequest(in);
    }

    /**
     * Executes the shard-level operation. This method is called once per shard serially on the receiving node.
     *
     * @param request      the node-level request
     * @param shardRouting the shard on which to execute the operation
     * @return the result of the shard-level operation for the shard
     */
    @Override
    protected SimpleQueryResults shardOperation(QueryActionRequest request, ShardRouting shardRouting) {
        log.info("shardOperation Start time : {}", System.nanoTime());
        if (!shardRouting.primary()) {
            return SimpleQueryResults.INSTANCE;
        }
        long stime = System.nanoTime();
        KNNIndexQueryShard knnIndexQueryShard = new KNNIndexQueryShard(
            indicesService.indexServiceSafe(shardRouting.shardId().getIndex()).getShard(shardRouting.shardId().id())
        );
        log.info("Time taken for Getting shard info is : {}", System.nanoTime() - stime);
        stime = System.nanoTime();
        List<SimpleQueryResults.SimpleQueryResult> results = knnIndexQueryShard.query(request.getQueryRequest());
        results.sort(Comparator.comparing(SimpleQueryResults.SimpleQueryResult::getScore, Comparator.reverseOrder()));
        log.info("Time taken for Query is : {}", System.nanoTime() - stime);
        return new SimpleQueryResults(results);
    }

    /**
     * Determines the shards on which this operation will be executed on. The operation is executed once per shard.
     *
     * @param clusterState    the cluster state
     * @param request         the underlying request
     * @param concreteIndices the concrete indices on which to execute the operation
     * @return the shards on which to execute the operation
     */
    @Override
    protected ShardsIterator shards(ClusterState clusterState, QueryActionRequest request, String[] concreteIndices) {
        return clusterState.routingTable().allShards(concreteIndices);
    }

    /**
     * Executes a global block check before polling the cluster state.
     *
     * @param state   the cluster state
     * @param request the underlying request
     * @return a non-null exception if the operation is blocked
     */
    @Override
    protected ClusterBlockException checkGlobalBlock(ClusterState state, QueryActionRequest request) {
        return state.blocks().globalBlockedException(ClusterBlockLevel.METADATA_READ);
    }

    /**
     * Executes a global request-level check before polling the cluster state.
     *
     * @param state           the cluster state
     * @param request         the underlying request
     * @param concreteIndices the concrete indices on which to execute the operation
     * @return a non-null exception if the operation if blocked
     */
    @Override
    protected ClusterBlockException checkRequestBlock(ClusterState state, QueryActionRequest request, String[] concreteIndices) {
        return state.blocks().indicesBlockedException(ClusterBlockLevel.METADATA_READ, concreteIndices);
    }
}
