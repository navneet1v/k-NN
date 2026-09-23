/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.clusterann.ClusterANNQuery;
import org.opensearch.knn.index.query.clusterann.ClusterANNRescoreQuery;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.lucenelib.OSKnnByteVectorQuery;
import org.opensearch.knn.index.query.lucenelib.OSKnnFloatVectorQuery;
import org.opensearch.knn.index.query.lucenelib.NestedKnnVectorQueryFactory;
import org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery;
import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.index.engine.KNNEngine.ENGINES_SUPPORTING_NESTED_FIELDS;

/**
 * Creates the Lucene k-NN queries
 */
@Log4j2
public class KNNQueryFactory extends BaseQueryFactory {
    /**
     * Creates a Lucene query for a particular engine.
     * @param createQueryRequest request object that has all required fields to construct the query
     * @return Lucene Query
     */
    public static Query create(CreateQueryRequest createQueryRequest) {
        // Engines that create their own custom segment files cannot use the Lucene's KnnVectorQuery. They need to
        // use the custom query type created by the plugin
        final String indexName = createQueryRequest.getIndexName();
        final String fieldName = createQueryRequest.getFieldName();
        final int k = createQueryRequest.getK();
        final float[] vector = createQueryRequest.getVector();
        final float[] originalVector = createQueryRequest.getOriginalVector();
        final byte[] byteVector = createQueryRequest.getByteVector();
        final VectorDataType vectorDataType = createQueryRequest.getVectorDataType();
        final Query filterQuery = getFilterQuery(createQueryRequest);
        final Map<String, ?> methodParameters = createQueryRequest.getMethodParameters();
        final RescoreContext rescoreContext = createQueryRequest.getRescoreContext().orElse(null);
        final boolean expandNested = createQueryRequest.isExpandNested();
        final boolean memoryOptimizedSearchEnabled = createQueryRequest.isMemoryOptimizedSearchEnabled();

        BitSetProducer parentFilter = null;
        int shardId = -1;
        if (createQueryRequest.getContext().isPresent()) {
            QueryShardContext context = createQueryRequest.getContext().get();
            parentFilter = context.getParentFilter();
            shardId = context.getShardId();
        }

        if (parentFilter == null && expandNested) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Invalid value provided for the [%s] field. [%s] is only supported with a nested field.",
                    EXPAND_NESTED,
                    EXPAND_NESTED
                )
            );
        }

        if (isClusterANNField(createQueryRequest)) {
            return createClusterANNQuery(
                fieldName,
                vector,
                vectorDataType,
                k,
                filterQuery,
                parentFilter,
                expandNested,
                rescoreContext,
                shardId
            );
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(createQueryRequest.getKnnEngine())) {
            final Query validatedFilterQuery = validateFilterQuerySupport(filterQuery, createQueryRequest.getKnnEngine());

            log.debug(
                "Creating custom k-NN query for index:{}, field:{}, k:{}, filterQuery:{}, efSearch:{}",
                indexName,
                fieldName,
                k,
                validatedFilterQuery,
                methodParameters
            );

            final KNNQuery knnQuery;
            switch (vectorDataType) {
                case BINARY:
                    knnQuery = KNNQuery.builder()
                        .field(fieldName)
                        .byteQueryVector(byteVector)
                        .indexName(indexName)
                        .parentsFilter(parentFilter)
                        .k(k)
                        .methodParameters(methodParameters)
                        .filterQuery(validatedFilterQuery)
                        .vectorDataType(vectorDataType)
                        .rescoreContext(rescoreContext)
                        .shardId(shardId)
                        .isMemoryOptimizedSearch(memoryOptimizedSearchEnabled)
                        .build();
                    break;
                default:
                    knnQuery = KNNQuery.builder()
                        .field(fieldName)
                        .queryVector(vector)
                        .originalQueryVector(originalVector)
                        .byteQueryVector(byteVector)
                        .indexName(indexName)
                        .parentsFilter(parentFilter)
                        .k(k)
                        .methodParameters(methodParameters)
                        .filterQuery(validatedFilterQuery)
                        .vectorDataType(vectorDataType)
                        .rescoreContext(rescoreContext)
                        .shardId(shardId)
                        .isMemoryOptimizedSearch(memoryOptimizedSearchEnabled)
                        .build();
            }

            if (memoryOptimizedSearchEnabled
                || createQueryRequest.getRescoreContext().isPresent()
                || (ENGINES_SUPPORTING_NESTED_FIELDS.contains(createQueryRequest.getKnnEngine()) && expandNested)) {
                return new NativeEngineKnnVectorQuery(knnQuery, QueryUtils.getInstance(), expandNested);
            }

            return knnQuery;
        }

        int overSampledK = k;
        boolean needsRescore = shouldRescore(rescoreContext);
        if (needsRescore) {
            // Will always do shard level rescoring whenever rescore is required.
            overSampledK = rescoreContext.getFirstPassK(k, false, getDimension(vector, byteVector));
        }

        int luceneK = Math.max(overSampledK, IndexHyperParametersUtil.getHNSWEFSearchValue(methodParameters, indexName));
        log.debug("Creating Lucene k-NN query for index: {}, field:{}, k: {}, luceneK: {}", indexName, fieldName, k, luceneK);
        Query luceneKnnQuery = new LuceneEngineKnnVectorQuery(
            getKnnVectorQuery(fieldName, vector, byteVector, luceneK, filterQuery, parentFilter, expandNested, vectorDataType, k)
        );
        return needsRescore ? new RescoreKNNVectorQuery(luceneKnnQuery, fieldName, k, vector, shardId) : luceneKnnQuery;

    }

    /**
     * What a ClusterANN query rescores with when the request says nothing: on, at 1x.
     *
     * <p><b>On</b>, because the scan ranks by a quantized estimate and the ordering it produces is not the one the user
     * asked for. <b>1x</b>, because that corrects the ordering without widening the scan: the same candidates, scored
     * against the vectors as written. Oversampling is the knob for recall and is left to the request, which is the only
     * place that knows whether the extra postings are worth reading.
     *
     * <p>Not {@link KNNVectorFieldType#resolveRescoreContext}, whose defaults come from the compression level and only
     * apply to {@code mode: on_disk} — a mode ClusterANN rejects, so that path resolves to no rescore at all.
     */
    private static final RescoreContext DEFAULT_CLUSTER_ANN_RESCORE_CONTEXT = RescoreContext.builder()
        .oversampleFactor(RescoreContext.MIN_OVERSAMPLE_FACTOR)
        .userProvided(false)
        .build();

    /**
     * Whether the query is against a ClusterANN field.
     *
     * <p>Read from the mapping rather than from the engine, because ClusterANN has none: it is an engineless method, and
     * the method name is the only thing that identifies it. A missing shard context means there is no mapping to consult,
     * which only happens in call paths that never reach a ClusterANN field.
     */
    private static boolean isClusterANNField(final CreateQueryRequest createQueryRequest) {
        final MappedFieldType fieldType = createQueryRequest.getContext()
            .map(context -> context.fieldMapper(createQueryRequest.getFieldName()))
            .orElse(null);
        return fieldType instanceof KNNVectorFieldType knnVectorFieldType && knnVectorFieldType.isClusterANN();
    }

    /**
     * Builds the ClusterANN query, and the rescore stage on top of it unless the request turns it off.
     *
     * <p>One class per stage, the first collecting its result and the second handing back a scorer:
     *
     * <pre>
     * ClusterANNQuery          approximate scan, one best child per parent when nested
     * ClusterANNRescoreQuery   exact scores for the oversampled candidates
     * </pre>
     *
     * <p>Only the outermost stage hands a scorer to the caller, so Lucene's collector reduces the result.
     *
     * <p>Neither is wrapped in a {@code LuceneEngineKnnVectorQuery}: each already answers {@code rewrite} with itself and
     * does its work when a weight is created, so OpenSearch rewriting again for the fetch phase costs nothing. The
     * {@code ef_search} floor that the Lucene path applies is left off too — there is no graph here for it to mean
     * anything about, and it would silently enlarge the candidate set.
     */
    private static Query createClusterANNQuery(
        final String fieldName,
        final float[] vector,
        final VectorDataType vectorDataType,
        final int k,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNested,
        final RescoreContext rescoreContext,
        final int shardId
    ) {
        if (vectorDataType != VectorDataType.FLOAT) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "ClusterANN supports only float vectors, got [%s]", vectorDataType)
            );
        }
        if (expandNested) {
            // The expand stage has to know which parents won, so it follows the rescore. Not ported yet; rejected rather
            // than silently returning one child per parent, which is a different result than the user asked for.
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] is not supported by ClusterANN yet", EXPAND_NESTED));
        }

        final RescoreContext resolvedRescoreContext = rescoreContext == null ? DEFAULT_CLUSTER_ANN_RESCORE_CONTEXT : rescoreContext;
        final boolean needsRescore = shouldRescore(resolvedRescoreContext);
        // Oversample only when something will reduce it again. Without a rescore the extra candidates are scanned and then
        // dropped by the collector, which is work for nothing.
        final int candidateK = needsRescore ? candidateK(k, resolvedRescoreContext) : k;

        log.debug(
            "Creating ClusterANN query for field:{}, k:{}, candidateK:{}, rescore:{}, oversampleFactor:{}",
            fieldName,
            k,
            candidateK,
            needsRescore,
            resolvedRescoreContext.getOversampleFactor()
        );

        final Query query = new ClusterANNQuery(fieldName, vector, candidateK, filterQuery, parentFilter, shardId);
        return needsRescore ? new ClusterANNRescoreQuery(query, fieldName, candidateK, k, vector, parentFilter) : query;
    }

    /**
     * How many candidates the scan collects for the rescore: {@code k × oversampleFactor}, and nothing more.
     *
     * <p>Not {@link RescoreContext#getFirstPassK}, which floors the answer at {@link RescoreContext#MIN_FIRST_PASS_RESULTS}
     * — a floor sized for a graph walk, where 100 candidates cost little more than 10. Here they are postings read off
     * disk, and it would turn the 1x default into 10x for a typical {@code k}: the factor has to mean what it says.
     *
     * <p>Never below {@code k}, which the cap could otherwise breach for a {@code k} larger than the cap itself.
     */
    private static int candidateK(final int k, final RescoreContext rescoreContext) {
        final int oversampled = (int) Math.ceil(k * rescoreContext.getOversampleFactor());
        return Math.max(k, Math.min(RescoreContext.MAX_FIRST_PASS_RESULTS, oversampled));
    }

    private static int getDimension(float[] floatQueryVector, byte[] byteQueryVector) {
        if (floatQueryVector != null) {
            return floatQueryVector.length;
        }
        if (byteQueryVector != null) {
            return byteQueryVector.length;
        }
        throw new IllegalStateException("QueryVector has neither float nor byte array");
    }

    private static Query validateFilterQuerySupport(final Query filterQuery, final KNNEngine knnEngine) {
        log.debug("filter query {}, knnEngine {}", filterQuery, knnEngine);
        if (filterQuery != null && KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            return filterQuery;
        }
        return null;
    }

    private static boolean shouldRescore(RescoreContext rescoreContext) {
        return rescoreContext != null && rescoreContext.isRescoreEnabled();
    }

    private static Query getKnnVectorQuery(
        final String fieldName,
        final float[] floatQueryVector,
        final byte[] byteQueryVector,
        final int luceneK,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNested,
        @NonNull final VectorDataType vectorDataType,
        final int k
    ) {
        if (parentFilter == null) {
            assert expandNested == false : "expandNested is allowed to be true only for nested fields.";
            return vectorDataType == VectorDataType.FLOAT
                ? new OSKnnFloatVectorQuery(fieldName, floatQueryVector, luceneK, filterQuery, k)
                : new OSKnnByteVectorQuery(fieldName, byteQueryVector, luceneK, filterQuery, k);
        }
        // If parentFilter is not null, it is a nested query. Therefore, we delegate creation of query to {@link
        // NestedKnnVectorQueryFactory}
        // which will create query to dedupe search result per parent so that we can get k parent results at the end.
        return vectorDataType == VectorDataType.FLOAT
            ? NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                floatQueryVector,
                luceneK,
                filterQuery,
                parentFilter,
                expandNested,
                k
            )
            : NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                byteQueryVector,
                luceneK,
                filterQuery,
                parentFilter,
                expandNested,
                k
            );
    }
}
