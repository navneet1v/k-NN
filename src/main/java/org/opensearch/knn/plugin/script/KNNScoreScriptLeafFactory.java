/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import lombok.Builder;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.VectorScorer;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

/**
 * A factory that creates KNNScoreScript instances for per-document scoring, and optionally provides
 * a custom {@link BulkScorer} for batch scoring via the {@link ScoreScript.BulkScoringLeafFactory}
 * interface.
 *
 * <p>When the bulk path is available (float vectors with a compatible space type and index version),
 * it uses {@link VectorScorer.Bulk} to batch-prefetch and score documents in groups of 64,
 * significantly reducing I/O latency. Otherwise, it falls back to the per-document scoring path.
 */
public class KNNScoreScriptLeafFactory implements ScoreScript.BulkScoringLeafFactory {
    private final Map<String, Object> params;
    private final SearchLookup lookup;
    private final String similaritySpace;
    private final String field;
    private final Object query;
    private final KNNScoringSpace knnScoringSpace;
    private final IndexSearcher searcher;
    private final MappedFieldType mappedFieldType;

    public KNNScoreScriptLeafFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher) {
        KNNCounter.SCRIPT_QUERY_REQUESTS.increment();
        this.params = params;
        this.lookup = lookup;
        this.field = getValue(params, "field").toString();
        this.similaritySpace = getValue(params, "space_type").toString();
        this.query = getValue(params, "query_value");
        this.searcher = searcher;
        this.mappedFieldType = lookup.doc().mapperService().fieldType(this.field);
        this.knnScoringSpace = KNNScoringSpaceFactory.create(this.similaritySpace, this.query, this.mappedFieldType);
    }

    private Object getValue(Map<String, Object> params, String fieldName) {
        final Object value = params.get(fieldName);
        if (value != null) return value;

        KNNCounter.SCRIPT_QUERY_ERRORS.increment();
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Missing parameter [%s]", fieldName));
    }

    @Override
    public boolean needs_score() {
        return false;
    }

    /**
     * For each segment, supply the KNNScoreScript that should be used to re-score the documents returned from the
     * query. Because the method to score the documents was set during factory construction, the scripts are agnostic of
     * the similarity space. The KNNScoringSpace will return the correct script, given the query, the field type, and
     * the similarity space.
     *
     * @param ctx LeafReaderContext for the segment
     * @return ScoreScript to be executed
     */
    @Override
    public ScoreScript newInstance(LeafReaderContext ctx) throws IOException {
        return knnScoringSpace.getScoreScript(params, field, lookup, ctx, this.searcher);
    }

    /**
     * Create a BulkScorer that scores documents in batches for improved I/O efficiency.
     *
     * <p><b>Contract:</b>
     * <ul>
     *   <li>{@code subQueryBulkScorer} is guaranteed non-null by the caller.</li>
     *   <li>If this method returns {@code null}, the caller falls back to the default per-document
     *       scoring path. The {@code subQueryBulkScorer} remains unconsumed and will be used by
     *       the fallback scorer.</li>
     *   <li>If this method returns a non-null {@link BulkScorer}, the caller uses it exclusively
     *       for this segment. The implementation takes ownership of {@code subQueryBulkScorer}
     *       and is responsible for driving or discarding it.</li>
     *   <li>The returned scorer must produce valid scores: non-negative, non-NaN, with the
     *       provided {@code boost} applied.</li>
     * </ul>
     *
     * @param context            the leaf reader context for this segment
     * @param subQueryBulkScorer the sub-query's BulkScorer that drives document iteration (never null)
     * @param subQueryScoreMode  the score mode of the sub-query
     * @param boost              the boost factor to apply to scores
     * @return a BulkScorer, or null to fall back to default per-document scoring
     */
    @Override
    public BulkScorer bulkScorer(
        final LeafReaderContext context,
        final BulkScorer subQueryBulkScorer,
        final ScoreMode subQueryScoreMode,
        float boost
    ) {
        if (isBulkScoringSupported() == false) {
            return null;
        }
        final KNNScoreScriptBulkScorerContext knnScoreScriptBulkScorerContext = KNNScoreScriptBulkScorerContext.builder()
            .spaceType(SpaceType.getSpace(similaritySpace))
            .queryVector(KNNScoringSpaceUtil.getProcessedQuery(query, (KNNVectorFieldType) mappedFieldType))
            .fieldName(field)
            .build();
        return new KNNScriptScoreBulkScorer(context, subQueryBulkScorer, boost, knnScoreScriptBulkScorerContext);
    }

    private boolean isBulkScoringSupported() {
        if ((mappedFieldType instanceof KNNVectorFieldType) == false) {
            return false;
        }
        final SpaceType spaceType = SpaceType.getSpace(similaritySpace);
        return spaceType == SpaceType.L2 || spaceType == SpaceType.INNER_PRODUCT;
    }

    @Builder
    public record KNNScoreScriptBulkScorerContext(Object queryVector, String fieldName, SpaceType spaceType) {
    }
}
