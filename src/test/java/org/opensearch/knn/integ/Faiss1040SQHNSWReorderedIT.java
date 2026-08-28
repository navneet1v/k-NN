/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Integration test for the locality-reordered Faiss SQ 1-bit (32x) format, toggled by the
 * {@code index.knn.advanced.reordering_enabled} setting.
 *
 * <p>Indexes the <b>same</b> 100 docs into two 32x-compressed indices — one with reordering enabled
 * (uses {@code Faiss1040SQHNSWReorderedKnnVectorsFormat}) and one disabled (baseline
 * {@code Faiss1040ScalarQuantizedKnnVectorsFormat}) — and asserts that a k=10 query returns the same
 * docs with the same scores from both. Since reordering only changes the on-disk layout + entry point
 * (not the score of any doc), the results must match.
 *
 * <p>Lifecycle exercised (per index): index 50 docs → refresh → search (partial/refresh path), index
 * 50 more → refresh → search (multi-segment path), force-merge to 1 segment → search (merge path).
 */
public class Faiss1040SQHNSWReorderedIT extends KNNRestTestCase {

    private static final String FIELD_NAME = "test_field";
    private static final int DIMENSION = 16;
    private static final int NUM_DOCS = 100;
    private static final int HALF = NUM_DOCS / 2;
    private static final int K = 10;

    @SneakyThrows
    public void testReordering_enabledVsDisabled_whenExhaustive_thenSameDocsAndScores() {
        final float[] query = new float[DIMENSION];
        Arrays.fill(query, 42.0f);

        final List<KNNResult> disabled = runLifecycleAndSearch("reorder-disabled-32x", false, query);
        final List<KNNResult> enabled = runLifecycleAndSearch("reorder-enabled-32x", true, query);

        assertEquals("both indices should return k results", K, disabled.size());
        assertEquals("both indices should return k results", K, enabled.size());

        // Compare as docId -> score maps: the reordered index must return the same docs with the same
        // scores as the baseline (scores are doc-intrinsic; the reorder is score-preserving).
        final Map<String, Float> disabledByDoc = disabled.stream().collect(Collectors.toMap(KNNResult::getDocId, KNNResult::getScore));
        final Map<String, Float> enabledByDoc = enabled.stream().collect(Collectors.toMap(KNNResult::getDocId, KNNResult::getScore));

        assertEquals("returned doc sets differ between reordered and baseline", disabledByDoc.keySet(), enabledByDoc.keySet());
        for (final Map.Entry<String, Float> entry : disabledByDoc.entrySet()) {
            assertEquals("score differs for doc " + entry.getKey(), entry.getValue(), enabledByDoc.get(entry.getKey()), 1e-6f);
        }
    }

    /**
     * Creates a 32x index with the given reordering setting, then: index 50 docs → refresh → search,
     * index 50 more → refresh → search, force-merge to 1 segment → return the final k=10 results.
     */
    @SneakyThrows
    private List<KNNResult> runLifecycleAndSearch(final String indexName, final boolean reorderingEnabled, final float[] query) {
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNNSettings.INDEX_KNN_REORDERING_ENABLED, reorderingEnabled)
            .build();
        createKnnIndex(indexName, settings, build32xMapping());

        // First half: index, refresh, search (validates the refresh / first-segment path).
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, HALF);
        refreshIndex(indexName);
        assertFalse("search after first refresh returned nothing", search(indexName, query, K).isEmpty());

        // Second half: index, refresh, search (validates multi-segment search).
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, HALF, NUM_DOCS - HALF);
        refreshIndex(indexName);
        assertFalse("multi-segment search returned nothing", search(indexName, query, K).isEmpty());

        // Merge to a single segment and search (validates the merge path).
        forceMergeKnnIndex(indexName, 1);
        assertFalse("post-merge search returned nothing", search(indexName, query, K).isEmpty());

        // Close and reopen the index, then search again (validates reload-from-disk: reopening the
        // codec/format, re-reading .veqlo/.vemlo + the graph, and re-running search without errors).
        closeIndex(indexName);
        openIndex(indexName);
        ensureGreen(indexName);
        return search(indexName, query, K);
    }

    private String build32xMapping() throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .field("compression_level", "32x")
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    @SneakyThrows
    private List<KNNResult> search(final String indexName, final float[] query, final int k) {
        final Response response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", query)
                .field("k", k)
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            k
        );
        assertOK(response);
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
    }
}
