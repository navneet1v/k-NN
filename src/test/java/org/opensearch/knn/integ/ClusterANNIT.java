/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;

@Log4j2
public class ClusterANNIT extends KNNRestTestCase {

    private static final int DIMENSION = 3;

    // ======================== Test 1: Invalid Mappings ========================

    @SneakyThrows
    public void testClusterMethod_invalidMappings() {
        // Reject engine specified
        String mappingWithEngine = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .field("engine", "faiss")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mappingWithEngine));

        // Reject mode specified
        String mappingWithMode = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mappingWithMode));

        // Reject byte data type
        String mappingWithByte = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("data_type", "byte")
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mappingWithByte));

        // Reject binary data type
        String mappingWithBinary = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 8)
            .field("data_type", "binary")
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mappingWithBinary));
    }

    // ======================== Test 2: Basic Index and Search ========================

    @SneakyThrows
    public void testBasicIndexAndSearch() {
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT };

        for (SpaceType spaceType : spaceTypes) {
            String indexName = INDEX_NAME + "_" + spaceType.getValue();
            createClusterANNIndex(indexName, FIELD_NAME, DIMENSION, spaceType);

            // Ingest docs
            addKnnDoc(indexName, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f });
            addKnnDoc(indexName, "2", FIELD_NAME, new Float[] { 4.0f, 5.0f, 6.0f });
            addKnnDoc(indexName, "3", FIELD_NAME, new Float[] { 7.0f, 8.0f, 9.0f });
            addKnnDoc(indexName, "4", FIELD_NAME, new Float[] { 1.1f, 2.1f, 3.1f });
            addKnnDoc(indexName, "5", FIELD_NAME, new Float[] { 10.0f, 11.0f, 12.0f });

            refreshIndex(indexName);
            forceMergeKnnIndex(indexName);

            // Search
            float[] queryVector = { 1.0f, 2.0f, 3.0f };
            int k = 3;
            List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, queryVector, k);

            // Verify results returned
            assertTrue("Expected results for space type " + spaceType, results.size() > 0);
            assertTrue("Expected at most k results", results.size() <= k);

            // Verify closest doc for L2 and cosine is doc "1" (exact match to query)
            // For inner product, doc "5" ([10,11,12]) has highest dot product with [1,2,3]
            if (spaceType == SpaceType.L2 || spaceType == SpaceType.COSINESIMIL) {
                assertEquals("Closest doc should be doc 1 for " + spaceType, "1", results.get(0).getDocId());
            }

            // Verify scores are positive
            for (KNNResult result : results) {
                assertTrue("Score should be positive for " + spaceType, result.getScore() > 0);
            }

            deleteIndex(indexName);
        }
    }

    // ======================== Test 3: Recall ========================

    @SneakyThrows
    public void testRecall_whenBruteForce_thenPerfectRecall() {
        int dimension = 128;
        URL testIndexVectors = ClusterANNIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = ClusterANNIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        URL groundTruthValues = ClusterANNIT.class.getClassLoader().getResource("data/test_ground_truth_l2_100.csv");
        assertNotNull(testIndexVectors);
        assertNotNull(testQueries);
        assertNotNull(groundTruthValues);
        TestUtils.TestData testData = new TestUtils.TestData(
            testIndexVectors.getPath(),
            testQueries.getPath(),
            groundTruthValues.getPath()
        );

        createClusterANNIndex(INDEX_NAME, FIELD_NAME, dimension, SpaceType.L2);

        // Ingest test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }
        refreshAllIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(INDEX_NAME));

        forceMergeKnnIndex(INDEX_NAME);

        // Query and check recall
        int k = 100;
        for (int i = 0; i < testData.queries.length; i++) {
            List<KNNResult> knnResults = runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[i], k);
            float recall = getRecall(
                Set.of(Arrays.copyOf(testData.groundTruthValues[i], k)),
                knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet())
            );
            assertTrue("Recall should be > 0.9, got: " + recall, recall > 0.9);
        }
    }

    // ======================== Test 4: Filtered Search ========================

    @SneakyThrows
    public void testFilteredSearch() {
        String filterField = "color";

        // Create index with vector + keyword field
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .field("space_type", SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(filterField)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(INDEX_NAME, mapping);

        // Ingest docs with color attribute
        addKnnDocWithAttributes(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f }, ImmutableMap.of(filterField, "red"));
        addKnnDocWithAttributes(INDEX_NAME, "2", FIELD_NAME, new Float[] { 4.0f, 5.0f, 6.0f }, ImmutableMap.of(filterField, "blue"));
        addKnnDocWithAttributes(INDEX_NAME, "3", FIELD_NAME, new Float[] { 7.0f, 8.0f, 9.0f }, ImmutableMap.of(filterField, "red"));
        addKnnDocWithAttributes(INDEX_NAME, "4", FIELD_NAME, new Float[] { 1.1f, 2.1f, 3.1f }, ImmutableMap.of(filterField, "blue"));
        addKnnDocWithAttributes(INDEX_NAME, "5", FIELD_NAME, new Float[] { 10.0f, 11.0f, 12.0f }, ImmutableMap.of(filterField, "red"));

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Search with matching filter — only red docs
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        String queryWithFilter = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(ArrayUtils.toObject(queryVector))
            .k(5)
            .filterFieldName(filterField)
            .filterValue("red")
            .build()
            .getQueryString();
        Response response = searchKNNIndex(INDEX_NAME, queryWithFilter, 5);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertTrue("Should return only red docs", docIds.stream().allMatch(id -> id.equals("1") || id.equals("3") || id.equals("5")));
        assertEquals("Should return 3 red docs", 3, docIds.size());

        // Search with non-matching filter — no results
        String queryNoMatch = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(ArrayUtils.toObject(queryVector))
            .k(5)
            .filterFieldName(filterField)
            .filterValue("green")
            .build()
            .getQueryString();
        Response responseNoMatch = searchKNNIndex(INDEX_NAME, queryNoMatch, 5);
        String entityNoMatch = EntityUtils.toString(responseNoMatch.getEntity());
        assertEquals("Should return 0 results for non-matching filter", 0, parseIds(entityNoMatch).size());
    }

    // ======================== Test 5: Document Lifecycle ========================

    @SneakyThrows
    public void testDocumentLifecycle() {
        createClusterANNIndex(INDEX_NAME, FIELD_NAME, DIMENSION, SpaceType.L2);

        // Ingest docs
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 4.0f, 5.0f, 6.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 7.0f, 8.0f, 9.0f });

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Verify all docs found
        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        List<KNNResult> results = runKnnQuery(INDEX_NAME, FIELD_NAME, queryVector, 5);
        assertEquals(3, results.size());

        // Delete doc 2
        deleteKnnDoc(INDEX_NAME, "2");
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Verify deleted doc absent
        results = runKnnQuery(INDEX_NAME, FIELD_NAME, queryVector, 5);
        assertEquals(2, results.size());
        assertTrue(results.stream().noneMatch(r -> r.getDocId().equals("2")));

        // Update doc 3 to be closer to query
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 1.1f, 2.1f, 3.1f });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Verify updated vector reflected — doc 3 should now be second closest
        results = runKnnQuery(INDEX_NAME, FIELD_NAME, queryVector, 5);
        assertEquals("Doc 1 still closest", "1", results.get(0).getDocId());
        assertEquals("Doc 3 now second closest after update", "3", results.get(1).getDocId());
    }

    // ======================== Test 6: Index Operations ========================

    @SneakyThrows
    public void testIndexOperations() {
        // Test force merge across multiple refreshes
        createClusterANNIndex(INDEX_NAME, FIELD_NAME, DIMENSION, SpaceType.L2);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f });
        refreshIndex(INDEX_NAME);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 4.0f, 5.0f, 6.0f });
        refreshIndex(INDEX_NAME);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 7.0f, 8.0f, 9.0f });
        refreshIndex(INDEX_NAME);

        // Force merge to 1 segment
        forceMergeKnnIndex(INDEX_NAME);

        float[] queryVector = { 1.0f, 2.0f, 3.0f };
        List<KNNResult> results = runKnnQuery(INDEX_NAME, FIELD_NAME, queryVector, 5);
        assertEquals(3, results.size());
        assertEquals("1", results.get(0).getDocId());

        deleteIndex(INDEX_NAME);

        // Test multiple vector fields
        String field2 = "vector_field_2";
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .field("space_type", SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(field2)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .field("space_type", SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(INDEX_NAME, mapping);

        // Ingest docs with both fields — different vectors per field
        String doc = XContentFactory.jsonBuilder()
            .startObject()
            .field(FIELD_NAME, new float[] { 1.0f, 2.0f, 3.0f })
            .field(field2, new float[] { 9.0f, 8.0f, 7.0f })
            .endObject()
            .toString();
        Request request = new Request("POST", "/" + INDEX_NAME + "/_doc/1?refresh=true");
        request.setJsonEntity(doc);
        client().performRequest(request);

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Search field 1
        List<KNNResult> results1 = runKnnQuery(INDEX_NAME, FIELD_NAME, new float[] { 1.0f, 2.0f, 3.0f }, 5);
        assertEquals(1, results1.size());

        // Search field 2
        List<KNNResult> results2 = runKnnQuery(INDEX_NAME, field2, new float[] { 9.0f, 8.0f, 7.0f }, 5);
        assertEquals(1, results2.size());
    }

    // ======================== Test 7: Edge Cases ========================

    @SneakyThrows
    public void testEdgeCases() {
        createClusterANNIndex(INDEX_NAME, FIELD_NAME, DIMENSION, SpaceType.L2);

        // Single document
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        List<KNNResult> results = runKnnQuery(INDEX_NAME, FIELD_NAME, new float[] { 1.0f, 2.0f, 3.0f }, 1);
        assertEquals(1, results.size());
        assertEquals("1", results.get(0).getDocId());

        // k > doc count
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 4.0f, 5.0f, 6.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 7.0f, 8.0f, 9.0f });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        results = runKnnQuery(INDEX_NAME, FIELD_NAME, new float[] { 1.0f, 2.0f, 3.0f }, 100);
        assertEquals("k > doc count should return all docs", 3, results.size());
    }

    // ======================== Helpers ========================

    private void createClusterANNIndex(String indexName, String fieldName, int dimension, SpaceType spaceType) throws IOException {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject("method")
            .field("name", METHOD_CLUSTER)
            .field("space_type", spaceType.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, mapping);
    }

    private List<KNNResult> runKnnQuery(String indexName, String fieldName, float[] queryVector, int k) throws Exception {
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(ArrayUtils.toObject(queryVector))
            .k(k)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(indexName, query, k);
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
    }

    private float getRecall(Set<String> truth, Set<String> result) {
        result.retainAll(truth);
        return (float) result.size() / truth.size();
    }
}
