/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class BinaryCagraTests extends KNNTestCase {

    Map<String, Object> parameters = ImmutableMap.of(
                        INDEX_DESCRIPTION_PARAMETER,
                        "BHNSW16",
                        KNNConstants.SPACE_TYPE,
                        SpaceType.HAMMING.getValue(),
                        KNNConstants.VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BINARY.getValue()
                    );

    @SneakyThrows
    private Map<Integer, Float> binaryIndex() {
        String indexPath = "data/_70_165_target_field.faissc";
        final IndexInput binaryCagra = loadHnswBinary(indexPath);

        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(binaryCagra, null);
        ByteVectorValues byteVectorValues = searcher.faissIndex.getByteValues(binaryCagra);

        FaissIdMapIndex faissIndex = (FaissIdMapIndex) searcher.faissIndex;

        int[] ids = new int[byteVectorValues.size()];
        for(int i = 0 ; i < byteVectorValues.size(); i++) {
            ids[i] = i;
        }


        byte[][] vectors = new byte[byteVectorValues.size()][];

        for(int i = 0 ; i < byteVectorValues.size(); i++) {
            byte[] vector = byteVectorValues.vectorValue(i);
            vectors[i] = new byte[96];
            System.arraycopy(vector, 0, vectors[i], 0, 96);
        }

        for(int i = 0 ; i < 2; i++) {
            for(int j = 0 ; j < 96; j++) {
                System.out.print(vectors[i][j] + " ");
            }
            System.out.println();
        }

        long indexAddress = JNIService.initIndex(vectors.length, 768, parameters, KNNEngine.FAISS);
        long vectorAddress = JNICommons.storeBinaryVectorData(0, vectors, ((long) (768 / 8) * vectors.length));

        JNIService.insertToIndex(ids, vectorAddress, 768, parameters, indexAddress, KNNEngine.FAISS);
        byte[] queryVector = new byte[96];
        for(int i = 0; i < 96; i++) {
            queryVector[i] = (byte)(i % 2);
        }
        KNNQueryResult[] results = JNIService.queryBinaryIndex(indexAddress, queryVector, 100, null, KNNEngine.FAISS,
                null, 0,
                null);
        Map<Integer, Float> resultSet = new HashMap<>();
        System.out.println("Printing Binary HNSW results");
        for(KNNQueryResult result : results) {
            float score = KNNEngine.FAISS.score(result.getScore(), SpaceType.HAMMING);
            //System.out.println(result.getId() + " : " + KNNEngine.FAISS.score(result.getScore(), SpaceType.HAMMING));
            System.out.println(result.getId() + " : " + score);
            System.out.println(result.getId() + " : " + result.getScore());
            resultSet.put(result.getId(), score);
        }
        return resultSet;
    }


    public void testSingleTest() throws IOException {
        Map<Integer, Float> simpleBinaryIndex = binaryIndex();
        Map<Integer, Float> cppset = testCPP();
        Map<Integer, Float> javaSet = test_cagra_mos();

        javaSet.forEach((id, score) -> {
            if(cppset.containsKey(id)) {
                System.out.println("java" + " " + id + " : " + score);
                System.out.println("cpp" + " " + id + " : " + cppset.get(id));
            }

        });
    }

    private Map<Integer, Float> testCPP() {
        String indexPath = "data/_70_165_target_field.faissc";
        final IndexInput binaryCagra = loadHnswBinary(indexPath);
        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(binaryCagra);
        long indexAddress = FaissService.loadBinaryIndexWithStream(indexInputWithBuffer);

        // 768/8
        byte[] queryVector = new byte[96];
        for(int i = 0; i < 96; i++) {
            queryVector[i] = (byte)(i % 2);
        }

        KNNQueryResult[] results = FaissService.queryBinaryIndexWithFilter(indexAddress, queryVector, 100, null, null, 0,
                null);
        Map<Integer, Float> resultSet = new HashMap<>();
        System.out.println("Printing Binary Cagra results");
        for(KNNQueryResult result : results) {
            float score = KNNEngine.FAISS.score(result.getScore(), SpaceType.HAMMING);
            System.out.println(result.getId() + " : " + score);
            System.out.println(result.getId() + " : " + result.getScore());
            resultSet.put(result.getId(), score);
        }
        return resultSet;
    }

    private Map<Integer, Float> test_cagra_mos() throws IOException {
        String indexPath = "data/_70_165_target_field.faissc";
        final IndexInput binaryCagra = loadHnswBinary(indexPath);

        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(binaryCagra, null);

        // Make collector
        final int k = 100;
        final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);

        // Build a query
        byte[] queryVector = new byte[96];
        for(int i = 0; i < 96; i++) {
            queryVector[i] = (byte)(i % 2);
        }

        // Start searching
        searcher.search(queryVector, knnCollector, AcceptDocs.fromLiveDocs(null, 10000));
        final TopDocs topDocs = knnCollector.topDocs();
        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        Map<Integer, Float> resultSet = new HashMap<>();
        System.out.println("Printing MOS results");
        for(ScoreDoc scoreDoc : scoreDocs) {
            System.out.println(scoreDoc.doc + " : " + scoreDoc.score);
            resultSet.put(scoreDoc.doc, scoreDoc.score);
        }
        return resultSet;
    }

        //    public void test_Cagra_CPlusPlus() {
//
//        testCPP();
//    }
//
//    public void test_cagra_with_mos() throws IOException {
//        test_cagra_mos();
//    }
}
