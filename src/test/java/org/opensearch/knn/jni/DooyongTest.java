/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.Pair;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.FilterIdsSelector;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class DooyongTest extends KNNTestCase {

    @SneakyThrows
    @Test
    public void testKdy() {
        final String dirName = "/Users/navneev/tmp/cagra-debug";
        final String fileName = "_70_165_target_field.faissc";
        final Directory directory = new MMapDirectory(Path.of(dirName));
        final IndexInput indexInput = directory.openInput(fileName, IOContext.READONCE);
        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(indexInput);
        final Map<String, Object> parameters = new HashMap<>();
        parameters.put("data_type", "binary");
        parameters.put("space_type", "innerproduct");
        final long indexAddress = JNIService.loadIndex(indexInputWithBuffer, parameters, KNNEngine.FAISS);
        System.out.println("Addr -> " + indexAddress);

        // Query
        final int k = 500;
        byte[] query = new byte[96];
        for (int i = 0; i < query.length; i++) {
            query[i] = (byte) ThreadLocalRandom.current().nextInt();
        }
        System.out.println("Query -> " + Arrays.toString(query));

        KNNQueryResult[] results = JNIService.queryBinaryIndex(
                indexAddress,
                query,
                k,
                Collections.emptyMap(),
                KNNEngine.FAISS,
                new long[0],
                FilterIdsSelector.FilterIdsSelectorType.BITMAP.getValue(),
                null
        );

        System.out.println("Len -> " + results.length);
        for (KNNQueryResult knnQueryResult : results) {
            System.out.println(knnQueryResult.getId() + " / " + KNNEngine.FAISS.score(knnQueryResult.getScore(), SpaceType.HAMMING));
        }

        System.out.println("+++++++++++++++++++++++++++++++++++++++++++++++++++++");

        // MOS
        indexInput.seek(0);
        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(indexInput, null);
        final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
        searcher.search(query, knnCollector, AcceptDocs.fromLiveDocs(null, Integer.MAX_VALUE));
        final TopDocs topDocs = knnCollector.topDocs();
        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        System.out.println("Len -> " + scoreDocs.length);

        for (ScoreDoc scoreDoc : scoreDocs) {
            System.out.println(scoreDoc.doc + " / " + scoreDoc.score);
        }

        // Recall - Faiss C++
        final ByteVectorValues values = searcher.faissIndex.getByteValues(indexInput.clone());
        final List<Integer> ids = new ArrayList<>();
        for (KNNQueryResult knnQueryResult : results) {
            ids.add(knnQueryResult.getId());
        }

        final int numVectors = searcher.faissIndex.getTotalNumberOfVectors();
        float recall = calcRecall(query, numVectors, values, ids, k);
        System.out.println("Faiss C++ recall : " + recall);

        // Recall - MOS
        ids.clear();
        for (ScoreDoc scoreDoc : scoreDocs) {
            ids.add(scoreDoc.doc);
        }
        recall = calcRecall(query, numVectors, values, ids, k);
        System.out.println("MOS recall : " + recall);
    }

    @SneakyThrows
    private float calcRecall(final byte[] query, final int numVectors, ByteVectorValues values, List<Integer> ids, int k) {
        // Min-heap sorted by distance (largest first)
        PriorityQueue<Candidate> pq = new PriorityQueue<>(k, Comparator.comparing((Candidate c) -> c.score));

        for (int i = 0; i < numVectors; ++i) {
            byte[] vec = values.vectorValue(i);
            float score = KNNVectorSimilarityFunction.HAMMING.compare(vec, query);

            if (pq.size() < 30) {
                pq.add(new Candidate(i, score));
            } else if (score > pq.peek().score) {
                // This one is better than the current worst; replace the worst
                pq.poll();
                pq.add(new Candidate(i, score));
            }
        }

        final Candidate[] candidates = pq.toArray(new Candidate[0]);
        System.out.println("+============ Ans ===========+");
        for (Candidate candidate : candidates) {
            System.out.println(candidate.docId + " / " + candidate.score);
        }
        System.out.println("+============ Ans ===========+");

        final Set<Integer> ans = new HashSet<>();
        for (Candidate candidate : candidates) {
            ans.add(candidate.docId);
        }

        final Set<Integer> acquired = new HashSet<>(ids);
        int intersection = 0;
        for (Integer t : ans) {
            if (acquired.contains(t)) {
                intersection++;
            }
        }
        return (float) intersection / ans.size();
    }


    class Candidate {
        final int docId;
        final float score;

        Candidate(int docId, float score) {
            this.docId = docId;
            this.score = score;
        }
    }
}




//    public void singTest() throws IOException {
//        Set<Pair<Integer, Float>> cppset = testCPP();
//        Set<Pair<Integer, Float>> javaSet = test_cagra_mos();
//        Set<Integer> cppIds = new HashSet<>();
//        for(Pair<Integer, Float> pair : cppset) {
//            cppIds.add(pair.getFirst());
//        }
//
//
//        for(Pair<Integer, Float> java: javaSet) {
//            if(cppIds.contains(java.getFirst())) {
//                System.out.println("Found " + java);
//            }
//        }
//    }
//
//    //    public void test_Cagra_CPlusPlus() {
////
////        testCPP();
////    }
////
////    public void test_cagra_with_mos() throws IOException {
////        test_cagra_mos();
////    }
//
//    Set<Pair<Integer, Float>> testCPP() {
//        String indexPath = "data/_70_165_target_field.faissc";
//        final IndexInput binaryCagra = loadHnswBinary(indexPath);
//        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(binaryCagra);
//        long indexAddress = FaissService.loadBinaryIndexWithStream(indexInputWithBuffer);
//
//        // 768/8
//        byte[] queryVector = new byte[96];
//        for(int i = 0; i < 96; i++) {
//            queryVector[i] = (byte)(i % 2);
//        }
//
//        KNNQueryResult[] results = FaissService.queryBinaryIndexWithFilter(indexAddress, queryVector, 100, null, null, 0,
//                null);
//        Set<Pair<Integer, Float>> resultSet = new HashSet<>();
//        for(KNNQueryResult result : results) {
//            //System.out.println(result.getId() + " : " + KNNEngine.FAISS.score(result.getScore(), SpaceType.HAMMING));
//            System.out.println(result.getId() + " : " + result.getScore());
//            resultSet.add(new Pair<>(result.getId(), result.getScore()));
//        }
//        return resultSet;
//    }
//
//    Set<Pair<Integer, Float>> test_cagra_mos() throws IOException {
//        String indexPath = "data/_70_165_target_field.faissc";
//        final IndexInput binaryCagra = loadHnswBinary(indexPath);
//
//        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(binaryCagra, null);
//
//        // Make collector
//        final int k = 100;
//        final KnnCollector knnCollector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
//
//        // Build a query
//        byte[] queryVector = new byte[96];
//        for(int i = 0; i < 96; i++) {
//            queryVector[i] = (byte)(i % 2);
//        }
//
//        // Start searching
//        searcher.search(queryVector, knnCollector, AcceptDocs.fromLiveDocs(null, 10000));
//        final TopDocs topDocs = knnCollector.topDocs();
//        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
//        Set<Pair<Integer, Float>> resultSet = new HashSet<>();
//        for(ScoreDoc scoreDoc : scoreDocs) {
//            System.out.println(scoreDoc.doc + " : " + scoreDoc.score);
//            resultSet.add(new Pair<>(scoreDoc.doc, scoreDoc.score));
//        }
//        return resultSet;
//    }
//}
