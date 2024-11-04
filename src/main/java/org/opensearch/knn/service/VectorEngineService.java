/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.service;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.SpaceType;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
public class VectorEngineService {
    private final Map<OSLuceneDocId, Integer> luceneDocIdToVectorEngineDocId = new ConcurrentHashMap<>();
    // Keeping the OSLuceneDocIds as list to ensure that if IndexReader is open we can reuse the docIds
    private final Map<Integer, List<OSLuceneDocId>> vectorEngineDocIdToLuceneDocId = new ConcurrentHashMap<>();
    private final Map<Integer, float[]> vectorEngineDocIdToVector = new ConcurrentHashMap<>();
    private final AtomicInteger currentVectorDocId = new AtomicInteger(0);

    private static VectorEngineService INSTANCE = null;

    public static VectorEngineService getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new VectorEngineService();
        }
        return INSTANCE;
    }

    public void ingestData(final OSLuceneDocId luceneDocId, float[] vector, final SpaceType spaceType) {
        log.debug("SpaceType during ingestion is : {}", spaceType);
        luceneDocIdToVectorEngineDocId.put(luceneDocId, currentVectorDocId.intValue());
        int currentDocId = currentVectorDocId.intValue();
        vectorEngineDocIdToLuceneDocId.getOrDefault(currentDocId, Collections.synchronizedList(new LinkedList<>())).add(luceneDocId);
        vectorEngineDocIdToVector.put(currentDocId, vector);
        currentVectorDocId.incrementAndGet();
    }

    public List<OSLuceneDocId> search(int k, float[] queryVector, final SpaceType spaceType) {
        int finalk = Math.min(k, vectorEngineDocIdToVector.size());
        PriorityQueue<VectorScoreDoc> scoreDocsQueue = new PriorityQueue<>((a, b) -> Float.compare(a.score, b.score));
        for (int docId : vectorEngineDocIdToVector.keySet()) {
            float score = spaceType.getKnnVectorSimilarityFunction()
                .getVectorSimilarityFunction()
                .compare(queryVector, vectorEngineDocIdToVector.get(docId));
            if (scoreDocsQueue.size() < finalk) {
                scoreDocsQueue.add(new VectorScoreDoc(score, docId));
            } else {
                assert scoreDocsQueue.peek() != null;
                if (score > scoreDocsQueue.peek().score) {
                    scoreDocsQueue.poll();
                    scoreDocsQueue.add(new VectorScoreDoc(score, docId));
                }
            }
        }

        return scoreDocsQueue.parallelStream()
            .flatMap(s -> vectorEngineDocIdToLuceneDocId.get(s.getDocId()).parallelStream()
                    .map(osLuceneDocId -> osLuceneDocId.cloneWithScore(s.getScore())))
            .collect(Collectors.toList());
    }

    public void removeOldSegmentKeys(final byte[] segmentId) {
        for(OSLuceneDocId osLuceneDocId : luceneDocIdToVectorEngineDocId.keySet()) {
            if(Arrays.equals(osLuceneDocId.getSegmentId(), segmentId)) {
                int vectorEngineDocId = luceneDocIdToVectorEngineDocId.remove(osLuceneDocId);

                List<OSLuceneDocId> osLuceneDocIds = vectorEngineDocIdToLuceneDocId.get(vectorEngineDocId);
                for(OSLuceneDocId docId : osLuceneDocIds) {
                    luceneDocIdToVectorEngineDocId.remove(docId);
                }
                // if all the keys are removed
                if(luceneDocIdToVectorEngineDocId.isEmpty()) {
                    // remove the node from vector search DS.
                    vectorEngineDocIdToVector.remove(vectorEngineDocId);
                    // remove the node from VectorEngine to Lucene DocId Map
                    vectorEngineDocIdToLuceneDocId.remove(vectorEngineDocId);
                    // remove the luceneDocId to Vector Engine DocId too.
                    luceneDocIdToVectorEngineDocId.remove(osLuceneDocId);
                }
            }
        }
    }

    @Value
    @Builder
    private static class VectorScoreDoc {
        float score;
        int docId;
    }

}
