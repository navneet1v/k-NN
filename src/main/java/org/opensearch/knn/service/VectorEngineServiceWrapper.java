/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.service;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.List;

@Log4j2
public class VectorEngineServiceWrapper {

    public static <T> void ingestData(final KNNVectorValues<T> knnVectorValues,
                                      final SegmentWriteState segmentWriteState, final FieldInfo fieldInfo)
        throws IOException {
        byte[] segmentId = segmentWriteState.segmentInfo.getId();
        VectorEngineService vectorEngineService = VectorEngineService.getInstance();
        SpaceType spaceType = FieldInfoExtractor.getSpaceType(null, fieldInfo);

        for (int docId = knnVectorValues.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = knnVectorValues.nextDoc()) {
            log.debug("Adding DocId: {}", docId);
            // we need to
            vectorEngineService.ingestData(
                OSLuceneDocId.builder().segmentDocId(docId).segmentId(segmentId).build(),
                (float[]) knnVectorValues.getVector(), spaceType
            );
        }
    }

    public static List<OSLuceneDocId> search(final KNNQuery knnQuery, final FieldInfo fieldInfo) {
        VectorEngineService vectorEngineService = VectorEngineService.getInstance();
        SpaceType spaceType = FieldInfoExtractor.getSpaceType(null, fieldInfo);
        return vectorEngineService.search(knnQuery.getK(), knnQuery.getQueryVector(), spaceType);
    }

    public static void close(final SegmentWriteState segmentWriteState) {
        VectorEngineService.getInstance().removeOldSegmentKeys(segmentWriteState.segmentInfo.getId());
    }
}
