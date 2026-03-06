/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.KnnVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexFloatFlat;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.io.IOException;

@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
class PrefetchableVectorValuesHelper {

    public static void mayBeDoPrefetch(final KnnVectorValues vectorValues, final int[] nodes, final int numNodes) throws IOException {
        switch (vectorValues) {
            case FaissIndexFloatFlat.FloatVectorValuesImpl floatImpl:
                floatImpl.prefetch(nodes, numNodes);
                break;
            case FaissIndexBinaryFlat.ByteVectorValuesImpl binaryImpl:
                binaryImpl.prefetch(nodes, numNodes);
                break;
            case HasIndexSlice luceneKNNVectorValues:
                // Since Lucene uses HasIndexSlice, we can use the slice to prefetch and for lucene base offset for
                // sliced index input is always 0.
                PrefetchHelper.prefetch(luceneKNNVectorValues.getSlice(), 0, vectorValues.getVectorByteLength(), nodes, numNodes);
                break;
            default:
                log.info("Not able to do prefetch on instance {}", vectorValues.getClass().getSimpleName());
        }
    }

}
