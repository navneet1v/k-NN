/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;

import java.io.IOException;

public class OpenSearch1030ClusterANNVectorsFormatComponentTest extends LuceneTestCase {

    public void testSearchRoundTrip(VectorSimilarityFunction simFunc) throws IOException {
        try (Directory dir = newDirectory()) {

        }

    }

}
