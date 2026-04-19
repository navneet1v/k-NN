/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.ClusterANN1040KnnVectorsFormat;

public class EngineLessMethodTests extends KNNTestCase {

    public void testFromName_cluster() {
        assertEquals(EngineLessMethod.CLUSTER, EngineLessMethod.fromName("cluster"));
    }

    public void testFromName_null() {
        assertNull(EngineLessMethod.fromName(null));
    }

    public void testFromName_unknown() {
        assertNull(EngineLessMethod.fromName("hnsw"));
    }

    public void testIsEngineLess_cluster() {
        assertTrue(EngineLessMethod.isEngineLess("cluster"));
    }

    public void testIsEngineLess_hnsw() {
        assertFalse(EngineLessMethod.isEngineLess("hnsw"));
    }

    public void testIsEngineLess_null() {
        assertFalse(EngineLessMethod.isEngineLess(null));
    }

    public void testGetFormat_cluster() {
        assertTrue(EngineLessMethod.CLUSTER.getFormat() instanceof ClusterANN1040KnnVectorsFormat);
    }
}
