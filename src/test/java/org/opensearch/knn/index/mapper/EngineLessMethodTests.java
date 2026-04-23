/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.ClusterANN1040KnnVectorsFormat;

public class EngineLessMethodTests extends KNNTestCase {

    @Override
    public void setUp() throws Exception {
        super.setUp();
        // Ensure ClusterANNMethod is registered
        assertNotNull(ClusterANNMethod.INSTANCE);
    }

    public void testFromName_cluster() {
        EngineLessMethod method = EngineLessMethod.fromName("cluster");
        assertNotNull(method);
        assertEquals("cluster", method.getName());
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

    public void testCreateFormat_cluster() {
        EngineLessMethod method = EngineLessMethod.fromName("cluster");
        assertNotNull(method);
        assertTrue(method.createFormat(1) instanceof ClusterANN1040KnnVectorsFormat);
    }

    public void testGetMapperFactory_cluster() {
        EngineLessMethod method = EngineLessMethod.fromName("cluster");
        assertNotNull(method);
        assertNotNull(method.getMapperFactory());
    }

    public void testGetMethodResolver_cluster() {
        EngineLessMethod method = EngineLessMethod.fromName("cluster");
        assertNotNull(method);
        assertNotNull(method.getMethodResolver());
    }
}
