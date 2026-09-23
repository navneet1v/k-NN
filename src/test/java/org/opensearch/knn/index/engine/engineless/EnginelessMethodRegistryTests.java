/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.engineless.cluster.ClusterANNMethod;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;

public class EnginelessMethodRegistryTests extends KNNTestCase {

    public void testClusterMethodIsRegistered() {
        assertTrue(EnginelessMethodRegistry.get(METHOD_CLUSTER).isPresent());
        assertSame(ClusterANNMethod.INSTANCE, EnginelessMethodRegistry.get(METHOD_CLUSTER).orElseThrow());
        assertTrue(EnginelessMethodRegistry.isEnginelessMethod(METHOD_CLUSTER));
    }

    public void testGetReturnsEmptyForUnknownName() {
        assertFalse(EnginelessMethodRegistry.get(null).isPresent());
        assertFalse(EnginelessMethodRegistry.get("").isPresent());
        assertFalse(EnginelessMethodRegistry.get("hnsw").isPresent());
        assertFalse(EnginelessMethodRegistry.isEnginelessMethod("hnsw"));
    }
}
