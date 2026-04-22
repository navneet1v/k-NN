/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1040Codec.ClusterANN1040KnnVectorsFormat;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

public class EngineLessCodecFormatResolverTests extends KNNTestCase {

    private final EngineLessCodecFormatResolver resolver = new EngineLessCodecFormatResolver();

    private KNNMethodContext clusterContext(Map<String, Object> params) {
        return new KNNMethodContext(KNNEngine.UNDEFINED, SpaceType.L2, new MethodComponentContext(METHOD_CLUSTER, params));
    }

    private Map<String, Object> paramsWithEncoder(int bits) {
        Map<String, Object> params = new HashMap<>();
        params.put(
            METHOD_ENCODER_PARAMETER,
            new MethodComponentContext(ClusterANNSQEncoder.NAME, new HashMap<>(Map.of(ClusterANNSQEncoder.BITS_PARAM, bits)))
        );
        return params;
    }

    // Basic resolution

    public void testResolve_cluster_noParams() {
        KnnVectorsFormat format = resolver.resolve("test-field", clusterContext(new HashMap<>()), null, 16, 100);
        assertTrue(format instanceof ClusterANN1040KnnVectorsFormat);
    }

    public void testResolve_unknownMethod_throws() {
        KNNMethodContext ctx = new KNNMethodContext(
            KNNEngine.UNDEFINED,
            SpaceType.L2,
            new MethodComponentContext("unknown", new HashMap<>())
        );
        expectThrows(IllegalArgumentException.class, () -> resolver.resolve("test-field", ctx, null, 16, 100));
    }

    public void testResolve_noArg_throws() {
        expectThrows(UnsupportedOperationException.class, () -> resolver.resolve());
    }

    // DocBits extraction from encoder

    public void testResolve_encoder1bit() {
        Map<String, Object> params = paramsWithEncoder(1);
        KnnVectorsFormat format = resolver.resolve("test-field", clusterContext(params), params, 16, 100);
        assertTrue(format instanceof ClusterANN1040KnnVectorsFormat);
    }

    public void testResolve_encoder2bit() {
        Map<String, Object> params = paramsWithEncoder(2);
        KnnVectorsFormat format = resolver.resolve("test-field", clusterContext(params), params, 16, 100);
        assertTrue(format instanceof ClusterANN1040KnnVectorsFormat);
    }

    public void testResolve_encoder4bit() {
        Map<String, Object> params = paramsWithEncoder(4);
        KnnVectorsFormat format = resolver.resolve("test-field", clusterContext(params), params, 16, 100);
        assertTrue(format instanceof ClusterANN1040KnnVectorsFormat);
    }

    // Default when no encoder

    public void testResolve_nullParams_usesDefault() {
        KnnVectorsFormat format = resolver.resolve("test-field", clusterContext(new HashMap<>()), null, 16, 100);
        assertTrue(format instanceof ClusterANN1040KnnVectorsFormat);
    }

    // Unknown encoder

    public void testResolve_unknownEncoder_throws() {
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, new MethodComponentContext("unknown_encoder", new HashMap<>()));
        expectThrows(IllegalArgumentException.class, () -> resolver.resolve("test-field", clusterContext(params), params, 16, 100));
    }
}
