/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

public class ClusterANNMethodResolverTests extends KNNTestCase {

    private final ClusterANNMethodResolver resolver = new ClusterANNMethodResolver();

    private KNNMethodConfigContext configContext(CompressionLevel compression) {
        return KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .versionCreated(Version.CURRENT)
            .compressionLevel(compression)
            .build();
    }

    private KNNMethodContext clusterMethodContext() {
        return new KNNMethodContext(KNNEngine.UNDEFINED, SpaceType.L2, new MethodComponentContext(METHOD_CLUSTER, new HashMap<>()), false);
    }

    private KNNMethodContext clusterMethodContextWithEncoder(int bits) {
        Map<String, Object> params = new HashMap<>();
        params.put(
            METHOD_ENCODER_PARAMETER,
            new MethodComponentContext(ClusterANNSQEncoder.NAME, new HashMap<>(Map.of(ClusterANNSQEncoder.BITS_PARAM, bits)))
        );
        return new KNNMethodContext(KNNEngine.UNDEFINED, SpaceType.L2, new MethodComponentContext(METHOD_CLUSTER, params), false);
    }

    // Path 1: compression_level only

    public void testResolve_compressionX32_resolves1Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContext(),
            configContext(CompressionLevel.x32),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x32, result.getCompressionLevel());
        MethodComponentContext encoder = (MethodComponentContext) result.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertNotNull(encoder);
        assertEquals(ClusterANNSQEncoder.NAME, encoder.getName());
        assertEquals(1, encoder.getParameters().get(ClusterANNSQEncoder.BITS_PARAM));
    }

    public void testResolve_compressionX16_resolves2Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContext(),
            configContext(CompressionLevel.x16),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x16, result.getCompressionLevel());
        MethodComponentContext encoder = (MethodComponentContext) result.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertEquals(2, encoder.getParameters().get(ClusterANNSQEncoder.BITS_PARAM));
    }

    public void testResolve_compressionX8_resolves4Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContext(),
            configContext(CompressionLevel.x8),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x8, result.getCompressionLevel());
        MethodComponentContext encoder = (MethodComponentContext) result.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertEquals(4, encoder.getParameters().get(ClusterANNSQEncoder.BITS_PARAM));
    }

    // Path 2: encoder only

    public void testResolve_encoderOnly_1bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContextWithEncoder(1),
            configContext(CompressionLevel.NOT_CONFIGURED),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x32, result.getCompressionLevel());
    }

    public void testResolve_encoderOnly_2bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContextWithEncoder(2),
            configContext(CompressionLevel.NOT_CONFIGURED),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x16, result.getCompressionLevel());
    }

    public void testResolve_encoderOnly_4bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContextWithEncoder(4),
            configContext(CompressionLevel.NOT_CONFIGURED),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x8, result.getCompressionLevel());
    }

    // Path 3: both — matching

    public void testResolve_bothMatching_noConflict() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContextWithEncoder(1),
            configContext(CompressionLevel.x32),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x32, result.getCompressionLevel());
    }

    // Path 3: both — conflicting

    public void testResolve_bothConflicting_throws() {
        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(clusterMethodContextWithEncoder(4), configContext(CompressionLevel.x32), false, SpaceType.L2)
        );
    }

    // Path 4: neither — default

    public void testResolve_neitherSet_defaultTo1Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            clusterMethodContext(),
            configContext(CompressionLevel.NOT_CONFIGURED),
            false,
            SpaceType.L2
        );
        // Default encoder should be created with 1-bit
        MethodComponentContext encoder = (MethodComponentContext) result.getKnnMethodContext()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertNotNull(encoder);
        assertEquals(1, encoder.getParameters().get(ClusterANNSQEncoder.BITS_PARAM));
    }
}
