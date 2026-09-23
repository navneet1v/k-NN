/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless.cluster;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

public class ClusterANNMethodResolverTests extends KNNTestCase {

    private final ClusterANNMethodResolver resolver = new ClusterANNMethodResolver();

    public void testDefaultCompressionIsX8SQ4Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            userContext(new HashMap<>()),
            configContext(VectorDataType.FLOAT, CompressionLevel.NOT_CONFIGURED),
            false,
            SpaceType.L2
        );

        assertEquals(CompressionLevel.x8, result.getCompressionLevel());
        assertEncoderSQBits(result.getKnnMethodContext(), 4);
        assertEquals(KNNEngine.UNDEFINED, result.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.L2, result.getKnnMethodContext().getSpaceType());
        assertEquals(METHOD_CLUSTER, result.getKnnMethodContext().getMethodComponentContext().getName());
    }

    public void test16xMapsToSQ2Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            userContext(new HashMap<>()),
            configContext(VectorDataType.FLOAT, CompressionLevel.x16),
            false,
            SpaceType.INNER_PRODUCT
        );

        assertEquals(CompressionLevel.x16, result.getCompressionLevel());
        assertEncoderSQBits(result.getKnnMethodContext(), 2);
    }

    public void test32xMapsToSQ1Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            userContext(new HashMap<>()),
            configContext(VectorDataType.FLOAT, CompressionLevel.x32),
            false,
            SpaceType.COSINESIMIL
        );

        assertEquals(CompressionLevel.x32, result.getCompressionLevel());
        assertEncoderSQBits(result.getKnnMethodContext(), 1);
    }

    public void test8xMapsToSQ4Bit() {
        ResolvedMethodContext result = resolver.resolveMethod(
            userContext(new HashMap<>()),
            configContext(VectorDataType.FLOAT, CompressionLevel.x8),
            false,
            SpaceType.L2
        );

        assertEquals(CompressionLevel.x8, result.getCompressionLevel());
        assertEncoderSQBits(result.getKnnMethodContext(), 4);
    }

    public void test4xRejected() {
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(new HashMap<>()),
                configContext(VectorDataType.FLOAT, CompressionLevel.x4),
                false,
                SpaceType.L2
            )
        );
        assertTrue(
            "Error message should call out the supported levels explicitly, got: " + e.getMessage(),
            e.getMessage().contains("8x")
                && e.getMessage().contains("16x")
                && e.getMessage().contains("32x")
                && e.getMessage().contains("4x")
        );
    }

    public void test1xRejected() {
        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(new HashMap<>()),
                configContext(VectorDataType.FLOAT, CompressionLevel.x1),
                false,
                SpaceType.L2
            )
        );
    }

    public void testByteVectorTypeRejected() {
        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(new HashMap<>()),
                configContext(VectorDataType.BYTE, CompressionLevel.x32),
                false,
                SpaceType.L2
            )
        );
    }

    public void testBinaryVectorTypeRejected() {
        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(new HashMap<>()),
                configContext(VectorDataType.BINARY, CompressionLevel.x32),
                false,
                SpaceType.L2
            )
        );
    }

    public void testUnsupportedSpaceTypeRejected() {
        // L1, LINF, HAMMING all outside the supported set of L2/INNER_PRODUCT/COSINESIMIL.
        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(new HashMap<>()),
                configContext(VectorDataType.FLOAT, CompressionLevel.x32),
                false,
                SpaceType.L1
            )
        );
        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(new HashMap<>()),
                configContext(VectorDataType.FLOAT, CompressionLevel.x32),
                false,
                SpaceType.HAMMING
            )
        );
    }

    public void testUserSuppliedEncoderIsHonored() {
        // User specifies encoder with bits=2 → resolves to x16 compression.
        Map<String, Object> userParams = new HashMap<>();
        userParams.put(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 2)));

        ResolvedMethodContext result = resolver.resolveMethod(
            userContext(userParams),
            configContext(VectorDataType.FLOAT, CompressionLevel.NOT_CONFIGURED),
            false,
            SpaceType.L2
        );

        assertEquals(CompressionLevel.x16, result.getCompressionLevel());
        assertEncoderSQBits(result.getKnnMethodContext(), 2);
    }

    public void testModeRejected() {
        // Cluster method does not support the "mode" mapping parameter — reject any configured mode.
        KNNMethodConfigContext cfg = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(8)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .build();

        expectThrows(ValidationException.class, () -> resolver.resolveMethod(userContext(new HashMap<>()), cfg, false, SpaceType.L2));
    }

    public void testUnknownTopLevelParameterRejected() {
        // Any key besides "encoder" in method.parameters should be rejected by MethodComponent.validate.
        Map<String, Object> userParams = new HashMap<>();
        userParams.put("some_junk_param", 42);

        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(userParams),
                configContext(VectorDataType.FLOAT, CompressionLevel.x32),
                false,
                SpaceType.L2
            )
        );
    }

    public void testUserEncoderConflictingWithCompressionRejected() {
        // User sets compression_level=16x but encoder bits=1 (which maps to x32) → conflict.
        Map<String, Object> userParams = new HashMap<>();
        userParams.put(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)));

        expectThrows(
            ValidationException.class,
            () -> resolver.resolveMethod(
                userContext(userParams),
                configContext(VectorDataType.FLOAT, CompressionLevel.x16),
                false,
                SpaceType.L2
            )
        );
    }

    private static KNNMethodContext userContext(Map<String, Object> params) {
        return new KNNMethodContext(KNNEngine.UNDEFINED, SpaceType.UNDEFINED, new MethodComponentContext(METHOD_CLUSTER, params));
    }

    private static KNNMethodConfigContext configContext(VectorDataType dataType, CompressionLevel compressionLevel) {
        return KNNMethodConfigContext.builder().vectorDataType(dataType).dimension(8).compressionLevel(compressionLevel).build();
    }

    private static void assertEncoderSQBits(KNNMethodContext ctx, int expectedBits) {
        Object encoder = ctx.getMethodComponentContext().getParameters().get(METHOD_ENCODER_PARAMETER);
        assertTrue(encoder instanceof MethodComponentContext);
        MethodComponentContext encoderCtx = (MethodComponentContext) encoder;
        assertEquals(ENCODER_SQ, encoderCtx.getName());
        assertEquals(expectedBits, encoderCtx.getParameters().get(SQ_BITS));
    }
}
