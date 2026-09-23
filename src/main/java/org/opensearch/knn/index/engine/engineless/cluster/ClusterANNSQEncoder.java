/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless.cluster;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * Scalar-quantization encoder for the ClusterANN engineless method. Maps a stamped {@code bits}
 * value on the resolved encoder context to a {@link CompressionLevel} (1 → x32, 2 → x16, 4 → x8).
 */
public class ClusterANNSQEncoder implements Encoder {

    public static final String NAME = ENCODER_SQ;
    public static final int DEFAULT_BITS = 4;

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);
    private static final Set<Integer> SUPPORTED_BITS = Set.of(1, 2, 4);

    private static final MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(NAME)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(SQ_BITS, new Parameter.IntegerParameter(SQ_BITS, DEFAULT_BITS, (v, ctx) -> SUPPORTED_BITS.contains(v)))
        .setRequiresTraining(false)
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (encoderContext != null && encoderContext.getParameters() != null) {
            Object bitsObj = encoderContext.getParameters().get(SQ_BITS);
            if (bitsObj instanceof Integer) {
                return bitsToCompressionLevel((Integer) bitsObj);
            }
        }
        return CompressionLevel.NOT_CONFIGURED;
    }

    /**
     * Inverse of {@link #calculateCompressionLevel}: maps a supported {@link CompressionLevel} to
     * the SQ bit width the encoder writes into segment metadata. Returns {@link #DEFAULT_BITS} for
     * anything outside the supported set (callers should reject unsupported levels up front).
     */
    public static int compressionLevelToBits(CompressionLevel level) {
        return switch (level) {
            case x8 -> 4;
            case x16 -> 2;
            case x32 -> 1;
            default -> DEFAULT_BITS;
        };
    }

    private static CompressionLevel bitsToCompressionLevel(int bits) {
        return switch (bits) {
            case 1 -> CompressionLevel.x32;
            case 2 -> CompressionLevel.x16;
            case 4 -> CompressionLevel.x8;
            default -> CompressionLevel.NOT_CONFIGURED;
        };
    }
}
