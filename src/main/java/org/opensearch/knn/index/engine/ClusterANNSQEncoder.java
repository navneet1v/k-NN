/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

/**
 * Scalar quantization encoder for the cluster ANN algorithm.
 * Supports 1-bit (x32), 2-bit (x16), and 4-bit (x8) quantization.
 */
public class ClusterANNSQEncoder implements Encoder {

    public static final String NAME = "sq";
    public static final String BITS_PARAM = "bits";
    public static final int DEFAULT_BITS = 1;
    private static final Set<Integer> VALID_BITS = Set.of(1, 2, 4);

    private static final MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(NAME)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(BITS_PARAM, new Parameter.IntegerParameter(BITS_PARAM, DEFAULT_BITS, (v, context) -> VALID_BITS.contains(v)))
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
        if (encoderContext == null || !encoderContext.getParameters().containsKey(BITS_PARAM)) {
            return CompressionLevel.NOT_CONFIGURED;
        }

        int bits = (int) encoderContext.getParameters().get(BITS_PARAM);
        // TODO: Find a better way to handle this.
        return switch (bits) {
            case 1 -> CompressionLevel.x32;
            case 2 -> CompressionLevel.x16;
            case 4 -> CompressionLevel.x8;
            default -> CompressionLevel.NOT_CONFIGURED;
        };
    }
}
