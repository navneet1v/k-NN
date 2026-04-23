/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.EngineLessMethod;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * {@link CodecFormatResolver} implementation for engine-less algorithms.
 * Follows the same pattern as {@link org.opensearch.knn.index.engine.lucene.LuceneCodecFormatResolver}:
 * determines quantization configuration from the resolved encoder in method parameters,
 * then delegates format creation to {@link EngineLessMethod}.
 */
@Log4j2
public class EngineLessCodecFormatResolver implements CodecFormatResolver {

    // TODO: Find a better place to keep this.
    private static final Map<String, Encoder> ENCODER_MAP = Map.of(ClusterANNSQEncoder.NAME, new ClusterANNSQEncoder());

    @Override
    public KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth
    ) {
        String methodName = methodContext.getMethodComponentContext().getName();
        EngineLessMethod method = EngineLessMethod.fromName(methodName);
        if (method == null) {
            throw new IllegalArgumentException(String.format("Unknown engine-less method: '%s'", methodName));
        }

        int docBits = resolveDocBits(field, params);
        return method.createFormat(docBits);
    }

    @Override
    public KnnVectorsFormat resolve() {
        throw new UnsupportedOperationException("Engine-less format resolver requires method context, use resolve(field, ...) instead");
    }

    /**
     * Resolves docBits from the encoder in method parameters.
     * If an encoder is present, uses its {@link Encoder#calculateCompressionLevel} to derive the bits.
     * Otherwise, falls back to the default.
     */
    private int resolveDocBits(String field, Map<String, Object> params) {
        if (params == null || !params.containsKey(METHOD_ENCODER_PARAMETER)) {
            log.debug("No encoder specified for field [{}], using default docBits={}", field, ClusterANNSQEncoder.DEFAULT_BITS);
            return ClusterANNSQEncoder.DEFAULT_BITS;
        }

        Object encoderObj = params.get(METHOD_ENCODER_PARAMETER);
        if (encoderObj instanceof MethodComponentContext encoderCtx) {
            Encoder encoder = ENCODER_MAP.get(encoderCtx.getName());
            if (encoder != null) {
                CompressionLevel level = encoder.calculateCompressionLevel(encoderCtx, KNNMethodConfigContext.EMPTY);
                int docBits = compressionToDocBits(level);
                log.debug("Resolved encoder [{}] for field [{}] with docBits={}", encoderCtx.getName(), field, docBits);
                return docBits;
            }
            throw new IllegalArgumentException(String.format("Unknown encoder '%s' for engine-less method", encoderCtx.getName()));
        }

        return ClusterANNSQEncoder.DEFAULT_BITS;
    }

    private static int compressionToDocBits(CompressionLevel level) {
        return switch (level) {
            case x8 -> 4;
            case x16 -> 2;
            case x32 -> 1;
            default -> ClusterANNSQEncoder.DEFAULT_BITS;
        };
    }
}
