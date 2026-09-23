/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.engineless.cluster.ClusterANNSQEncoder;

import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * Codec-time format resolution for engineless methods. Reads the resolved SQ bit width ("docBits")
 * stamped by the method resolver from the encoder context in the method parameters, then delegates
 * format construction to the {@link EnginelessMethod} looked up in {@link EnginelessMethodRegistry}.
 *
 * <p>SearchServicesESKNN models this as a {@code CodecFormatResolver} implementation alongside the
 * Lucene and Faiss resolvers; this branch's {@code BasePerFieldKnnVectorsFormat} has no such
 * abstraction yet, so the same logic is exposed as a static entry point.
 */
@Log4j2
public final class EnginelessCodecFormatResolver {

    private EnginelessCodecFormatResolver() {}

    public static KnnVectorsFormat resolve(String field, KNNMethodContext methodContext, Map<String, Object> params) {
        String methodName = methodContext.getMethodComponentContext().getName();
        EnginelessMethod method = EnginelessMethodRegistry.get(methodName)
            .orElseThrow(() -> new IllegalArgumentException(String.format(Locale.ROOT, "Unknown engineless method: \"%s\"", methodName)));
        return method.resolveKnnVectorsFormat(resolveDocBits(field, params));
    }

    private static int resolveDocBits(String field, Map<String, Object> params) {
        if (params == null || !(params.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
            log.debug(
                String.format(
                    Locale.ROOT,
                    "No encoder specified for field [%s], using default docBits=%d",
                    field,
                    ClusterANNSQEncoder.DEFAULT_BITS
                )
            );
            return ClusterANNSQEncoder.DEFAULT_BITS;
        }
        MethodComponentContext encoderCtx = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        Object bitsObj = encoderCtx.getParameters().get(SQ_BITS);
        if (!(bitsObj instanceof Integer)) {
            log.debug(
                String.format(
                    Locale.ROOT,
                    "No bits on encoder for field [%s], using default docBits=%d",
                    field,
                    ClusterANNSQEncoder.DEFAULT_BITS
                )
            );
            return ClusterANNSQEncoder.DEFAULT_BITS;
        }
        int docBits = (Integer) bitsObj;
        log.debug(String.format(Locale.ROOT, "Resolved encoder [%s] for field [%s] with docBits=%d", encoderCtx.getName(), field, docBits));
        return docBits;
    }
}
