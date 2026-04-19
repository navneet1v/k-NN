/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.mapper.EngineLessMethod;

import java.util.Map;

/**
 * {@link CodecFormatResolver} implementation for engine-less algorithms.
 * Delegates format creation to {@link EngineLessMethod#getFormat()} based on the method name
 * in the {@link KNNMethodContext}.
 */
public class EngineLessCodecFormatResolver implements CodecFormatResolver {

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
        return method.getFormat();
    }

    @Override
    public KnnVectorsFormat resolve() {
        throw new UnsupportedOperationException("Engine-less format resolver requires method context, use resolve(field, ...) instead");
    }
}
