/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless;

import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.OriginalMappingParameters;

import java.util.Map;

/**
 * Produces a {@link KNNVectorFieldMapper} for an engineless method. Signature mirrors
 * {@code EngineFieldMapper#createFieldMapper} so the dispatch in
 * {@link org.opensearch.knn.index.mapper.KNNVectorFieldMapper.Builder} can slot the engineless
 * branch alongside the existing engine-backed and model-based branches.
 */
public interface EnginelessMapperFactory {

    KNNVectorFieldMapper createFieldMapper(
        String fullName,
        String simpleName,
        Map<String, String> meta,
        KNNMethodConfigContext knnMethodConfigContext,
        FieldMapper.MultiFields multiFields,
        FieldMapper.CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters
    );
}
