/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;

import java.util.Map;

/**
 * Factory interface for creating engine-less field mappers. Each engine-less algorithm
 * registers a factory in {@link KNNVectorFieldMapper#ENGINE_LESS_MAPPER_FACTORIES}.
 */
@FunctionalInterface
public interface EngineLessMapperFactory {
    KNNVectorFieldMapper create(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        KNNMethodConfigContext knnMethodConfigContext,
        FieldMapper.MultiFields multiFields,
        FieldMapper.CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters
    );
}
