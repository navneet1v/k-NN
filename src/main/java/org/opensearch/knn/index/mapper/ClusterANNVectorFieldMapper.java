/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Field mapper for engine-less cluster-based ANN algorithm.
 * Uses Lucene's native KnnFloatVectorField for vector storage.
 */
public class ClusterANNVectorFieldMapper extends KNNVectorFieldMapper {

    private final VectorValidator vectorValidator;
    private final VectorTransformer vectorTransformer;

    public static ClusterANNVectorFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        KNNMethodConfigContext knnMethodConfigContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters
    ) {
        final SpaceType spaceType = originalMappingParameters.getResolvedKnnMethodContext().getSpaceType();
        final KNNMethodContext resolvedMethodContext = originalMappingParameters.getResolvedKnnMethodContext();
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            new KNNMappingConfig() {
                @Override
                public Optional<KNNMethodContext> getKnnMethodContext() {
                    return Optional.of(resolvedMethodContext);
                }

                @Override
                public int getDimension() {
                    return knnMethodConfigContext.getDimension();
                }

                @Override
                public CompressionLevel getCompressionLevel() {
                    return knnMethodConfigContext.getCompressionLevel();
                }

                @Override
                public Version getIndexCreatedVersion() {
                    return knnMethodConfigContext.getVersionCreated();
                }
            }
        );
        // Engine-less fields always search via the Lucene reader path
        mappedFieldType.alwaysUseMemoryOptimizedSearch = true;
        return new ClusterANNVectorFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext.getVersionCreated(),
            originalMappingParameters,
            spaceType,
            knnMethodConfigContext.getDimension()
        );
    }

    private ClusterANNVectorFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion,
        OriginalMappingParameters originalMappingParameters,
        SpaceType spaceType,
        int dimension
    ) {
        super(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            indexCreatedVersion,
            originalMappingParameters
        );
        this.useLuceneBasedVectorField = true;
        this.vectorValidator = new SpaceVectorValidator(spaceType);
        this.vectorTransformer = VectorTransformerFactory.getVectorTransformer(spaceType);

        String methodName = originalMappingParameters.getResolvedKnnMethodContext().getMethodComponentContext().getName();
        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        this.fieldType.putAttribute(KNN_METHOD, methodName);
        this.fieldType.putAttribute(SPACE_TYPE, spaceType.getValue());
        this.fieldType.putAttribute(DIMENSION, String.valueOf(dimension));
        this.fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        this.fieldType.setVectorAttributes(
            dimension,
            VectorEncoding.FLOAT32,
            spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
        );
        this.fieldType.freeze();
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return vectorValidator;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }

    @Override
    protected VectorTransformer getVectorTransformer() {
        return vectorTransformer;
    }
}
