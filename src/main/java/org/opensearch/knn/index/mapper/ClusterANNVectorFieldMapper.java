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
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Field mapper for the ClusterANN engineless method. Produces a {@link KNNVectorFieldType} whose
 * {@link KNNMappingConfig} carries the resolved {@link KNNMethodContext} (engine =
 * {@link org.opensearch.knn.index.engine.KNNEngine#UNDEFINED}) and persists the routing name plus
 * the SQ bit width into segment field attributes so downstream readers can identify the field
 * without re-parsing the full method params JSON.
 */
public class ClusterANNVectorFieldMapper extends KNNVectorFieldMapper {

    private final VectorValidator vectorValidator;

    public static ClusterANNVectorFieldMapper createFieldMapper(
        String fullName,
        String simpleName,
        Map<String, String> meta,
        KNNMethodConfigContext knnMethodConfigContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        OriginalMappingParameters originalMappingParameters
    ) {
        final KNNMethodContext resolvedMethodContext = originalMappingParameters.getResolvedKnnMethodContext();
        if (resolvedMethodContext == null) {
            throw new IllegalStateException(
                String.format(Locale.ROOT, "\"%s\" method requires a resolved KNNMethodContext", METHOD_CLUSTER)
            );
        }

        final KNNMappingConfig mappingConfig = new KNNMappingConfig() {
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
                CompressionLevel cl = knnMethodConfigContext.getCompressionLevel();
                // The resolver always stamps a level (x8 / 4-bit SQ when the mapping names none), so this is a fallback only.
                return CompressionLevel.isConfigured(cl) ? cl : CompressionLevel.x8;
            }

            @Override
            public Version getIndexCreatedVersion() {
                return knnMethodConfigContext.getVersionCreated();
            }
        };

        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullName,
            meta,
            knnMethodConfigContext.getVectorDataType(),
            mappingConfig,
            mappingConfig.getIndexCreatedVersion()
        );

        return new ClusterANNVectorFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            originalMappingParameters,
            resolvedMethodContext,
            knnMethodConfigContext
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
        OriginalMappingParameters originalMappingParameters,
        KNNMethodContext resolvedMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        super(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            mappedFieldType.getKnnMappingConfig().getIndexCreatedVersion(),
            originalMappingParameters
        );
        this.useLuceneBasedVectorField = true;
        final SpaceType spaceType = resolvedMethodContext.getSpaceType();
        this.vectorValidator = new SpaceVectorValidator(spaceType);

        final int dimension = knnMethodConfigContext.getDimension();
        final FieldType clusterFieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        clusterFieldType.putAttribute(KNN_METHOD, METHOD_CLUSTER);
        clusterFieldType.putAttribute(DIMENSION, String.valueOf(dimension));
        clusterFieldType.putAttribute(SPACE_TYPE, spaceType.getValue());
        clusterFieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, mappedFieldType.getVectorDataType().getValue());
        resolveSQBits(resolvedMethodContext).ifPresent(bits -> clusterFieldType.putAttribute(SQ_BITS, String.valueOf(bits)));
        clusterFieldType.setVectorAttributes(
            dimension,
            VectorEncoding.FLOAT32,
            spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
        );
        clusterFieldType.freeze();
        this.fieldType = clusterFieldType;
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return vectorValidator;
    }

    /** The frozen Lucene field type (attributes + vector attributes) this mapper indexes with. Package-private for tests. */
    FieldType luceneFieldType() {
        return fieldType;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }

    private static Optional<Integer> resolveSQBits(KNNMethodContext resolvedMethodContext) {
        Map<String, Object> methodParams = resolvedMethodContext.getMethodComponentContext().getParameters();
        Object encoder = methodParams == null ? null : methodParams.get(METHOD_ENCODER_PARAMETER);
        if (!(encoder instanceof MethodComponentContext)) {
            return Optional.empty();
        }
        Object bits = ((MethodComponentContext) encoder).getParameters().get(SQ_BITS);
        return bits instanceof Integer ? Optional.of((Integer) bits) : Optional.empty();
    }
}
