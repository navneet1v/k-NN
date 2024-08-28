/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.common.Explicit;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * Mapper used when you don't want to build an underlying KNN struct - you just want to store vectors in flat file.
 * Currently, one way achieve this is by marking index.knn as false index settings. This mapper is used for that case.
 * But with version 2.17 of OpenSearch we are adding another capability where user can do index:false in mappings
 * to provide us details that he doesn't want to create k-NN datastructures.
 */
public class FlatVectorFieldMapper extends KNNVectorFieldMapper {

    private final PerDimensionValidator perDimensionValidator;

    public static FlatVectorFieldMapper createFieldMapper(
        String fullname,
        String simpleName,
        Map<String, String> metaValue,
        KNNMethodConfigContext knnMethodConfigContext,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues
    ) {
        final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
            fullname,
            metaValue,
            knnMethodConfigContext.getVectorDataType(),
            knnMethodConfigContext::getDimension,
            false
        );
        return new FlatVectorFieldMapper(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodConfigContext
        );
    }

    private FlatVectorFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
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
            knnMethodConfigContext.getVersionCreated(),
            null
        );
        // the parent class sets this attribute to always true, only for FlatMapper we will put this as false, because
        // this mapper is used for doing exact search.
        this.index = false;
        // setting it explicitly false here to ensure that when flatmapper is used Lucene based Vector field is not created.
        // there are 2 cases for flat fieldMapper, one where index: false in mapping and another is index.knn: false as
        // index setting
        this.useLuceneBasedVectorField = KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(indexCreatedVersion)
            && knnMethodConfigContext.isIndexKNN();
        this.perDimensionValidator = selectPerDimensionValidator(vectorDataType);
        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);
        if (useLuceneBasedVectorField) {
            int adjustedDimension = mappedFieldType.vectorDataType == VectorDataType.BINARY
                ? knnMethodConfigContext.getDimension() / 8
                : knnMethodConfigContext.getDimension();
            final VectorEncoding encoding = mappedFieldType.vectorDataType == VectorDataType.FLOAT
                ? VectorEncoding.FLOAT32
                : VectorEncoding.BYTE;
            fieldType.setVectorAttributes(
                adjustedDimension,
                encoding,
                SpaceType.DEFAULT.getKnnVectorSimilarityFunction().getVectorSimilarityFunction()
            );
        } else {
            fieldType.setDocValuesType(DocValuesType.BINARY);
        }
        this.fieldType.putAttribute(DIMENSION, String.valueOf(knnMethodConfigContext.getDimension()));
        // setting default space type here once Space Type is moved to top level field we will use that value to set
        // the space type
        this.fieldType.putAttribute(SPACE_TYPE, SpaceType.DEFAULT.getValue());
        this.fieldType.freeze();
    }

    private PerDimensionValidator selectPerDimensionValidator(VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BIT_VALIDATOR;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            return PerDimensionValidator.DEFAULT_BYTE_VALIDATOR;
        }

        return PerDimensionValidator.DEFAULT_FLOAT_VALIDATOR;
    }

    @Override
    protected VectorValidator getVectorValidator() {
        return VectorValidator.NOOP_VECTOR_VALIDATOR;
    }

    @Override
    protected PerDimensionValidator getPerDimensionValidator() {
        return perDimensionValidator;
    }

    @Override
    protected PerDimensionProcessor getPerDimensionProcessor() {
        return PerDimensionProcessor.NOOP_PROCESSOR;
    }
}
