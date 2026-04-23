/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.ClusterANN1040KnnVectorsFormat;
import org.opensearch.knn.index.engine.ClusterANNMethodResolver;
import org.opensearch.knn.index.engine.MethodResolver;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;

/**
 * {@link EngineLessMethod} implementation for the cluster-based ANN algorithm (IVF with SOAR).
 * Uses {@link ClusterANN1040KnnVectorsFormat} for indexing and search, and
 * {@link ClusterANNMethodResolver} for compression/encoder resolution.
 *
 * <p>Self-registers via static initializer — class must be loaded at startup
 * (triggered by {@link KNNVectorFieldMapper}).
 */
public class ClusterANNMethod implements EngineLessMethod {

    /** Singleton instance. */
    public static final ClusterANNMethod INSTANCE = new ClusterANNMethod();

    static {
        EngineLessMethod.register(INSTANCE);
    }

    private final MethodResolver methodResolver = new ClusterANNMethodResolver();

    private ClusterANNMethod() {}

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return METHOD_CLUSTER;
    }

    /** {@inheritDoc} */
    @Override
    public EngineLessMapperFactory getMapperFactory() {
        return ClusterANNVectorFieldMapper::createFieldMapper;
    }

    /**
     * {@inheritDoc}
     *
     * @param docBits scalar quantization bits (1, 2, or 4) for the cluster ANN codec
     */
    @Override
    public KnnVectorsFormat createFormat(int docBits) {
        return new ClusterANN1040KnnVectorsFormat(docBits);
    }

    /** {@inheritDoc} */
    @Override
    public MethodResolver getMethodResolver() {
        return methodResolver;
    }
}
