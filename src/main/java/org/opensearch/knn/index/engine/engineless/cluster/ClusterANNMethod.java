/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless.cluster;

import com.google.common.collect.ImmutableSet;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.engineless.EnginelessMapperFactory;
import org.opensearch.knn.index.engine.engineless.EnginelessMethod;
import org.opensearch.knn.index.mapper.ClusterANNVectorFieldMapper;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * ClusterANN engineless method — IVF + SOAR + scalar quantization. The codec format is
 * {@link KNN1030ClusterANNVectorsFormat}, synced from {@code SearchServicesKnnVectorFormats}; the
 * resolved SQ bit width selects its {@link QuantizationParams}.
 */
public final class ClusterANNMethod implements EnginelessMethod {

    public static final ClusterANNMethod INSTANCE = new ClusterANNMethod();

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);
    private static final ClusterANNSQEncoder SQ_ENCODER = new ClusterANNSQEncoder();
    public static final Map<String, Encoder> SUPPORTED_ENCODERS = Map.of(SQ_ENCODER.getName(), SQ_ENCODER);

    public static final MethodComponent METHOD_COMPONENT = initMethodComponent();

    private final MethodResolver resolver = new ClusterANNMethodResolver();
    private final EnginelessMapperFactory mapperFactory = ClusterANNVectorFieldMapper::createFieldMapper;

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_CLUSTER)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .build();
    }

    private static Parameter.MethodComponentContextParameter initEncoderParameter() {
        MethodComponentContext defaultEncoder = new MethodComponentContext(
            ClusterANNSQEncoder.NAME,
            new HashMap<>(Map.of(SQ_BITS, ClusterANNSQEncoder.DEFAULT_BITS))
        );
        return new Parameter.MethodComponentContextParameter(
            METHOD_ENCODER_PARAMETER,
            defaultEncoder,
            SUPPORTED_ENCODERS.values().stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent))
        );
    }

    @Override
    public String getName() {
        return METHOD_CLUSTER;
    }

    @Override
    public MethodResolver getMethodResolver() {
        return resolver;
    }

    @Override
    public EnginelessMapperFactory getMapperFactory() {
        return mapperFactory;
    }

    @Override
    public KnnVectorsFormat resolveKnnVectorsFormat(int bits) {
        return new KNN1030ClusterANNVectorsFormat(
            KNN1030ClusterANNVectorsFormat.FORMAT_NAME,
            QuantizationParams.of(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, bits)
        );
    }
}
