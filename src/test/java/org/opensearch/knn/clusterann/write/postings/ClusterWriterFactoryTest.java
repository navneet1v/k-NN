/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

/**
 * Verifies {@link ClusterWriterFactory} maps a quantization backend to its cluster writer and to its stable
 * {@code .clam} {@code quantizerId}. Unsupported backends can't be constructed — {@link ClusterANNEncoding} is a
 * closed enum and the factory's switches are exhaustive — so adding one is a compile error, not a runtime check.
 */
class ClusterWriterFactoryTest {

    private static final int DIMENSION = 4;
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;

    @Test
    void buildsScalarQuantizedWriterForOptimizedSqBackend() {
        final ClusterWriter writer = ClusterWriterFactory.newWriter(
            QuantizationParams.of(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, 1),
            8,
            DIMENSION,
            METRIC,
            new float[DIMENSION]
        );
        assertInstanceOf(OptimizedScalarQuantizedClusterWriter.class, writer);
    }

    @Test
    void mapsOptimizedSqBackendToItsOnDiskQuantizerId() {
        assertEquals(
            ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ,
            ClusterWriterFactory.quantizerId(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION)
        );
    }
}
