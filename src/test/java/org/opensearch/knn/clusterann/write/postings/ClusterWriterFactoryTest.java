/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.write.QuantizationParams;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies {@link ClusterWriterFactory} maps a quantizer scheme to its cluster writer and rejects unknown schemes.
 */
class ClusterWriterFactoryTest {

    private static final int DIMENSION = 4;
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;

    @Test
    void buildsScalarQuantizedWriterForOptimizedSqScheme() {
        final ClusterWriter writer = ClusterWriterFactory.newWriter(
            new QuantizationParams(ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ, (byte) 1),
            8,
            DIMENSION,
            METRIC,
            new float[DIMENSION]
        );
        assertInstanceOf(OptimizedScalarQuantizedClusterWriter.class, writer);
    }

    @Test
    void rejectsUnsupportedScheme() {
        final QuantizationParams unsupported = new QuantizationParams(99, (byte) 1);
        assertThrows(
            IllegalArgumentException.class,
            () -> ClusterWriterFactory.newWriter(unsupported, 8, DIMENSION, METRIC, new float[DIMENSION])
        );
    }
}
