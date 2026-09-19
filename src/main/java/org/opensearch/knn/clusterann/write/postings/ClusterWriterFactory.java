/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.write.QuantizationParams;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;

/**
 * Selects the {@link ClusterWriter} implementation for a cluster's quantization scheme, bound to its centre.
 * The write-side analogue of the read-side cluster factory: the single place that maps a {@code .clam}
 * quantizer scheme to a concrete cluster writer. The output is supplied later, per {@link ClusterWriter#write}.
 */
public final class ClusterWriterFactory {

    private ClusterWriterFactory() {}

    /**
     * Returns the cluster writer for {@code quantization}'s scheme, bound to one cluster's {@code centroid}.
     *
     * @param quantization the field's quantizer scheme and code width
     * @param blockSize    vectors per code block
     * @param dimension    the field's vector dimension
     * @param metric       the field's similarity function, used to build the quantizer
     * @param centroid     this cluster's centre, already rotation-prepared, that members are quantized against
     * @throws IllegalArgumentException if the quantizer scheme is not supported
     */
    public static ClusterWriter newWriter(
        final QuantizationParams quantization,
        final int blockSize,
        final int dimension,
        final VectorSimilarityFunction metric,
        final float[] centroid
    ) {
        return switch (quantization.quantizerId()) {
            case ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ -> new OptimizedScalarQuantizedClusterWriter(
                blockSize,
                dimension,
                metric,
                quantization.docBits(),
                centroid
            );
            default -> throw new IllegalArgumentException("unsupported quantizer scheme: " + quantization.quantizerId());
        };
    }
}
