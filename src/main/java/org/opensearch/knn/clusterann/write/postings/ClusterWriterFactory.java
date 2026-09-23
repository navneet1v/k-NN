/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;

/**
 * The write side's single place that maps a quantization backend to its concrete {@link ClusterWriter} and to
 * the stable {@code .clam} {@code quantizerId} recorded for it. Both mappings switch over the closed
 * {@link ClusterANNEncoding} enum, so every backend is handled and adding one is a compile error.
 */
public final class ClusterWriterFactory {

    private ClusterWriterFactory() {}

    /**
     * Returns the cluster writer for {@code quantizationParams}'s backend, bound to one cluster's {@code centroid}.
     *
     * @param quantizationParams the field's encoding and code width
     * @param blockSize    vectors per code block
     * @param dimension    the field's vector dimension
     * @param metric       the field's similarity function, used to build the quantizer
     * @param centroid     this cluster's centre, already rotation-prepared, that members are quantized against
     */
    public static ClusterWriter newWriter(
        final QuantizationParams quantizationParams,
        final int blockSize,
        final int dimension,
        final VectorSimilarityFunction metric,
        final float[] centroid
    ) {
        return switch (quantizationParams.encoding()) {
            case OPTIMIZED_SCALAR_QUANTIZATION -> new OptimizedScalarQuantizedClusterWriter(
                blockSize,
                dimension,
                metric,
                quantizationParams.docBits(),
                centroid
            );
        };
    }

    /**
     * The stable {@code .clam} {@code quantizerId} a backend is recorded under, for the writer to persist. The
     * read side maps this code back to its cluster implementation.
     */
    public static int quantizerId(final ClusterANNEncoding encoding) {
        return switch (encoding) {
            case OPTIMIZED_SCALAR_QUANTIZATION -> ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ;
        };
    }
}
