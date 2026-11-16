/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8ClusterWriter;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;

/**
 * Creates the {@link ClusterWriter} for each cluster of one field. It is the single place that maps a quantization
 * backend to its concrete {@link ClusterWriter}, to the vectors per code block it lays a field out with, and to the
 * stable {@code .clam} {@code quantizerId} recorded for it. Each mapping switches over the closed {@link ClusterANNEncoding} enum, so
 * every backend is handled and adding one is a compile error.
 */
public final class ClusterWriterFactory {

    private final QuantizationParams quantizationParams;
    private final int dimension;
    private final VectorSimilarityFunction metric;
    private final int storageBlockBytes;

    /**
     * @param quantizationParams the field's encoding and code width
     * @param dimension          the field's vector dimension
     * @param metric             the field's similarity function, used to build the quantizer
     * @param storageBlockBytes  bytes the storage layer fetches in one block, which the field's layout sizes itself
     *     against
     */
    public ClusterWriterFactory(
        final QuantizationParams quantizationParams,
        final int dimension,
        final VectorSimilarityFunction metric,
        final int storageBlockBytes
    ) {
        validateDocBits(quantizationParams);
        this.quantizationParams = quantizationParams;
        this.dimension = dimension;
        this.metric = metric;
        this.storageBlockBytes = storageBlockBytes;
    }

    /**
     * Rejects a {@code docBits} the chosen backend cannot honour, so the width recorded in {@code .clam} cannot
     * drift from the bytes written. The data-blind int8 grid is fixed at 8-bit codes
     * ({@link DataBlindInt8ClusterWriter#DOC_BITS}), so {@code docBits} is not a free parameter for that backend;
     * the optimized-scalar backend validates its own width where it builds its quantizer.
     *
     * <p>Call this at the write seam <em>before</em> any {@code .clam} entry is written — including for an empty
     * field, which records {@code docBits} but never constructs a factory — so empty and non-empty fields fail
     * identically.
     *
     * @throws IllegalArgumentException if {@code quantizationParams} names a width the backend cannot store
     */
    public static void validateDocBits(final QuantizationParams quantizationParams) {
        if (quantizationParams.encoding() == ClusterANNEncoding.DATA_BLIND_INT8
            && quantizationParams.docBits() != DataBlindInt8ClusterWriter.DOC_BITS) {
            throw new IllegalArgumentException(
                "DATA_BLIND_INT8 encodes "
                    + DataBlindInt8ClusterWriter.DOC_BITS
                    + "-bit codes, but docBits="
                    + quantizationParams.docBits()
                    + " was requested"
            );
        }
    }

    /**
     * Returns the cluster writer for one cluster of the field, bound to its {@code centroid}.
     *
     * @param centroid this cluster's centre, already rotation-prepared, that members are quantized against
     */
    public ClusterWriter newWriter(final float[] centroid) {
        return switch (quantizationParams.encoding()) {
            case OPTIMIZED_SCALAR_QUANTIZATION -> new OptimizedScalarQuantizedClusterWriter(
                storageBlockBytes,
                dimension,
                metric,
                quantizationParams.docBits(),
                centroid
            );
            case DATA_BLIND_INT8 -> new org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8ClusterWriter(
                storageBlockBytes,
                dimension
            );
        };
    }

    /**
     * Vectors per code block a field's cluster writers lay it out with; what {@code .clam} records for it.
     *
     * @param quantizationParams the field's encoding and code width
     * @param dimension          the field's vector dimension
     * @param storageBlockBytes  bytes the storage layer fetches in one block
     */
    public static int vectorsPerBlock(final QuantizationParams quantizationParams, final int dimension, final int storageBlockBytes) {
        return switch (quantizationParams.encoding()) {
            case OPTIMIZED_SCALAR_QUANTIZATION -> OptimizedScalarQuantizedClusterWriter.vectorsPerBlock(
                storageBlockBytes,
                dimension,
                quantizationParams.docBits()
            );
            case DATA_BLIND_INT8 -> org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8ClusterWriter.vectorsPerBlock(
                storageBlockBytes,
                dimension
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
            case DATA_BLIND_INT8 -> ClusterANNFormatConstants.QUANTIZER_DATA_BLIND_INT8;
        };
    }
}
