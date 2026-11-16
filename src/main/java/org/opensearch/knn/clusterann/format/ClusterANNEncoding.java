/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format;

/**
 * The quantization backend a field's vectors are encoded with — the algorithm family. It is an index-time
 * choice threaded down from the format and names the backend alone: the bit width is carried by
 * {@link QuantizationParams}, and the stable {@code .clam} {@code quantizerId} for a backend together with the
 * set of bit widths it supports are owned by the read and write paths.
 *
 * <p>Two backends today — Lucene's optimized scalar quantization and data-blind absolute int8; adding another
 * is a new constant.
 */
public enum ClusterANNEncoding {

    /** Lucene's optimized scalar quantization ({@code OptimizedScalarQuantizer}). */
    OPTIMIZED_SCALAR_QUANTIZATION(ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ),

    /** Data-blind absolute int8 (IVFaster {@code Int8Quantizer}): Hadamard rotation + maxAbs/127, no centroid. */
    DATA_BLIND_INT8(ClusterANNFormatConstants.QUANTIZER_DATA_BLIND_INT8);

    private final int quantizerId;

    ClusterANNEncoding(int quantizerId) {
        this.quantizerId = quantizerId;
    }

    /**
     * The backend a {@code .clam} {@code quantizerId} names.
     *
     * @throws IllegalArgumentException if no backend has that id
     */
    public static ClusterANNEncoding fromQuantizerId(int quantizerId) {
        for (ClusterANNEncoding encoding : values()) {
            if (encoding.quantizerId == quantizerId) {
                return encoding;
            }
        }
        throw new IllegalArgumentException("Unsupported quantizerId: " + quantizerId);
    }
}
