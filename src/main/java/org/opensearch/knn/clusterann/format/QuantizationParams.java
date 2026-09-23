/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format;

/**
 * How a field's vectors are quantized into {@code .clap} code blocks: the quantization backend
 * ({@link ClusterANNEncoding}) and how many bits each coordinate takes ({@code docBits}). Both are recorded in
 * {@code .clam} so the reader decodes the blocks the same way, and carrying them together keeps the pair
 * consistent as it is threaded through the writers. The width is validated where it is used — by the writer as
 * it builds the quantizer and by the reader as it decodes. Prefer {@link #of(ClusterANNEncoding, int)} (or
 * {@link #DEFAULT}) for construction.
 *
 * @param encoding the quantization backend
 * @param docBits  bits per quantized coordinate, driving the code-block width
 */
public record QuantizationParams(ClusterANNEncoding encoding, int docBits) {

    /** The default encoding: Lucene optimized scalar quantization at 2 bits per coordinate. */
    public static final QuantizationParams DEFAULT = of(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, 2);

    /**
     * A quantization for the given backend at {@code bits} bits per coordinate.
     */
    public static QuantizationParams of(final ClusterANNEncoding encoding, final int bits) {
        return new QuantizationParams(encoding, bits);
    }
}
