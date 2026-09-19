/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write;

/**
 * How a field's vectors are quantized into {@code .clap} code blocks: the quantizer scheme
 * ({@link org.opensearch.knn.clusterann.format.ClusterANNFormatConstants#QUANTIZER_OPTIMIZED_SQ}, ...) and
 * how many bits each coordinate takes ({@code docBits}). Both are recorded in {@code .clam} so the reader
 * decodes the blocks the same way; grouping them keeps the two from drifting apart as they are threaded
 * through the writers.
 *
 * @param quantizerId the quantizer scheme code
 * @param docBits     bits per quantized coordinate, driving the code-block width
 */
public record QuantizationParams(int quantizerId, byte docBits) {
}
