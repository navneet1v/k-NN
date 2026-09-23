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
 * <p>There is one backend today (Lucene's optimized scalar quantization); adding another is a new constant.
 */
public enum ClusterANNEncoding {

    /** Lucene's optimized scalar quantization ({@code OptimizedScalarQuantizer}). */
    OPTIMIZED_SCALAR_QUANTIZATION
}
