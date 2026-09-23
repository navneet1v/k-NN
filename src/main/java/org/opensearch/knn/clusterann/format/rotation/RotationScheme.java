/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;

/**
 * The rotation schemes this codec knows, each holding the {@code .clam} byte code that names it. One typed place
 * owns the metric&rarr;scheme&rarr;code mapping so the write and read paths agree on which byte means which
 * rotation, rather than each switching over the raw {@code ROTATION_*} constants.
 */
public enum RotationScheme {

    /** No rotation: an unrotated field, stored in input space with no {@code .clar} region. */
    NONE(ClusterANNFormatConstants.ROTATION_NONE),

    /** A dense random Gaussian rotation held in {@code .clar}, applied to EUCLIDEAN fields. */
    RANDOM_GAUSSIAN(ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN);

    private final int code;

    RotationScheme(final int code) {
        this.code = code;
    }

    /** The byte {@code .clam} records for this scheme (the {@code ClusterANNFormatConstants.ROTATION_*} code). */
    public int code() {
        return code;
    }

    /** Whether this scheme applies a rotation (so the field has a {@code .clar} region and rotated centroids). */
    public boolean rotates() {
        return this != NONE;
    }

    /**
     * The scheme a field with this metric is written in: EUCLIDEAN is rotated (random Gaussian, to keep 1-bit codes
     * accurate under L2); inner product and cosine are not. The write side's entry point.
     */
    public static RotationScheme forMetric(final VectorSimilarityFunction metric) {
        return metric == VectorSimilarityFunction.EUCLIDEAN ? RANDOM_GAUSSIAN : NONE;
    }

    /**
     * The scheme a {@code .clam} byte names — the read side's entry point.
     *
     * @throws IllegalArgumentException if no scheme claims the code (a segment written by a newer format)
     */
    public static RotationScheme fromCode(final int code) {
        for (final RotationScheme scheme : values()) {
            if (scheme.code == code) {
                return scheme;
            }
        }
        throw new IllegalArgumentException("Unsupported rotation code: " + code);
    }
}
