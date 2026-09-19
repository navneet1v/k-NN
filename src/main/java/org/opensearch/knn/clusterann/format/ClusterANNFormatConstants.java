/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format;

/**
 * The stable on-disk constants shared by the ClusterANN read and write paths, so both sides agree
 * byte-for-byte. These are part of the format contract: changing a value is a format-version change, not a
 * refactor.
 */
public final class ClusterANNFormatConstants {

    private ClusterANNFormatConstants() {}

    /** {@code .clam} rotation code for an unrotated field. */
    public static final int ROTATION_NONE = 0;

    /** {@code .clam} rotation code for a dense random Gaussian rotation held in {@code .clar}. */
    public static final int ROTATION_RANDOM_GAUSSIAN = 1;

    /** {@code .clam} quantizer code for optimized scalar quantization. */
    public static final int QUANTIZER_OPTIMIZED_SQ = 0;
}
