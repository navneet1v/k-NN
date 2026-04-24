/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

/**
 * Single source of truth for the ClusterANN IVF file format.
 *
 * <p>Two files:
 * <ul>
 *   <li>{@code .clam} — metadata + centroids + centroid stats (read once, cached)</li>
 *   <li>{@code .clap} — posting lists + block-columnar quantized vectors (read during search)</li>
 * </ul>
 */
public final class ClusterANNFormatConstants {

    // File extensions
    public static final String META_EXTENSION = "clam";
    public static final String POSTINGS_EXTENSION = "clap";

    // Codec identity
    public static final String CODEC_NAME = "ClusterANN1040";
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;
    public static final int END_OF_FIELDS = -1;

    // IVF parameters
    public static final int MIN_ADC_VECTORS = 32;
    public static final int TARGET_CLUSTER_SIZE = 512;
    public static final float SOAR_LAMBDA = 1.0f;

    // SIMD block size for quantized scoring
    public static final int BLOCK_SIZE = 32;

    // File alignment (16 bytes = 128-bit SIMD register width)
    public static final int SECTION_ALIGNMENT = 16;

    private ClusterANNFormatConstants() {}
}
