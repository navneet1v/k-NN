/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

/**
 * Single source of truth for the ClusterANN IVF file format v2.
 *
 * <p>Two files:
 * <ul>
 *   <li>{@code .clam} — metadata + centroids (read once at init, cached)</li>
 *   <li>{@code .clap} — posting lists + quantized vectors (read during search)</li>
 * </ul>
 *
 * <p>Raw vectors stored separately by FlatVectorsWriter in {@code .vec}.
 */
public final class ClusterANNFormatConstants {

    // File extensions (2 files)
    public static final String META_EXTENSION = "clam";
    public static final String POSTINGS_EXTENSION = "clap";

    // Codec
    public static final String CODEC_NAME = "ClusterANN1040";
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;
    public static final int END_OF_FIELDS = -1;

    // Thresholds
    public static final int MIN_ADC_VECTORS = 32;
    public static final int TARGET_CLUSTER_SIZE = 512;
    public static final float SOAR_LAMBDA = 1.0f;

    // SIMD block size for quantized scoring
    public static final int BLOCK_SIZE = 32;

    // Posting list encoding markers
    public static final byte POSTING_CONTINUOUS = (byte) -1;
    public static final byte POSTING_GROUP_VINT = (byte) 1;

    private ClusterANNFormatConstants() {}
}
