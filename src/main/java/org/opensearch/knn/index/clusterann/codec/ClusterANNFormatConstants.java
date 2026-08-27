/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * Single source of truth for the ClusterANN IVF file format.
 *
 * <p>Four files:
 * <ul>
 *   <li>{@code .clam} — metadata + centroid stats + offset tables (read once, cached)</li>
 *   <li>{@code .clac} — centroids (raw + transformed) + ordToDoc/ordToCentroid region (mmap'd)</li>
 *   <li>{@code .clar} — serialized random rotation, one per rotated field (mmap'd; L2 only)</li>
 *   <li>{@code .clap} — posting lists + block-columnar quantized vectors (read during search)</li>
 * </ul>
 */
public final class ClusterANNFormatConstants {

    // File extensions
    public static final String META_EXTENSION = "clam";
    public static final String POSTINGS_EXTENSION = "clap";
    public static final String CENTROIDS_EXTENSION = "clac";
    public static final String ROTATION_EXTENSION = "clar";

    // Codec identity
    public static final String CODEC_NAME = "ClusterANN1040";
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;
    public static final int END_OF_FIELDS = -1;

    // IVF parameters
    public static final int MIN_ADC_VECTORS = 32;
    public static final int TARGET_CLUSTER_SIZE = 512;
    public static final float SOAR_LAMBDA = 1.0f;

    // Filtering: max filterCost * dimension to use exact scoring (Tier 1)
    // At 768d: threshold / 768 ≈ 2666 docs per segment triggers exact path
    public static final long EXACT_FILTER_THRESHOLD = 2_048_000L;

    // SIMD block size for quantized scoring
    public static final int BLOCK_SIZE = 32;

    // File alignment (16 bytes = 128-bit SIMD register width)
    public static final int SECTION_ALIGNMENT = 16;

    private ClusterANNFormatConstants() {}
}
