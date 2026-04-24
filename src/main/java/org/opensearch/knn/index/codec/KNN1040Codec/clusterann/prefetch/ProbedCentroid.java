/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.clusterann.prefetch;

/**
 * Metadata about a centroid to probe, known before reading the posting list.
 * Computed from the offset table at probe time — no I/O needed.
 *
 * @param centroidIdx   original centroid index (for centroids[][] lookup)
 * @param fileOffset    byte offset in .clap
 * @param postingBytes  total bytes for this centroid (primary + SOAR)
 * @param centroidDist  distance from query to centroid
 */
public record ProbedCentroid(int centroidIdx, long fileOffset, long postingBytes, float centroidDist) {
}
