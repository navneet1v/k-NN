/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

/**
 * Where each cluster's postings landed in {@code .clap}, returned by {@link PostingsWriter} so the seek
 * offsets can be recorded in {@code .clam}.
 *
 * <p>All per-centroid arrays are indexed by centroid id and have length {@code numCentroids}.
 *
 * @param clapOffset      absolute start of this field's region in {@code .clap}
 * @param clapLength      byte length of this field's region in {@code .clap}
 * @param centroidOffsets per-centroid posting offset, relative to {@code clapOffset} (the field's region
 *     start), since the reader slices the field's region and seeks within it
 * @param centroidLengths exact posting byte length per centroid (prefetch hint)
 * @param clusterSizes    posting entry count per centroid (primary + secondary assignment)
 */
public record PostingsRegions(long clapOffset, long clapLength, long[] centroidOffsets, int[] centroidLengths, int[] clusterSizes) {
}
