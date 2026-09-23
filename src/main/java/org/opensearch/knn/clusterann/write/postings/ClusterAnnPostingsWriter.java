/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter.ClusterMembers;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter.ClusterRegion;
import org.opensearch.knn.clusterann.ClusteringResult;
import org.opensearch.knn.clusterann.format.rotation.Rotation;

import java.io.IOException;

/**
 * Serializes a field's postings to {@code .clap}. Partitions the clustering result's ordinals by centroid
 * (primary and secondary assignments combined), then writes one cluster region per centroid through the
 * {@link ClusterWriter} for the field's quantization scheme.
 *
 * <p>Only the ordinal partition is materialized for all centroids; each member's distance and secondary flag
 * are derived for one cluster at a time from the per-ordinal arrays, so the whole per-field distance/flag
 * copies are never held. When the field is rotated (L2), the vectors and centroids are rotated on the fly as
 * they are read, so no rotated copy of the field is materialized either.
 */
public final class ClusterAnnPostingsWriter implements PostingsWriter {

    private final int blockSize;
    private final QuantizationParams quantizationParams;
    private final Rotation rotation;

    public ClusterAnnPostingsWriter(final int blockSize, final QuantizationParams quantizationParams, final Rotation rotation) {
        this.blockSize = blockSize;
        this.quantizationParams = quantizationParams;
        this.rotation = rotation;
    }

    /**
     * Writes a field's postings to {@code .clap} — one cluster region per centroid, in centroid-id order —
     * and returns their offsets/lengths as {@link PostingsRegions}. {@code out} is owned by the caller: this
     * writes into it but never closes it.
     */
    @Override
    public PostingsRegions write(
        final IndexOutput out,
        final ClusteringResult clusters,
        final FloatVectorValues vectors,
        final VectorSimilarityFunction metric
    ) throws IOException {
        final long clapOffset = out.getFilePointer();
        final int numCentroids = clusters.numCentroids();
        final CentroidMembers[] postings = partitionByCentroid(clusters);

        final long[] centroidOffsets = new long[numCentroids];
        final int[] centroidLengths = new int[numCentroids];
        final int[] clusterSizes = new int[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            final ClusterMembers members = sortMembersByDistance(clusters, postings[c], metric);
            final ClusterRegion region = writeCluster(out, members, clusters.centroids()[c], vectors, metric);
            // Relative to this field's .clap start (clapOffset): the reader slices the region and seeks within it.
            centroidOffsets[c] = region.startOffset() - clapOffset;
            centroidLengths[c] = clusterRegionLength(c, region.length());
            clusterSizes[c] = postings[c].ordinals().length;
        }

        final long clapLength = out.getFilePointer() - clapOffset;
        return new PostingsRegions(clapOffset, clapLength, centroidOffsets, centroidLengths, clusterSizes);
    }

    /** The region length as an int (its {@code .clap}/{@code .clam} type), throwing rather than truncating if it overflows. */
    private static int clusterRegionLength(final int centroid, final long length) {
        try {
            return Math.toIntExact(length);
        } catch (final ArithmeticException e) {
            throw new IllegalStateException(
                "cluster " + centroid + " region is " + length + " bytes, exceeding the int .clap length limit",
                e
            );
        }
    }

    /** Writes one cluster's region — centroid and members rotated into the same space — and returns where it landed. */
    private ClusterRegion writeCluster(
        final IndexOutput out,
        final ClusterMembers members,
        final float[] centroid,
        final FloatVectorValues vectors,
        final VectorSimilarityFunction metric
    ) throws IOException {
        final int dimension = vectors.dimension();
        final float[] preparedCentroid = rotateCentroid(rotation, centroid, dimension);
        final ClusterWriter clusterWriter = ClusterWriterFactory.newWriter(
            quantizationParams,
            blockSize,
            dimension,
            metric,
            preparedCentroid
        );
        final FloatVectorValues clusterVectors = new PreparedFloatVectorValues(vectors, members.ordinals(), rotation);
        return clusterWriter.write(out, clusterVectors, members);
    }

    /**
     * Orders one centroid's members for pruning: compute each member's distance to the centroid, find the
     * permutation that orders them ({@code farthestFirst} sets the direction), then reorder ordinals +
     * distances + secondary flags through it. The secondary flag already came from the partition, so only the
     * distance is computed here.
     */
    private static ClusterMembers sortMembersByDistance(
        final ClusteringResult clusters,
        final CentroidMembers posting,
        final VectorSimilarityFunction metric
    ) {
        final float[] distances = memberDistances(clusters, posting);
        final int[] order = sortOrderByDistance(distances, metric);
        return reorder(posting, distances, order);
    }

    /**
     * Each member's distance to its centroid, in the posting's order — the sort key. A secondary member
     * (marked during the secondary-assignment scatter) takes its secondary distance; a primary takes its
     * primary distance. These are squared-L2 distances (lower = closer) for every metric —
     * {@code ClusterBuilder} records L2 regardless of similarity — not metric scores;
     * {@link NearestFirstDistanceSorter} picks the pruning direction from the metric.
     */
    private static float[] memberDistances(final ClusteringResult clusters, final CentroidMembers posting) {
        final float[] primaryDistances = clusters.distances();
        final float[] soarDistances = clusters.soarDistances();
        final int[] ordinals = posting.ordinals();
        final boolean[] secondary = posting.secondary();

        final float[] distances = new float[ordinals.length];
        for (int k = 0; k < ordinals.length; k++) {
            final int ord = ordinals[k];
            distances[k] = secondary[k] ? soarDistances[ord] : primaryDistances[ord];
        }
        return distances;
    }

    /**
     * The permutation that orders members by {@code distances} in {@code metric}'s pruning order. Ties are
     * not order-preserving.
     */
    private static int[] sortOrderByDistance(final float[] distances, final VectorSimilarityFunction metric) {
        final int[] order = new int[distances.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        NearestFirstDistanceSorter.sort(order, distances, metric);
        return order;
    }

    /** Reorders the posting's members and their distances through {@code order} into nearest-first {@link ClusterMembers}. */
    private static ClusterMembers reorder(final CentroidMembers posting, final float[] distances, final int[] order) {
        final int[] ordinals = posting.ordinals();
        final boolean[] secondary = posting.secondary();
        final int[] sortedOrdinals = new int[order.length];
        final float[] sortedDistances = new float[order.length];
        final boolean[] sortedSecondary = new boolean[order.length];
        for (int i = 0; i < order.length; i++) {
            final int from = order[i];
            sortedOrdinals[i] = ordinals[from];
            sortedDistances[i] = distances[from];
            sortedSecondary[i] = secondary[from];
        }
        return new ClusterMembers(sortedOrdinals, sortedDistances, sortedSecondary);
    }

    /** Rotates a centroid into quantization space; an unrotated field's {@code IdentityRotation} makes this a copy. */
    private static float[] rotateCentroid(final Rotation rotation, final float[] centroid, final int dimension) throws IOException {
        final float[] rotated = new float[dimension];
        rotation.rotate(centroid, rotated);
        return rotated;
    }

    /** One centroid's members from the partition: parallel ordinals and their secondary-assignment flags, in scatter order. */
    private record CentroidMembers(int[] ordinals, boolean[] secondary) {
    }

    /**
     * Partitions ordinals by centroid: {@code postings[c]} is centroid {@code c}'s members — its primary
     * assignments plus its secondary assignments — each ordinal carrying the secondary flag of the pass it was
     * scattered in. Ordinals with no assignment (a negative or out-of-range id) are skipped.
     *
     * <p>Counting sort in two steps: first tally how many members each centroid gets (primary + secondary
     * assignments) so each posting can be sized exactly, then place each member into its centroid's posting —
     * the primary assignments first (secondary flag false), then the secondary assignments (flag true).
     */
    private static CentroidMembers[] partitionByCentroid(final ClusteringResult clusters) {
        final int numCentroids = clusters.numCentroids();

        final int[] memberCounts = new int[numCentroids];
        addMemberCounts(memberCounts, clusters.assignments());
        addMemberCounts(memberCounts, clusters.soarAssignments());

        final CentroidMembers[] postings = new CentroidMembers[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            postings[c] = new CentroidMembers(new int[memberCounts[c]], new boolean[memberCounts[c]]);
        }

        final int[] nextPosition = new int[numCentroids];
        addMembersToPostings(postings, nextPosition, clusters.assignments(), false);
        addMembersToPostings(postings, nextPosition, clusters.soarAssignments(), true);
        return postings;
    }

    /** Counts, per centroid, how many ordinals {@code assignments} sends to it, adding into {@code memberCounts}. */
    private static void addMemberCounts(final int[] memberCounts, final int[] assignments) {
        for (final int c : assignments) {
            if (c >= 0 && c < memberCounts.length) {
                memberCounts[c]++;
            }
        }
    }

    /**
     * Places each ordinal into its centroid's posting, tagging whether it is a secondary assignment.
     * {@code nextPosition} tracks the next free position in each centroid's posting across the two passes (the
     * primary assignments, then the secondary assignments).
     */
    private static void addMembersToPostings(
        final CentroidMembers[] postings,
        final int[] nextPosition,
        final int[] assignments,
        final boolean secondary
    ) {
        for (int ord = 0; ord < assignments.length; ord++) {
            final int c = assignments[ord];
            if (c >= 0 && c < postings.length) {
                final int position = nextPosition[c]++;
                postings[c].ordinals()[position] = ord;
                postings[c].secondary()[position] = secondary;
            }
        }
    }
}
