/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write;

import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.IdentityRotation;

import java.io.IOException;

/**
 * Serializes a field's centroid geometry to {@code .clac}: region 1 (the per-ordinal {@code ordToCentroid}
 * assignment), region 2 (each centroid's vector and squared norm), and — for rotated fields — region 3 (each
 * centroid rotated into the quantization space, with its squared norm). Returns the region offsets so
 * {@code .clam} can record them.
 *
 * <p>The squared norm stored beside each centroid is computed here from the centroid vector, so it is
 * persisted once at write time and never recomputed per query, for every metric. Region 2 centroids are
 * stored as the clustering produced them: raw (input space) for L2 and inner product, unit-norm for cosine.
 * Region 3 exists only when the field is rotated (a non-identity {@link Rotation}); it is skipped for an
 * unrotated field (the identity rotation), whose quantization space is the input space already held in
 * region 2, so its offset is {@link #NO_TRANSFORMED_REGION}.
 *
 * <pre>
 *   region 1  @ clacOffset             ordToCentroid: numVectors x int
 *   region 2  @ clacCentroidsOffset       per centroid: vector[dim] floats | normSq float
 *   region 3  @ clacRotatedCentroidsOffset  per centroid: rotatedVector[dim] floats | normSq float   (rotated only)
 * </pre>
 *
 * <p>Every value is a 4-byte int/float. Floats are written as {@code Float.floatToIntBits}. An empty field
 * (no vectors, no centroids) writes nothing and reports region offsets of {@code 0} with no region 3.
 *
 * <p>Stateless: one field's centroids are written by a single {@link #write} call, so it is a static utility.
 */
public final class CentroidsWriter {

    /** {@link CentroidOffsets#clacRotatedCentroidsOffset()} of a field with no region 3 (an unrotated field). */
    private static final long NO_TRANSFORMED_REGION = -1L;

    private CentroidsWriter() {}

    /**
     * Writes {@code centroids} to {@code out} at the current position and returns the region offsets. Region 3
     * is written only for a rotated field (a non-identity {@code rotation}); an unrotated field skips it and
     * reports {@link #NO_TRANSFORMED_REGION}.
     *
     * @param out       the {@code .clac} output, positioned at this field's region start
     * @param centroids this field's centroid geometry
     * @param rotation  the field's rotation, applied to each centroid for region 3; the identity rotation
     *     marks an unrotated field, which has no region 3
     * @return the region offsets for {@code .clam}
     */
    public static CentroidOffsets write(final IndexOutput out, final CentroidData centroids, final Rotation rotation) throws IOException {
        final long clacOffset = writeOrdToCentroid(out, centroids.ordToCentroid());
        final long clacCentroidOffset = writeCentroids(out, centroids) - clacOffset;
        final long roatated = maybeWriteRotatedCentroids(out, centroids, rotation);
        final long clacRotatedOffset = roatated == NO_TRANSFORMED_REGION ? NO_TRANSFORMED_REGION : roatated - clacOffset;
        return new CentroidOffsets(clacOffset, clacCentroidOffset, clacRotatedOffset);
    }

    /** Region 1: the primary centroid id per vector ordinal ({@code ordToCentroid}). Returns the region's start offset. */
    private static long writeOrdToCentroid(final IndexOutput out, final int[] ordToCentroid) throws IOException {
        final long offset = out.getFilePointer();
        for (final int centroid : ordToCentroid) {
            out.writeInt(centroid);
        }
        return offset;
    }

    /** Region 2: each centroid's vector and its (computed) squared norm. Returns the region's start offset. */
    private static long writeCentroids(final IndexOutput out, final CentroidData centroids) throws IOException {
        final long offset = out.getFilePointer();
        for (final float[] centroid : centroids.centroids()) {
            writeVectorWithNorm(out, centroid);
        }
        return offset;
    }

    /**
     * Region 3 (rotated fields only): each centroid rotated into the quantization space, with its (computed)
     * squared norm. Skipped for an unrotated field — the identity rotation would only copy region 2's
     * centroids — returning {@link #NO_TRANSFORMED_REGION}; otherwise returns the region's start offset.
     */
    private static long maybeWriteRotatedCentroids(final IndexOutput out, final CentroidData centroids, final Rotation rotation)
        throws IOException {
        if (rotation instanceof IdentityRotation) {
            return NO_TRANSFORMED_REGION;
        }
        final long offset = out.getFilePointer();
        for (final float[] centroid : centroids.centroids()) {
            final float[] rotated = new float[centroid.length];
            rotation.rotate(centroid, rotated);
            writeVectorWithNorm(out, rotated);
        }
        return offset;
    }

    /** Writes a vector followed by its squared norm ({@code ‖v‖²}), computed from the vector itself. */
    private static void writeVectorWithNorm(final IndexOutput out, final float[] vector) throws IOException {
        for (final float value : vector) {
            out.writeInt(Float.floatToIntBits(value));
        }
        out.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(vector, vector)));
    }

    /**
     * A field's centroid geometry, grouped by the {@code .clac} regions it serializes into. {@code centroids}
     * is indexed by centroid id, {@code ordToCentroid} by vector ordinal. Squared norms are not carried here —
     * {@link CentroidsWriter} computes them from the vectors — and neither are the rotated centroids, which the
     * writer derives by applying the field's rotation to {@code centroids}.
     *
     * @param ordToCentroid primary centroid id per vector ordinal (region 1)
     * @param centroids     centroid vectors, one row per centroid (region 2, and the source for region 3)
     */
    public record CentroidData(int[] ordToCentroid, float[][] centroids) {
    }

    /**
     * The {@code .clac} region offsets, returned so {@code .clam} can record them.
     *
     * @param clacOffset            absolute start of this field's {@code .clac} region, which begins with region 1
     *     (per-ordinal {@code ordToCentroid}) — the offset the reader cuts the field's region at
     * @param clacCentroidsOffset      start of region 2 (raw centroids: vector, normSq), <em>relative to</em>
     *     {@code clacOffset}
     * @param clacRotatedCentroidsOffset start of region 3 (rotated centroids: vector, normSq), relative to
     *     {@code clacOffset}, or {@link #NO_TRANSFORMED_REGION} for an unrotated field with no region 3
     */
    public record CentroidOffsets(long clacOffset, long clacCentroidsOffset, long clacRotatedCentroidsOffset) {
    }
}
