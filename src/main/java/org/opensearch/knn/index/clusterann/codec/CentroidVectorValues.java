/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Objects;

/**
 * The {@code .clac} centroid-block codec: the single source of truth for that block's on-disk
 * layout. It is both the read view ({@link FloatVectorValues} over the raw region, used for probe
 * ranking) and the write/offset side ({@link #write} serializes the block, {@link #transformedOffset}
 * locates the transformed region). Keeping write, read, and offset math in one class means the
 * layout is defined once. Ordinals are dense and identity-mapped to "docIds" ({@code ord == docId})
 * since centroids are not real documents.
 *
 * <p><b>Centroid-block layout</b> (per field, at {@code clacCentroidOffset}). The trailing squared
 * L2 norm {@code ‖c‖²} is stored where it is consumed: the raw region carries it only for L2 (the
 * ranking norm-trick); the transformed region always carries it (ADC reads the transformed centroid
 * and IP/cosine ADC needs {@code ‖c‖²}). The orthonormal rotation preserves the norm, so raw and
 * transformed norms are equal.
 * <pre>
 *   [raw region:         numCentroids × (dimension [+1 if L2]) × 4 bytes]  // [vec..]([‖c‖²])
 *   [transformed region: numCentroids × (dimension + 1)       × 4 bytes]  // [Rvec..][‖c‖²]
 * </pre>
 *
 * <p>The read view hides the {@code .clac} {@link IndexInput} and offset arithmetic, and stays
 * metric-agnostic — it is only a vector store, so the caller decides how to interpret the vectors.
 * {@link #vectorValue} reads a centroid's vector and, when the layout stores it, the trailing
 * {@code ‖c‖²} in a single seek; that norm is then available via {@link #norm()}. Callers compute
 * the geometric quantities ranking and CLIP need ({@code ⟨q,c⟩}, {@code ‖q−c‖}) from these. Each
 * centroid occupies {@code dimension (+1 if the norm is stored)} floats.
 *
 * <p>The backing input is a bounded {@link IndexInput#slice slice} of exactly the centroid region,
 * so a stray read cannot reach the rotation / region-1 data that follow — it throws
 * {@code EOFException} instead. Ordinals are also range-checked in {@link #vectorValue}.
 *
 * <p>Built per query and used single-threaded; {@link #vectorValue} returns a reused buffer and
 * caches the norm as state, so read {@link #norm()} and consume the buffer before the next call.
 */
public final class CentroidVectorValues extends FloatVectorValues {

    private final IndexInput input;
    private final int numCentroids;
    private final int dimension;
    private final int stride;
    private final boolean hasNorm;
    private final float[] buffer;
    private float lastNorm; // ‖c‖² of the centroid last returned by vectorValue (0 if none stored)

    /**
     * @param input        the .clac input; a bounded slice of the centroid region is taken internally
     *                     (the slice is not owned/closed by this instance).
     * @param offset       start of the raw centroid region in {@code input} (= clacCentroidOffset).
     * @param numCentroids centroid count.
     * @param dimension    vector dimension.
     * @param hasNorm      storage-layout flag: whether each centroid stores a trailing {@code ‖c‖²}
     *                     (written only for L2). Affects stride and what {@link #norm()} returns;
     *                     this type does not interpret the metric — that's the caller's concern.
     */
    public CentroidVectorValues(IndexInput input, long offset, int numCentroids, int dimension,
                                boolean hasNorm) throws IOException {
        this(
            input.slice("clac-centroids", offset, (long) numCentroids * (dimension + (hasNorm ? 1 : 0)) * Float.BYTES),
            numCentroids,
            dimension,
            hasNorm
        );
    }

    /** Wraps an already-sliced centroid region (used by {@link #copy()}). */
    private CentroidVectorValues(IndexInput regionSlice, int numCentroids, int dimension, boolean hasNorm) {
        this.input = regionSlice;
        this.numCentroids = numCentroids;
        this.dimension = dimension;
        this.hasNorm = hasNorm;
        this.stride = dimension + (hasNorm ? 1 : 0);
        this.buffer = new float[dimension];
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return numCentroids;
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        Objects.checkIndex(ord, numCentroids);
        // Region-relative seek into the bounded slice; a read past the region throws EOFException.
        // Single seek: read the vector and, when stored, the trailing ‖c‖² that immediately follows.
        input.seek((long) ord * stride * Float.BYTES);
        input.readFloats(buffer, 0, dimension);
        lastNorm = hasNorm ? Float.intBitsToFloat(input.readInt()) : 0f;
        return buffer;
    }

    /**
     * Squared L2 norm {@code ‖c‖²} of the centroid returned by the most recent {@link #vectorValue}
     * call, read in the same seek as the vector.
     *
     * @throws IllegalStateException if this region stores no norm (i.e. non-L2 layout)
     */
    public float norm() {
        ensureHasNorm();
        return lastNorm;
    }

    private void ensureHasNorm() {
        if (!hasNorm) {
            throw new IllegalStateException("Centroids do not have stored norms");
        }
    }

    @Override
    public FloatVectorValues copy() {
        return new CentroidVectorValues(input.clone(), numCentroids, dimension, hasNorm);
    }

    // ===== Write side (centroid-block codec) =====

    /**
     * Write the raw + transformed centroid regions to {@code .clac} (no header — sizes come from
     * {@code .clam}). Each centroid is {@code dimension} floats, optionally followed by its squared
     * L2 norm.
     *
     * @param rawHasNorm whether the raw region carries the trailing norm (true only for L2). The
     *                   transformed region always carries it.
     */
    public static void write(IndexOutput output, float[][] centroids, float[][] transformedCentroids,
                             int numCentroids, int dimension, boolean rawHasNorm) throws IOException {
        writeCentroidRegion(output, centroids, numCentroids, dimension, rawHasNorm);
        writeCentroidRegion(output, transformedCentroids, numCentroids, dimension, true);
    }

    /** Write one region as {@code numCentroids × [dimension floats]([‖c‖² float])}. */
    private static void writeCentroidRegion(IndexOutput output, float[][] vectors, int numCentroids,
                                            int dimension, boolean withNorm) throws IOException {
        for (int c = 0; c < numCentroids; c++) {
            float normSq = 0f;
            for (int d = 0; d < dimension; d++) {
                float v = vectors[c][d];
                output.writeInt(Float.floatToIntBits(v));
                normSq += v * v;
            }
            if (withNorm) {
                output.writeInt(Float.floatToIntBits(normSq));
            }
        }
    }

    /**
     * Start offset of a field's transformed-centroid region, computed from field layout alone
     * (query-independent). The transformed region follows the raw region, whose stride is
     * {@code dimension (+1 if rawHasNorm)}.
     */
    public static long transformedOffset(long clacCentroidOffset, int numCentroids, int dimension, boolean rawHasNorm) {
        int rawStride = dimension + (rawHasNorm ? 1 : 0);
        return clacCentroidOffset + (long) numCentroids * rawStride * Float.BYTES;
    }
}
