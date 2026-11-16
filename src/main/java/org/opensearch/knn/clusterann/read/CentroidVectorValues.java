/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * One region of centroids, addressed by centroid ordinal.
 *
 * <p>The input <em>is</em> the region — a slice covering these centroids and nothing else — so a field's raw and
 * rotated centroids are two of these over two slices, rather than one cursor and an offset to keep straight.
 *
 * <p>A {@link FloatVectorValues} rather than a bespoke reader so the planner's sweep over centroids can use
 * Lucene's own scorers to rank them, and so {@link #copy} carries Lucene's meaning: a private cursor over the
 * same data. Centroids are not documents, so nothing here is doc-addressed — {@code ord} is a centroid ordinal
 * throughout.
 *
 * <p>Each centroid occupies {@code dimension} floats followed by one more holding ‖c‖² — the value the scorer needs
 * alongside the vector, which it would otherwise have to recompute per cluster. Every record carries it, whatever the
 * similarity: the ADC transforms for inner product and cosine need it as much as Euclidean does, and making it
 * conditional only moved the cost to a caller that had to measure it instead. The records are fixed width, so a
 * centroid is one seek away rather than a scan.
 *
 * <p><b>Not thread-safe.</b> It carries a moving file pointer and a buffer reused across calls, so a shared
 * instance must be {@link #copy}'d before use and the returned array consumed before the next call.
 */
public final class CentroidVectorValues extends FloatVectorValues {

    private final IndexInput input;
    private final int numCentroids;
    private final int dimension;

    /** Fixed width of one record, which is what makes a centroid reachable by arithmetic. */
    private final int bytesPerCentroid;

    /** Reused across calls; valid until the next {@link #vectorValue}. */
    private final float[] vector;

    private float norm;

    /**
     * @param input the region's own slice of {@code .clac}; this instance reads through it and moves its pointer
     * @param numCentroids centroids in the region
     * @param dimension the field's vector dimension; each record is this many floats plus one for ‖c‖²
     */
    public CentroidVectorValues(IndexInput input, int numCentroids, int dimension) {
        this.input = input;
        this.numCentroids = numCentroids;
        this.dimension = dimension;
        this.bytesPerCentroid = floatsPerCentroid(dimension) * Float.BYTES;
        this.vector = new float[dimension];
    }

    /**
     * Floats one centroid record occupies: the vector, plus the trailing ‖c‖² every record carries.
     *
     * <p>Stated here because this is the class that reads the record, and a caller slicing the region has to size it
     * with the same width — two statements of one layout is how a stride drifts.
     */
    public static int floatsPerCentroid(int dimension) {
        return dimension + 1;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return numCentroids;
    }

    /**
     * The centroid at {@code ord}. The array is reused, so a caller holding two of these at once is holding the
     * same array twice — read it before asking for another.
     */
    @Override
    public float[] vectorValue(int ord) throws IOException {
        if (ord < 0 || ord >= numCentroids) {
            throw new IndexOutOfBoundsException("Centroid ordinal must be in [0, " + numCentroids + "), got: " + ord);
        }
        input.seek((long) ord * bytesPerCentroid);
        input.readFloats(vector, 0, dimension);
        norm = Float.intBitsToFloat(input.readInt());
        return vector;
    }

    /**
     * The value stored alongside the centroid read by the last {@link #vectorValue} — the squared norm ‖c‖², which
     * is the form the ADC scorer consumes. Named for the field on disk rather than for the quantity, so read the
     * type it is handed to rather than the name here.
     */
    public float norm() {
        return norm;
    }

    /**
     * A private cursor over the same centroids. Clones the input rather than sharing it, so two callers cannot
     * move each other's file pointer or overwrite each other's buffer.
     *
     * <p>Narrows {@link FloatVectorValues#copy()} to this type: a copy of a centroid region is still one, and a
     * caller that wants {@link #norm()} would otherwise have to cast to reach it.
     */
    @Override
    public CentroidVectorValues copy() throws IOException {
        return new CentroidVectorValues(input.clone(), numCentroids, dimension);
    }
}
