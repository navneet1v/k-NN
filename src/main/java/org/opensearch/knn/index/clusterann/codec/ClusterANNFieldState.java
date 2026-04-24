/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.opensearch.knn.index.clusterann.*;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.clusterann.DistanceMetric;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.END_OF_FIELDS;

/**
 * Per-field state read from .clam.
 *
 * <p>.clam per-field layout:
 * <pre>
 * fieldNumber:      int
 * numVectors:       int
 * dimension:        int
 * numCentroids:     int
 * metricName:       String
 * docBits:          byte
 * postingsOffset:   long
 * centroidDocCounts: numCentroids × int   (docs per centroid, for filter estimation)
 * centroidNorms:     numCentroids × float (||c||² per centroid, for fast ADC correction)
 * centroids:         numCentroids × dimension × float
 * offsetTable:       numCentroids × long
 * </pre>
 */
public final class ClusterANNFieldState {

    public final int fieldNumber;
    public final int numVectors;
    public final int dimension;
    public final int numCentroids;
    public final DistanceMetric metric;
    public final byte docBits;
    public final long postingsOffset;

    // Eager-loaded (small: one int + one float per centroid)
    public final int[] centroidDocCounts;
    public final float[] centroidNorms;

    // Lazy-loaded (large: dimension floats per centroid)
    public float[][] centroids;
    public long[] centroidOffsets;

    private final long centroidsFilePos;

    private ClusterANNFieldState(
        int fieldNumber,
        int numVectors,
        int dimension,
        int numCentroids,
        DistanceMetric metric,
        byte docBits,
        long postingsOffset,
        int[] centroidDocCounts,
        float[] centroidNorms,
        long centroidsFilePos
    ) {
        this.fieldNumber = fieldNumber;
        this.numVectors = numVectors;
        this.dimension = dimension;
        this.numCentroids = numCentroids;
        this.metric = metric;
        this.docBits = docBits;
        this.postingsOffset = postingsOffset;
        this.centroidDocCounts = centroidDocCounts;
        this.centroidNorms = centroidNorms;
        this.centroidsFilePos = centroidsFilePos;
    }

    public boolean isEmpty() {
        return numVectors == 0;
    }

    public void ensureLoaded(IndexInput metaInput) throws IOException {
        if (centroids != null) return;

        metaInput.seek(centroidsFilePos);

        centroids = new float[numCentroids][dimension];
        for (int c = 0; c < numCentroids; c++) {
            metaInput.readFloats(centroids[c], 0, dimension);
        }

        centroidOffsets = new long[numCentroids];
        metaInput.readLongs(centroidOffsets, 0, numCentroids);
    }

    public static Map<Integer, ClusterANNFieldState> readAll(IndexInput metaInput, SegmentReadState state) throws IOException {
        Map<Integer, ClusterANNFieldState> fields = new HashMap<>();
        while (true) {
            int fieldNumber = metaInput.readInt();
            if (fieldNumber == END_OF_FIELDS) break;

            int numVectors = metaInput.readInt();
            int dimension = metaInput.readInt();
            int numCentroids = metaInput.readInt();
            String metricName = metaInput.readString();
            byte docBits = metaInput.readByte();
            long postingsOffset = metaInput.readLong();

            DistanceMetric metric;
            try {
                metric = DistanceMetric.valueOf(metricName);
            } catch (IllegalArgumentException e) {
                metric = DistanceMetric.L2;
            }

            // Read centroid stats (eager — small)
            int[] docCounts = new int[numCentroids];
            float[] norms = new float[numCentroids];
            if (numCentroids > 0) {
                metaInput.readInts(docCounts, 0, numCentroids);
                metaInput.readFloats(norms, 0, numCentroids);
            }

            long centroidsFilePos = metaInput.getFilePointer();

            // Skip centroids + offset table (lazy loaded)
            if (numCentroids > 0) {
                long centroidBytes = (long) numCentroids * dimension * Float.BYTES;
                long offsetBytes = (long) numCentroids * Long.BYTES;
                metaInput.seek(centroidsFilePos + centroidBytes + offsetBytes);
            }

            fields.put(
                fieldNumber,
                new ClusterANNFieldState(
                    fieldNumber,
                    numVectors,
                    dimension,
                    numCentroids,
                    metric,
                    docBits,
                    postingsOffset,
                    docCounts,
                    norms,
                    centroidsFilePos
                )
            );
        }
        return fields;
    }
}
