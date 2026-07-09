/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.opensearch.knn.index.clusterann.*;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.END_OF_FIELDS;

/**
 * Per-field state read from .clam.
 *
 * <p>.clam per-field layout:
 * <pre>
 * fieldNumber:       int
 * numVectors:        int
 * dimension:         int
 * numCentroids:      int
 * metricName:        String
 * docBits:           byte
 * postingsOffset:    long
 * centroidDocCounts: numCentroids × int
 * centroidNorms:     numCentroids × float
 * postingSizes:      numCentroids × int   (exact bytes per centroid for prefetch)
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

    // Eager-loaded (small: per centroid)
    public final int[] centroidDocCounts;
    public final float[] centroidNorms;
    public final int[] postingSizes;

    // Lazy-loaded (large: dimension floats per centroid)
    public float[][] centroids;
    public long[] centroidOffsets;
    public org.opensearch.knn.index.clusterann.algorithm.RandomRotation randomRotation;
    public float[][] transformedCentroids;

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
        int[] postingSizes,
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
        this.postingSizes = postingSizes;
        this.centroidsFilePos = centroidsFilePos;
    }

    public boolean isEmpty() {
        return numVectors == 0;
    }

    public void ensureLoaded(IndexInput metaInput) throws IOException {
        if (centroids != null) return;

        metaInput.seek(centroidsFilePos);

        // Bulk read all centroids into flat buffer, then slice
        int totalFloats = numCentroids * dimension;
        float[] flat = new float[totalFloats];
        metaInput.readFloats(flat, 0, totalFloats);
        centroids = new float[numCentroids][];
        for (int c = 0; c < numCentroids; c++) {
            centroids[c] = new float[dimension];
            System.arraycopy(flat, c * dimension, centroids[c], 0, dimension);
        }

        centroidOffsets = new long[numCentroids];
        metaInput.readLongs(centroidOffsets, 0, numCentroids);

        // Read randomRotation
        randomRotation = org.opensearch.knn.index.clusterann.algorithm.RandomRotation.read(metaInput);

        // Read transformed centroids (used for ADC scoring)
        int totalFloats2 = numCentroids * dimension;
        float[] flat2 = new float[totalFloats2];
        metaInput.readFloats(flat2, 0, totalFloats2);
        transformedCentroids = new float[numCentroids][];
        for (int c = 0; c < numCentroids; c++) {
            transformedCentroids[c] = new float[dimension];
            System.arraycopy(flat2, c * dimension, transformedCentroids[c], 0, dimension);
        }
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
            int[] postingSizes = new int[numCentroids];
            if (numCentroids > 0) {
                metaInput.readInts(docCounts, 0, numCentroids);
                metaInput.readFloats(norms, 0, numCentroids);
                metaInput.readInts(postingSizes, 0, numCentroids);
            }

            long centroidsFilePos = metaInput.getFilePointer();

            // Read centroids + offset table + randomRotation + transformed centroids eagerly
            float[][] loadedCentroids = null;
            long[] loadedOffsets = null;
            org.opensearch.knn.index.clusterann.algorithm.RandomRotation loadedRandomRotation = null;
            float[][] loadedTransformedCentroids = null;
            if (numCentroids > 0) {
                int totalFloats = numCentroids * dimension;
                float[] flat = new float[totalFloats];
                metaInput.readFloats(flat, 0, totalFloats);
                loadedCentroids = new float[numCentroids][];
                for (int c = 0; c < numCentroids; c++) {
                    loadedCentroids[c] = new float[dimension];
                    System.arraycopy(flat, c * dimension, loadedCentroids[c], 0, dimension);
                }
                loadedOffsets = new long[numCentroids];
                metaInput.readLongs(loadedOffsets, 0, numCentroids);
                loadedRandomRotation = org.opensearch.knn.index.clusterann.algorithm.RandomRotation.read(metaInput);
                float[] flat2 = new float[totalFloats];
                metaInput.readFloats(flat2, 0, totalFloats);
                loadedTransformedCentroids = new float[numCentroids][];
                for (int c = 0; c < numCentroids; c++) {
                    loadedTransformedCentroids[c] = new float[dimension];
                    System.arraycopy(flat2, c * dimension, loadedTransformedCentroids[c], 0, dimension);
                }
            }

            ClusterANNFieldState fs = new ClusterANNFieldState(
                fieldNumber,
                numVectors,
                dimension,
                numCentroids,
                metric,
                docBits,
                postingsOffset,
                docCounts,
                norms,
                postingSizes,
                centroidsFilePos
            );
            fs.centroids = loadedCentroids;
            fs.centroidOffsets = loadedOffsets;
            fs.randomRotation = loadedRandomRotation;
            fs.transformedCentroids = loadedTransformedCentroids;
            fields.put(fieldNumber, fs);
        }
        return fields;
    }
}
