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

    // Lazy-loaded (small: offsets only)
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
        if (centroidOffsets != null) return;

        metaInput.seek(centroidsFilePos);

        // Skip centroids — they are in .clac (off-heap)
        metaInput.skipBytes((long) numCentroids * dimension * Float.BYTES);

        // Read offset table
        centroidOffsets = new long[numCentroids];
        metaInput.readLongs(centroidOffsets, 0, numCentroids);

        // Skip randomRotation — it is in .clac (off-heap)
        // Format: [numBlocks:int][blockDim:int][blocks...][permutation...]
        // We need to skip the correct number of bytes
        long rotationStart = metaInput.getFilePointer();
        int numBlocks = metaInput.readInt();
        int blockDim = metaInput.readInt();
        for (int b = 0; b < numBlocks; b++) {
            int bDim = metaInput.readInt();
            metaInput.skipBytes((long) bDim * bDim * Float.BYTES);
        }
        for (int b = 0; b < numBlocks; b++) {
            int pLen = metaInput.readInt();
            metaInput.skipBytes((long) pLen * Integer.BYTES);
        }

        // Skip transformed centroids — they are in .clac (off-heap)
        metaInput.skipBytes((long) numCentroids * dimension * Float.BYTES);
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

            // Skip centroids, read offsets only, skip rotation + transformed centroids
            // Centroids and rotation are in .clac (off-heap)
            long[] loadedOffsets = null;
            if (numCentroids > 0) {
                // Skip raw centroids
                metaInput.skipBytes((long) numCentroids * dimension * Float.BYTES);
                // Read offset table
                loadedOffsets = new long[numCentroids];
                metaInput.readLongs(loadedOffsets, 0, numCentroids);
                // Skip rotation
                int numBlocks = metaInput.readInt();
                int blockDim = metaInput.readInt();
                for (int b = 0; b < numBlocks; b++) {
                    int bDim = metaInput.readInt();
                    metaInput.skipBytes((long) bDim * bDim * Float.BYTES);
                }
                for (int b = 0; b < numBlocks; b++) {
                    int pLen = metaInput.readInt();
                    metaInput.skipBytes((long) pLen * Integer.BYTES);
                }
                // Skip transformed centroids
                metaInput.skipBytes((long) numCentroids * dimension * Float.BYTES);
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
            fs.centroidOffsets = loadedOffsets;
            fields.put(fieldNumber, fs);
        }
        return fields;
    }
}
