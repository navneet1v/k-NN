/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.clusterann.DistanceMetric;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.codec.KNN1040Codec.ClusterANNFormatConstants.END_OF_FIELDS;

/**
 * Per-field state read from .clam. Format v2: centroids + offset table inline.
 *
 * <p>.clam per-field layout:
 * <pre>
 * fieldNumber:    int
 * numVectors:     int
 * dimension:      int
 * numCentroids:   int
 * metricName:     String
 * docBits:        byte
 * postingsOffset: long
 * centroids:      numCentroids × dimension × float (original order)
 * offsetTable:    numCentroids × long (original order)
 * </pre>
 */
final class ClusterANNFieldState {

    final int fieldNumber;
    final int numVectors;
    final int dimension;
    final int numCentroids;
    final DistanceMetric metric;
    final byte docBits;
    final long postingsOffset;

    // Lazy-loaded
    float[][] centroids;
    long[] centroidOffsets;

    private final long centroidsFilePos;

    private ClusterANNFieldState(
        int fieldNumber,
        int numVectors,
        int dimension,
        int numCentroids,
        DistanceMetric metric,
        byte docBits,
        long postingsOffset,
        long centroidsFilePos
    ) {
        this.fieldNumber = fieldNumber;
        this.numVectors = numVectors;
        this.dimension = dimension;
        this.numCentroids = numCentroids;
        this.metric = metric;
        this.docBits = docBits;
        this.postingsOffset = postingsOffset;
        this.centroidsFilePos = centroidsFilePos;
    }

    boolean isEmpty() {
        return numVectors == 0;
    }

    void ensureLoaded(IndexInput metaInput) throws IOException {
        if (centroids != null) return;

        metaInput.seek(centroidsFilePos);

        // Centroids (original order)
        centroids = new float[numCentroids][dimension];
        for (int c = 0; c < numCentroids; c++) {
            metaInput.readFloats(centroids[c], 0, dimension);
        }

        // Offset table (original order)
        centroidOffsets = new long[numCentroids];
        metaInput.readLongs(centroidOffsets, 0, numCentroids);
    }

    static Map<Integer, ClusterANNFieldState> readAll(IndexInput metaInput, SegmentReadState state) throws IOException {
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

            long centroidsFilePos = metaInput.getFilePointer();

            // Skip centroids + offset table
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
                    centroidsFilePos
                )
            );
        }
        return fields;
    }
}
