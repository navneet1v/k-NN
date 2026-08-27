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
 * <p>.clam per-field layout (centroid vectors live in .clac; rotation in .clar):
 * <pre>
 * fieldNumber:        int
 * numVectors:         int
 * dimension:          int
 * numCentroids:       int
 * metricName:         String
 * docBits:            byte
 * quantizerId:        byte                 (QuantizerType; scalar-only today)
 * postingsOffset:     long
 * postingsLength:     long                 (this field's .clap posting region byte length)
 * centroidDocCounts:  numCentroids × int
 * postingSizes:       numCentroids × int   (exact bytes per centroid for prefetch)
 * offsetTable:        numCentroids × long
 * clacCentroidOffset: long                 (.clac centroid block start)
 * rotationOffset:     long                 (.clar rotation start; -1 if none)
 * clacRegion1Offset:  long                 (.clac ordToDoc/ordToCentroid start)
 * </pre>
 */
public final class ClusterANNFieldState {

    public final int fieldNumber;
    public final int numVectors;
    public final int dimension;
    public final int numCentroids;
    public final DistanceMetric metric;
    public final byte docBits;
    public final QuantizerType quantizerType;
    public final long postingsOffset;
    // Byte length of this field's posting region in .clap (postings are contiguous from postingsOffset).
    public final long postingsLength;
    // Offset in .clac where this field's centroid block (CentroidVectorValues) begins; -1 if empty.
    public final long clacCentroidOffset;
    // Offset in .clar where this field's serialized rotation begins; -1 if no rotation (e.g. IP).
    public final long rotationOffset;
    // Offset in .clac where region 1 (ordToDoc DISI config + ordToCentroid) begins; -1 if empty.
    public final long clacRegion1Offset;

    // Eager-loaded (small: per centroid)
    public final int[] centroidDocCounts;
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
        QuantizerType quantizerType,
        long postingsOffset,
        long postingsLength,
        long clacCentroidOffset,
        long rotationOffset,
        long clacRegion1Offset,
        int[] centroidDocCounts,
        int[] postingSizes,
        long centroidsFilePos
    ) {
        this.fieldNumber = fieldNumber;
        this.numVectors = numVectors;
        this.dimension = dimension;
        this.numCentroids = numCentroids;
        this.metric = metric;
        this.docBits = docBits;
        this.quantizerType = quantizerType;
        this.postingsOffset = postingsOffset;
        this.postingsLength = postingsLength;
        this.clacCentroidOffset = clacCentroidOffset;
        this.rotationOffset = rotationOffset;
        this.clacRegion1Offset = clacRegion1Offset;
        this.centroidDocCounts = centroidDocCounts;
        this.postingSizes = postingSizes;
        this.centroidsFilePos = centroidsFilePos;
    }

    public boolean isEmpty() {
        return numVectors == 0;
    }

    public void ensureLoaded(IndexInput metaInput) throws IOException {
        // Offset table is loaded eagerly in readAll (offsetTable follows postingSizes directly now
        // that centroids/rotation/transformed live only in .clac). Retained for API compatibility.
        if (centroidOffsets != null) return;

        metaInput.seek(centroidsFilePos);
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
            QuantizerType quantizerType = QuantizerType.fromId(metaInput.readByte());
            long postingsOffset = metaInput.readLong();
            long postingsLength = metaInput.readLong();

            DistanceMetric metric;
            try {
                metric = DistanceMetric.valueOf(metricName);
            } catch (IllegalArgumentException e) {
                metric = DistanceMetric.L2;
            }

            // Read centroid stats (eager — small). Centroid norms now live in .clac (with the
            // centroid vectors), not here.
            int[] docCounts = new int[numCentroids];
            int[] postingSizes = new int[numCentroids];
            if (numCentroids > 0) {
                metaInput.readInts(docCounts, 0, numCentroids);
                metaInput.readInts(postingSizes, 0, numCentroids);
            }

            long centroidsFilePos = metaInput.getFilePointer();

            // Offset table then clac region-1 offset. Centroids and transformed centroids live in
            // .clac; the rotation lives in .clar (its offset is stored here, not duplicated).
            long[] loadedOffsets = null;
            long clacCentroidOffset = -1;
            long rotationOffset = -1;
            long clacRegion1Offset = -1;
            if (numCentroids > 0) {
                loadedOffsets = new long[numCentroids];
                metaInput.readLongs(loadedOffsets, 0, numCentroids);
                clacCentroidOffset = metaInput.readLong();
                rotationOffset = metaInput.readLong();
                clacRegion1Offset = metaInput.readLong();
            }

            ClusterANNFieldState fs = new ClusterANNFieldState(
                fieldNumber,
                numVectors,
                dimension,
                numCentroids,
                metric,
                docBits,
                quantizerType,
                postingsOffset,
                postingsLength,
                clacCentroidOffset,
                rotationOffset,
                clacRegion1Offset,
                docCounts,
                postingSizes,
                centroidsFilePos
            );
            fs.centroidOffsets = loadedOffsets;
            fields.put(fieldNumber, fs);
        }
        return fields;
    }
}
