/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.clusterann.DistanceMetric;

/**
 * Per-field cached state for ClusterANN index structures. Loaded lazily from
 * the metadata file (.clam) and centroid file (.clac) on first access per field.
 *
 * <p>Immutable after construction — safe for concurrent reads across queries.
 */
final class ClusterANNFieldState {

    final int fieldNumber;
    final int numVectors;
    final int dimension;
    final int numCentroids;
    final DistanceMetric metric;
    final byte docBits;
    final long centroidsOffset;
    final long postingsOffset;
    final long quantizedOffset;

    // Lazily loaded centroids (numCentroids × dimension)
    volatile float[][] centroids;

    // Per-centroid posting list offsets (loaded from offset table in .clap)
    volatile long[] primaryPostingOffsets;
    volatile long[] soarPostingOffsets;

    private ClusterANNFieldState(
        int fieldNumber,
        int numVectors,
        int dimension,
        int numCentroids,
        DistanceMetric metric,
        byte docBits,
        long centroidsOffset,
        long postingsOffset,
        long quantizedOffset
    ) {
        this.fieldNumber = fieldNumber;
        this.numVectors = numVectors;
        this.dimension = dimension;
        this.numCentroids = numCentroids;
        this.metric = metric;
        this.docBits = docBits;
        this.centroidsOffset = centroidsOffset;
        this.postingsOffset = postingsOffset;
        this.quantizedOffset = quantizedOffset;
    }

    /**
     * Load centroids from disk into memory (thread-safe lazy init).
     */
    void ensureCentroidsLoaded(IndexInput centroidsInput) throws IOException {
        if (centroids != null) return;
        synchronized (this) {
            if (centroids != null) return;
            float[][] c = new float[numCentroids][dimension];
            centroidsInput.seek(centroidsOffset);
            for (int i = 0; i < numCentroids; i++) {
                for (int d = 0; d < dimension; d++) {
                    c[i][d] = Float.intBitsToFloat(centroidsInput.readInt());
                }
            }
            centroids = c;
        }
    }

    /**
     * Load per-centroid posting offsets from the offset table at the end of .clap.
     * The last 8 bytes of the postings data (before codec footer) contain the offset table position.
     */
    void ensurePostingOffsetsLoaded(IndexInput postingsInput) throws IOException {
        if (primaryPostingOffsets != null) return;
        synchronized (this) {
            if (primaryPostingOffsets != null) return;

            // Read offset table position (written as last long before footer)
            // The offset table starts at a position stored at: postingsOffset + (data length)
            // We stored offsetTablePos as the very last long in the postings data
            // Find it by reading the offset table from the known structure:
            // After all posting lists, there's: primaryOffsets[C] + soarOffsets[C] + offsetTablePos
            // Total offset table size = (2*C + 1) * 8 bytes
            // offsetTablePos is at the end, just before codec footer

            // We need to find where the offset table starts.
            // The writer wrote offsetTablePos as the last long.
            // We can compute: end of data = postingsOffset + all posting data
            // But simpler: seek to end - footerLength - 8, read offsetTablePos
            long footerLength = org.apache.lucene.codecs.CodecUtil.footerLength();
            long fileLength = postingsInput.length();
            long offsetTablePosLocation = fileLength - footerLength - 8;
            postingsInput.seek(offsetTablePosLocation);
            long offsetTablePos = postingsInput.readLong();

            // Read offset arrays
            postingsInput.seek(offsetTablePos);
            long[] primary = new long[numCentroids];
            for (int c = 0; c < numCentroids; c++) {
                primary[c] = postingsInput.readLong();
            }
            long[] soar = new long[numCentroids];
            for (int c = 0; c < numCentroids; c++) {
                soar[c] = postingsInput.readLong();
            }
            primaryPostingOffsets = primary;
            soarPostingOffsets = soar;
        }
    }

    boolean isEmpty() {
        return numVectors == 0 || numCentroids == 0;
    }

    /**
     * Read all field states from the metadata file.
     *
     * @return map from field number to field state
     */
    static Map<Integer, ClusterANNFieldState> readAll(IndexInput metaInput, SegmentReadState state) throws IOException {
        // Note: codec header already consumed by reader's openInput()

        Map<Integer, ClusterANNFieldState> fields = new HashMap<>();
        while (true) {
            int fieldNumber = metaInput.readInt();
            if (fieldNumber == ClusterANN1040KnnVectorsWriter.END_OF_FIELDS) break;

            int numVectors = metaInput.readInt();
            int dimension = metaInput.readInt();
            int numCentroids = metaInput.readInt();
            String metricName = metaInput.readString();
            byte docBits = metaInput.readByte();
            long centroidsOffset = metaInput.readLong();
            long postingsOffset = metaInput.readLong();
            long quantizedOffset = metaInput.readLong();

            DistanceMetric metric;
            try {
                metric = DistanceMetric.valueOf(metricName);
            } catch (IllegalArgumentException e) {
                metric = DistanceMetric.L2;
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
                    centroidsOffset,
                    postingsOffset,
                    quantizedOffset
                )
            );
        }
        return fields;
    }
}
