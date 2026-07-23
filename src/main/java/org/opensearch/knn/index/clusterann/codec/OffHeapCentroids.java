/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.clusterann.algorithm.RandomRotation;

import java.io.Closeable;
import java.io.IOException;

/**
 * Off-heap centroid storage in .clac file.
 * All data accessed via mmap (IndexInput) — zero permanent heap usage.
 *
 * File layout per field:
 *   [fieldNumber: int]
 *   [numCentroids: int]
 *   [dimension: int]
 *   [raw centroids: numCentroids × dimension × 4 bytes]
 *   [transformed centroids: numCentroids × dimension × 4 bytes]
 *   [rotation data: RandomRotation serialized]
 *   [rotationPresent: byte (0 or 1)]
 */
public final class OffHeapCentroids {

    private OffHeapCentroids() {}

    /** Write centroids and rotation to .clac file. */
    public static void write(IndexOutput output, int fieldNumber, float[][] centroids,
                             float[][] transformedCentroids, int numCentroids, int dimension,
                             RandomRotation rotation) throws IOException {
        output.writeInt(fieldNumber);
        output.writeInt(numCentroids);
        output.writeInt(dimension);
        // Raw centroids
        for (int c = 0; c < numCentroids; c++) {
            for (int d = 0; d < dimension; d++) {
                output.writeInt(Float.floatToIntBits(centroids[c][d]));
            }
        }
        // Transformed centroids
        for (int c = 0; c < numCentroids; c++) {
            for (int d = 0; d < dimension; d++) {
                output.writeInt(Float.floatToIntBits(transformedCentroids[c][d]));
            }
        }
        // Rotation
        if (rotation != null) {
            output.writeByte((byte) 1);
            rotation.write(output);
        } else {
            output.writeByte((byte) 0);
        }
    }

    /** Reader for off-heap centroids and rotation. Thread-safe via IndexInput.clone(). */
    public static class Reader implements Closeable {
        private final IndexInput input;
        private final int numCentroids;
        private final int dimension;
        private final long rawDataStart;
        private final long transformedDataStart;
        private final long rotationStart;
        private final boolean hasRotation;

        public Reader(IndexInput input, int expectedFieldNumber) throws IOException {
            this.input = input;
            int fieldNumber = input.readInt();
            if (fieldNumber != expectedFieldNumber) {
                throw new IOException("Field mismatch in .clac: expected " + expectedFieldNumber + " got " + fieldNumber);
            }
            this.numCentroids = input.readInt();
            this.dimension = input.readInt();
            this.rawDataStart = input.getFilePointer();
            this.transformedDataStart = rawDataStart + (long) numCentroids * dimension * Float.BYTES;
            // Seek past transformed centroids to find rotation
            long rotationMarkerPos = transformedDataStart + (long) numCentroids * dimension * Float.BYTES;
            input.seek(rotationMarkerPos);
            this.hasRotation = input.readByte() == 1;
            this.rotationStart = input.getFilePointer();
        }

        public int numCentroids() { return numCentroids; }
        public int dimension() { return dimension; }
        public boolean hasRotation() { return hasRotation; }

        /** Read one raw centroid into buffer. Thread-safe. */
        public void readCentroid(int idx, float[] buffer) throws IOException {
            IndexInput slice = input.clone();
            slice.seek(rawDataStart + (long) idx * dimension * Float.BYTES);
            slice.readFloats(buffer, 0, dimension);
        }

        /** Read one transformed centroid into buffer. Thread-safe. */
        public void readTransformedCentroid(int idx, float[] buffer) throws IOException {
            IndexInput slice = input.clone();
            slice.seek(transformedDataStart + (long) idx * dimension * Float.BYTES);
            slice.readFloats(buffer, 0, dimension);
        }

        /** Read all raw centroids into flat buffer [numCentroids × dimension]. Thread-safe. */
        public void readAllCentroids(float[] flatBuffer) throws IOException {
            IndexInput slice = input.clone();
            slice.seek(rawDataStart);
            slice.readFloats(flatBuffer, 0, numCentroids * dimension);
        }

        /** Read all transformed centroids into flat buffer. Thread-safe. */
        public void readAllTransformedCentroids(float[] flatBuffer) throws IOException {
            IndexInput slice = input.clone();
            slice.seek(transformedDataStart);
            slice.readFloats(flatBuffer, 0, numCentroids * dimension);
        }

        /**
         * Apply rotation to query vector, reading rotation matrix from off-heap.
         * Reads rotation blocks from mmap on each call — no heap retention.
         */
        public void transformQuery(float[] vector, float[] out) throws IOException {
            if (!hasRotation) {
                System.arraycopy(vector, 0, out, 0, vector.length);
                return;
            }
            IndexInput slice = input.clone();
            slice.seek(rotationStart);
            int numBlocks = slice.readInt();
            int blockDim = slice.readInt();
            int outIdx = 0;

            // Read permutation offsets — we need permutation after blocks
            // First pass: compute blocks data size to find permutation
            long blocksStartPos = slice.getFilePointer();
            // Skip blocks to find permutation
            for (int b = 0; b < numBlocks; b++) {
                int bDim = slice.readInt();
                slice.skipBytes((long) bDim * bDim * Float.BYTES);
            }
            // Read permutations
            int[][] permutation = new int[numBlocks][];
            for (int b = 0; b < numBlocks; b++) {
                int pLen = slice.readInt();
                permutation[b] = new int[pLen];
                for (int j = 0; j < pLen; j++) {
                    permutation[b][j] = slice.readInt();
                }
            }

            // Second pass: read blocks and apply transform
            IndexInput blockSlice = input.clone();
            blockSlice.seek(blocksStartPos);
            float[] row = new float[blockDim];
            for (int b = 0; b < numBlocks; b++) {
                int bDim = blockSlice.readInt();
                int[] perm = permutation[b];
                for (int i = 0; i < bDim; i++) {
                    blockSlice.readFloats(row, 0, bDim);
                    float dot = 0f;
                    for (int j = 0; j < bDim; j++) {
                        dot += row[j] * vector[perm[j]];
                    }
                    out[outIdx + i] = dot;
                }
                outIdx += bDim;
            }
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}
