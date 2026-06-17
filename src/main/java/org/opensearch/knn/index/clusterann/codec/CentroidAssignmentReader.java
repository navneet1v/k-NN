/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.FixedBitSet;

import java.io.Closeable;
import java.io.IOException;

/**
 * Reads the .claf (cluster assignment for filter) file.
 * Provides ord→centroid mapping for filter-aware centroid selection.
 *
 * At query time with a filter:
 * 1. Iterate filter's matching docIDs
 * 2. Look up centroid assignment for each
 * 3. Build acceptCentroids BitSet + per-centroid match counts
 * 4. Rank centroids by distance / matchCount (filter-density weighted)
 */
public final class CentroidAssignmentReader implements Closeable {
    private final IndexInput input;
    private final int numVectors;
    private final int numCentroids;
    private final long dataStart;

    public CentroidAssignmentReader(IndexInput input, int expectedFieldNumber) throws IOException {
        this.input = input;
        int fieldNumber = input.readInt();
        if (fieldNumber != expectedFieldNumber) {
            throw new IOException("Field number mismatch in .claf: expected " + expectedFieldNumber + " got " + fieldNumber);
        }
        this.numVectors = input.readInt();
        this.numCentroids = input.readInt();
        this.dataStart = input.getFilePointer();
    }

    /**
     * Given a BitSet of accepted ordinals (from Lucene filter), compute:
     * - which centroids have at least one match (acceptCentroids)
     * - how many matches each centroid has (matchCounts)
     *
     * @param acceptedOrds BitSet of ordinals that pass the filter
     * @param matchCounts  output array [numCentroids] — populated with per-centroid counts
     * @return FixedBitSet of centroids that have at least one matching doc
     */
    public FixedBitSet computeCentroidFilter(FixedBitSet acceptedOrds, int[] matchCounts) throws IOException {
        FixedBitSet acceptCentroids = new FixedBitSet(numCentroids);
        // Read all assignments in one bulk read
        IndexInput slice = input.clone();
        slice.seek(dataStart);
        byte[] raw = new byte[numVectors * Short.BYTES];
        slice.readBytes(raw, 0, raw.length);

        for (int ord = acceptedOrds.nextSetBit(0); ord != -1 && ord < numVectors; ord = acceptedOrds.nextSetBit(ord + 1)) {
            int centroid = ((raw[ord * 2] & 0xFF) << 8) | (raw[ord * 2 + 1] & 0xFF);
            acceptCentroids.set(centroid);
            matchCounts[centroid]++;
        }
        return acceptCentroids;
    }

    public int numCentroids() { return numCentroids; }
    public int numVectors() { return numVectors; }

    @Override
    public void close() throws IOException {
        input.close();
    }
}
