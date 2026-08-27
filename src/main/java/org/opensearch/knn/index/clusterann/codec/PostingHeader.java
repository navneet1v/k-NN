/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * One posting's metadata: the columns that precede its vector payload in {@code .clap}, plus where that
 * payload begins.
 *
 * <p>These columns are IVF fundamentals rather than artifacts of a quantizer — {@code pos → ord}
 * membership, SOAR dedup, and per-vector distance to the centroid (SOAR being the de-facto standard) —
 * so parsing them is shared by every encoding family. Keeping it here as a plain parse step, rather than
 * behind an interface, lets a future family reuse it without anything storage-specific escaping into the
 * search path.
 *
 * @param ordinals            {@code pos → global vector ord}, in stored (distance-sorted) order
 * @param distancesToCentroid {@code ‖c−v‖} per position, parallel to {@code ordinals}; feeds geometry pruning
 * @param payloadOffset       where this posting's vector payload starts, past these columns
 */
record PostingHeader(int[] ordinals, float[] distancesToCentroid, long payloadOffset) {

    /**
     * Parse the header of the posting at {@code postingOffset} in one sequential pass on a private
     * clone, leaving the shared input's pointer untouched. Sequential on purpose: the target is a
     * buffer-pool / EFS-backed directory where prefetch matters and back-seeks are costly.
     */
    /**
     * On-disk size of a {@code count}-vector header — what {@link #parse} will read. Slightly
     * over-estimates the SOAR word-count {@code vInt}, which is fine for its only use: sizing a prefetch
     * hint. Kept here because this type owns the header's layout.
     */
    static long byteLength(int count) {
        long soarWords = (count + 63) / 64;
        return (long) count * Integer.BYTES        // ordinals
            + Integer.BYTES                       // soar word-count vInt (over-estimated)
            + soarWords * Long.BYTES              // soar bitset
            + (long) count * Float.BYTES;         // ‖c−v‖ column
    }

    static PostingHeader parse(IndexInput clap, long postingOffset, int count) throws IOException {
        IndexInput in = clap.clone();
        in.seek(postingOffset);
        int[] ordinals = new int[count];
        in.readInts(ordinals, 0, count);
        int soarWords = in.readVInt();
        in.skipBytes((long) soarWords * Long.BYTES);        // SOAR bitset (position-based; unused for now)
        float[] distancesToCentroid = new float[count];
        for (int i = 0; i < count; i++) {
            distancesToCentroid[i] = Float.intBitsToFloat(in.readInt());
        }
        return new PostingHeader(ordinals, distancesToCentroid, in.getFilePointer());
    }
}
