/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Compressed encoding for posting list document IDs. Uses delta encoding with
 * variable-width packing based on the maximum delta value.
 *
 * <p>Format:
 * <ul>
 *   <li>count (VInt)</li>
 *   <li>encoding byte: -1 = continuous (no deltas stored), 0 = all same, else = bitsPerValue</li>
 *   <li>min docId (VInt)</li>
 *   <li>packed deltas (bit-packed at bitsPerValue)</li>
 * </ul>
 *
 * <p>For continuous IDs (0, 1, 2, ..., n-1), only the count and min are stored (2 VInts).
 * For clustered IDs with small deltas, bit-packing reduces size significantly vs VInt per ID.
 */
public final class PostingListCodec {

    private static final byte CONTINUOUS = (byte) -1;
    private static final byte ALL_SAME = (byte) 0;

    private PostingListCodec() {}

    /**
     * Write a posting list with compressed doc IDs.
     *
     * @param docIds sorted document IDs
     * @param out    output to write to
     */
    public static void write(int[] docIds, DataOutput out) throws IOException {
        int count = docIds.length;
        out.writeVInt(count);
        if (count == 0) return;

        int min = docIds[0];

        // Check if continuous (0, 1, 2, ... or min, min+1, min+2, ...)
        boolean continuous = true;
        for (int i = 1; i < count; i++) {
            if (docIds[i] != min + i) {
                continuous = false;
                break;
            }
        }

        if (continuous) {
            out.writeByte(CONTINUOUS);
            out.writeVInt(min);
            return;
        }

        // Compute deltas and find max delta
        int[] deltas = new int[count];
        deltas[0] = 0;
        int maxDelta = 0;
        for (int i = 1; i < count; i++) {
            deltas[i] = docIds[i] - min - i; // delta from expected continuous position
            maxDelta = Math.max(maxDelta, deltas[i]);
        }

        if (maxDelta == 0) {
            // All same offset pattern (shouldn't happen with sorted unique IDs, but handle it)
            out.writeByte(ALL_SAME);
            out.writeVInt(min);
            return;
        }

        // Determine bits per value
        int bpv = 32 - Integer.numberOfLeadingZeros(maxDelta);
        out.writeByte((byte) bpv);
        out.writeVInt(min);

        // Bit-pack deltas
        long buffer = 0;
        int bitsInBuffer = 0;
        for (int i = 1; i < count; i++) {
            buffer |= ((long) deltas[i]) << bitsInBuffer;
            bitsInBuffer += bpv;
            while (bitsInBuffer >= 8) {
                out.writeByte((byte) (buffer & 0xFF));
                buffer >>>= 8;
                bitsInBuffer -= 8;
            }
        }
        // Flush remaining bits
        if (bitsInBuffer > 0) {
            out.writeByte((byte) (buffer & 0xFF));
        }
    }

    /**
     * Read a posting list with compressed doc IDs.
     *
     * @param in input to read from
     * @return sorted document IDs
     */
    public static int[] read(IndexInput in) throws IOException {
        int count = in.readVInt();
        if (count == 0) return new int[0];

        byte encoding = in.readByte();
        int min = in.readVInt();

        int[] docIds = new int[count];

        if (encoding == CONTINUOUS) {
            for (int i = 0; i < count; i++) {
                docIds[i] = min + i;
            }
            return docIds;
        }

        if (encoding == ALL_SAME) {
            for (int i = 0; i < count; i++) {
                docIds[i] = min + i;
            }
            return docIds;
        }

        // Bit-packed deltas
        int bpv = encoding & 0xFF;
        long mask = (1L << bpv) - 1;
        docIds[0] = min;

        long buffer = 0;
        int bitsInBuffer = 0;
        for (int i = 1; i < count; i++) {
            while (bitsInBuffer < bpv) {
                buffer |= ((long) (in.readByte() & 0xFF)) << bitsInBuffer;
                bitsInBuffer += 8;
            }
            int delta = (int) (buffer & mask);
            buffer >>>= bpv;
            bitsInBuffer -= bpv;
            docIds[i] = min + i + delta;
        }

        return docIds;
    }
}
