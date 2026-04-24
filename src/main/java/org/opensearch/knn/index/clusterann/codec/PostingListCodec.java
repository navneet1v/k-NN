/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Adaptive per-posting-list encoding for sorted integer arrays (doc IDs).
 *
 * <p>Chooses the tightest encoding for each posting list independently:
 * <ul>
 *   <li>CONTINUOUS — sequential IDs, store only start</li>
 *   <li>DELTA_FIXED16 — deltas fit in 16 bits, bulk decode via readShorts</li>
 *   <li>PACKED_32 — full ints, bulk decode via readInts</li>
 * </ul>
 */
public final class PostingListCodec {

    private static final byte ENCODING_EMPTY = 0;
    private static final byte ENCODING_CONTINUOUS = 1;
    private static final byte ENCODING_DELTA_FIXED16 = 2;
    private static final byte ENCODING_PACKED_32 = 3;

    private PostingListCodec() {}

    public static void write(int[] ids, DataOutput out) throws IOException {
        int count = ids.length;
        out.writeVInt(count);
        if (count == 0) return;

        // Check continuous
        boolean continuous = true;
        for (int i = 1; i < count; i++) {
            if (ids[i] != ids[0] + i) {
                continuous = false;
                break;
            }
        }
        if (continuous) {
            out.writeByte(ENCODING_CONTINUOUS);
            out.writeVInt(ids[0]);
            return;
        }

        // Check if deltas fit in unsigned 16-bit
        boolean fits16 = true;
        for (int i = 1; i < count; i++) {
            int delta = ids[i] - ids[i - 1];
            if (delta < 0 || delta > 0xFFFF) {
                fits16 = false;
                break;
            }
        }

        if (fits16) {
            out.writeByte(ENCODING_DELTA_FIXED16);
            out.writeInt(ids[0]);
            for (int i = 1; i < count; i++) {
                out.writeShort((short) (ids[i] - ids[i - 1]));
            }
        } else {
            out.writeByte(ENCODING_PACKED_32);
            for (int i = 0; i < count; i++) {
                out.writeInt(ids[i]);
            }
        }
    }

    public static int read(IndexInput in, int[] buffer) throws IOException {
        int count = in.readVInt();
        if (count == 0) return 0;
        if (count > buffer.length) {
            buffer = new int[count]; // shouldn't happen in normal operation
        }
        byte encoding = in.readByte();
        switch (encoding) {
            case ENCODING_CONTINUOUS -> {
                int start = in.readVInt();
                for (int i = 0; i < count; i++)
                    buffer[i] = start + i;
            }
            case ENCODING_DELTA_FIXED16 -> {
                buffer[0] = in.readInt();
                for (int i = 1; i < count; i++) {
                    buffer[i] = buffer[i - 1] + Short.toUnsignedInt(in.readShort());
                }
            }
            case ENCODING_PACKED_32 -> {
                in.readInts(buffer, 0, count);
            }
            default -> throw new IOException("Unknown posting encoding: " + encoding);
        }
        return count;
    }

    public static int[] read(IndexInput in) throws IOException {
        int count = in.readVInt();
        if (count == 0) return new int[0];
        int[] ids = new int[count];
        byte encoding = in.readByte();
        switch (encoding) {
            case ENCODING_CONTINUOUS -> {
                int start = in.readVInt();
                for (int i = 0; i < count; i++)
                    ids[i] = start + i;
            }
            case ENCODING_DELTA_FIXED16 -> {
                ids[0] = in.readInt();
                for (int i = 1; i < count; i++) {
                    ids[i] = ids[i - 1] + Short.toUnsignedInt(in.readShort());
                }
            }
            case ENCODING_PACKED_32 -> {
                in.readInts(ids, 0, count);
            }
            default -> throw new IOException("Unknown posting encoding: " + encoding);
        }
        return ids;
    }
}
