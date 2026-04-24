/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

import static org.opensearch.knn.index.codec.KNN1040Codec.ClusterANNFormatConstants.POSTING_CONTINUOUS;
import static org.opensearch.knn.index.codec.KNN1040Codec.ClusterANNFormatConstants.POSTING_GROUP_VINT;

/**
 * Encodes/decodes sorted integer arrays (doc IDs or ordinals) using delta + VInt.
 */
public final class PostingListCodec {

    private PostingListCodec() {}

    public static void write(int[] ids, DataOutput out) throws IOException {
        int count = ids.length;
        out.writeVInt(count);
        if (count == 0) return;

        boolean continuous = true;
        for (int i = 1; i < count; i++) {
            if (ids[i] != ids[0] + i) {
                continuous = false;
                break;
            }
        }
        if (continuous) {
            out.writeByte(POSTING_CONTINUOUS);
            out.writeVInt(ids[0]);
            return;
        }

        out.writeByte(POSTING_GROUP_VINT);
        out.writeVInt(ids[0]);
        for (int i = 1; i < count; i++) {
            out.writeVInt(ids[i] - ids[i - 1]);
        }
    }

    public static int[] read(IndexInput in) throws IOException {
        int count = in.readVInt();
        if (count == 0) return new int[0];
        int[] ids = new int[count];
        readInto(in, ids, count);
        return ids;
    }

    public static int read(IndexInput in, int[] buffer) throws IOException {
        int count = in.readVInt();
        if (count == 0) return 0;
        if (count > buffer.length) {
            // Buffer too small — allocate (shouldn't happen in normal operation)
            int[] tmp = new int[count];
            readInto(in, tmp, count);
            System.arraycopy(tmp, 0, buffer, 0, buffer.length);
            return count;
        }
        readInto(in, buffer, count);
        return count;
    }

    private static void readInto(IndexInput in, int[] ids, int count) throws IOException {
        byte encoding = in.readByte();
        if (encoding == POSTING_CONTINUOUS) {
            int start = in.readVInt();
            for (int i = 0; i < count; i++)
                ids[i] = start + i;
            return;
        }
        ids[0] = in.readVInt();
        for (int i = 1; i < count; i++) {
            ids[i] = ids[i - 1] + in.readVInt();
        }
    }
}
