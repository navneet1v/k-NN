/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

public final class Int4DotProduct {

    private Int4DotProduct() {}

    private static final VarHandle LONG_LE = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /** 1-bit doc × 4-bit query at a byte {@code offset} into {@code docs}. */
    public static float bit(byte[] query, byte[] docs, int offset, int len) {
        long sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        int r = 0;
        for (final int upperBound = len & -Long.BYTES; r < upperBound; r += Long.BYTES) {
            long d = (long) LONG_LE.get(docs, offset + r);
            sum0 += Long.bitCount((long) LONG_LE.get(query, r) & d);
            sum1 += Long.bitCount((long) LONG_LE.get(query, r + len) & d);
            sum2 += Long.bitCount((long) LONG_LE.get(query, r + len * 2) & d);
            sum3 += Long.bitCount((long) LONG_LE.get(query, r + len * 3) & d);
        }
        for (; r < len; r++) {
            int d = docs[offset + r] & 0xFF;
            sum0 += Integer.bitCount((query[r] & d) & 0xFF);
            sum1 += Integer.bitCount((query[r + len] & d) & 0xFF);
            sum2 += Integer.bitCount((query[r + len * 2] & d) & 0xFF);
            sum3 += Integer.bitCount((query[r + len * 3] & d) & 0xFF);
        }
        return sum0 + sum1 * 2L + sum2 * 4L + sum3 * 8L;
    }

    /** 2-bit doc × 4-bit query at a byte {@code offset} into {@code docs}. */
    public static float dibit(byte[] query, byte[] docs, int offset, int len) {
        int stripeSize = len / 2;
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int d0 = docs[offset + i] & 0xFF, d1 = docs[offset + i + stripeSize] & 0xFF;
            int q0 = query[i] & 0xFF, q1 = query[i + stripeSize] & 0xFF;
            int q2 = query[i + stripeSize * 2] & 0xFF, q3 = query[i + stripeSize * 3] & 0xFF;
            sum += Integer.bitCount(q0 & d0) + Integer.bitCount(q0 & d1) * 2L + Integer.bitCount(q1 & d0) * 2L + Integer.bitCount(q1 & d1)
                * 4L + Integer.bitCount(q2 & d0) * 4L + Integer.bitCount(q2 & d1) * 8L + Integer.bitCount(q3 & d0) * 8L + Integer.bitCount(
                    q3 & d1
                ) * 16L;
        }
        return sum;
    }
}
