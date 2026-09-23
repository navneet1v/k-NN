/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

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

    /**
     * 2-bit doc × 4-bit query at a byte {@code offset} into {@code docs}.
     *
     * <p>Eight bytes at a time, as {@link #bit} already did: a query plane paired with a doc plane becomes one
     * {@code Long.bitCount} over 64 codes rather than eight {@code Integer.bitCount} calls over 8. The pairing itself is
     * unavoidable — four query planes against two doc planes is eight popcounts per stripe either way — but taking it a
     * byte at a time cost another factor of eight on top, which is why 2-bit search was far slower than the ~2x its
     * extra plane implies.
     *
     * <p>Each pair accumulates separately and is weighted once at the end, so the multiplies leave the inner loop: query
     * plane {@code i} carries {@code 2^i} and doc plane {@code j} carries {@code 2^j}, so a pair is worth
     * {@code 2^(i+j)}.
     */
    public static float dibit(byte[] query, byte[] docs, int offset, int len) {
        final int stripeSize = len / 2;
        long q0d0 = 0, q0d1 = 0, q1d0 = 0, q1d1 = 0, q2d0 = 0, q2d1 = 0, q3d0 = 0, q3d1 = 0;
        int r = 0;
        for (final int upperBound = stripeSize & -Long.BYTES; r < upperBound; r += Long.BYTES) {
            final long d0 = (long) LONG_LE.get(docs, offset + r);
            final long d1 = (long) LONG_LE.get(docs, offset + r + stripeSize);
            final long p0 = (long) LONG_LE.get(query, r);
            final long p1 = (long) LONG_LE.get(query, r + stripeSize);
            final long p2 = (long) LONG_LE.get(query, r + stripeSize * 2);
            final long p3 = (long) LONG_LE.get(query, r + stripeSize * 3);
            q0d0 += Long.bitCount(p0 & d0);
            q0d1 += Long.bitCount(p0 & d1);
            q1d0 += Long.bitCount(p1 & d0);
            q1d1 += Long.bitCount(p1 & d1);
            q2d0 += Long.bitCount(p2 & d0);
            q2d1 += Long.bitCount(p2 & d1);
            q3d0 += Long.bitCount(p3 & d0);
            q3d1 += Long.bitCount(p3 & d1);
        }
        for (; r < stripeSize; r++) {
            final int d0 = docs[offset + r] & 0xFF, d1 = docs[offset + r + stripeSize] & 0xFF;
            final int p0 = query[r] & 0xFF, p1 = query[r + stripeSize] & 0xFF;
            final int p2 = query[r + stripeSize * 2] & 0xFF, p3 = query[r + stripeSize * 3] & 0xFF;
            q0d0 += Integer.bitCount(p0 & d0);
            q0d1 += Integer.bitCount(p0 & d1);
            q1d0 += Integer.bitCount(p1 & d0);
            q1d1 += Integer.bitCount(p1 & d1);
            q2d0 += Integer.bitCount(p2 & d0);
            q2d1 += Integer.bitCount(p2 & d1);
            q3d0 += Integer.bitCount(p3 & d0);
            q3d1 += Integer.bitCount(p3 & d1);
        }
        return q0d0 + (q0d1 + q1d0) * 2L + (q1d1 + q2d0) * 4L + (q2d1 + q3d0) * 8L + q3d1 * 16L;
    }
}
