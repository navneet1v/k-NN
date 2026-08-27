/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * Bit-packed popcount dot-product kernels for ADC scoring: a 4-bit half-byte-transposed query dotted
 * against {@code docBits ∈ {1,2,4}} transposed doc codes. The query always has 4 stripes; the doc has
 * as many stripes as its bit width. Each kernel sums {@code popcount(query_stripe & doc_stripe)}
 * weighted by the product of the two stripes' bit significances, recovering the integer dot product
 * of the two quantized vectors.
 *
 * <p>The offset kernels dot one doc vector living at a byte {@code offset} into a packed block; the
 * {@code len}-argument is that vector's packed byte length (the query stride is derived from it).
 */
public final class Int4DotProduct {

    private Int4DotProduct() {}

    private static final VarHandle LONG_LE =
        MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /** 4-bit doc × 4-bit query over the full arrays (both half-byte transposed). */
    public static long int4NibbleDotProduct(byte[] query, byte[] doc) {
        return (long) nibble(query, doc, 0, doc.length);
    }

    /** 1-bit doc × 4-bit query at a byte {@code offset} into {@code docs}. */
    static float bit(byte[] query, byte[] docs, int offset, int len) {
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
    static float dibit(byte[] query, byte[] docs, int offset, int len) {
        int stripeSize = len / 2;
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int d0 = docs[offset + i] & 0xFF, d1 = docs[offset + i + stripeSize] & 0xFF;
            int q0 = query[i] & 0xFF, q1 = query[i + stripeSize] & 0xFF;
            int q2 = query[i + stripeSize * 2] & 0xFF, q3 = query[i + stripeSize * 3] & 0xFF;
            sum += Integer.bitCount(q0 & d0) + Integer.bitCount(q0 & d1) * 2L + Integer.bitCount(q1 & d0) * 2L
                + Integer.bitCount(q1 & d1) * 4L + Integer.bitCount(q2 & d0) * 4L + Integer.bitCount(q2 & d1) * 8L
                + Integer.bitCount(q3 & d0) * 8L + Integer.bitCount(q3 & d1) * 16L;
        }
        return sum;
    }

    /** 4-bit doc × 4-bit query at a byte {@code offset} into {@code docs}. */
    static float nibble(byte[] query, byte[] docs, int offset, int len) {
        int stripeSize = len / 4;
        long sum = 0;
        for (int i = 0; i < stripeSize; i++) {
            int d0 = docs[offset + i] & 0xFF, d1 = docs[offset + i + stripeSize] & 0xFF;
            int d2 = docs[offset + i + stripeSize * 2] & 0xFF, d3 = docs[offset + i + stripeSize * 3] & 0xFF;
            int q0 = query[i] & 0xFF, q1 = query[i + stripeSize] & 0xFF;
            int q2 = query[i + stripeSize * 2] & 0xFF, q3 = query[i + stripeSize * 3] & 0xFF;
            sum += Integer.bitCount(q0 & d0) + Integer.bitCount(q0 & d1) * 2L + Integer.bitCount(q0 & d2) * 4L
                + Integer.bitCount(q0 & d3) * 8L + Integer.bitCount(q1 & d0) * 2L + Integer.bitCount(q1 & d1) * 4L
                + Integer.bitCount(q1 & d2) * 8L + Integer.bitCount(q1 & d3) * 16L + Integer.bitCount(q2 & d0) * 4L
                + Integer.bitCount(q2 & d1) * 8L + Integer.bitCount(q2 & d2) * 16L + Integer.bitCount(q2 & d3) * 32L
                + Integer.bitCount(q3 & d0) * 8L + Integer.bitCount(q3 & d1) * 16L + Integer.bitCount(q3 & d2) * 32L
                + Integer.bitCount(q3 & d3) * 64L;
        }
        return sum;
    }
}
