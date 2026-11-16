/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * Unsigned-byte dot product for data-blind int8 codes: {@code Σᵢ (query[i]&0xFF)·(docs[offset+i]&0xFF)}
 * over {@code dim} dimensions, both sides one unsigned byte per dimension.
 *
 * <p>This is the symmetric 8-bit analogue of {@link Int4DotProduct#nibble}: whole codes multiply, so it is a
 * plain multiply-accumulate with no popcount. Eight bytes are read at a time through a little-endian
 * {@link VarHandle} and multiplied lane by lane after widening, which keeps the hot loop free of per-byte
 * bounds checks and lets the JIT vectorize the widening multiply; a scalar tail finishes the final
 * {@code dim % 8} bytes.
 *
 * <p>The returned sum is the raw unsigned-code dot {@code Σ aᵢbᵢ}; the signed-residual and scale corrections
 * are applied by the caller (see the data-blind int8 scorer), exactly as {@link Int4DotProduct}'s callers do.
 */
public final class Int8DotProduct {

    private Int8DotProduct() {}

    private static final VarHandle LONG_LE = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /**
     * {@code Σᵢ (query[i]&0xFF)·(docs[offset+i]&0xFF)} for {@code i} in {@code [0, dim)}.
     *
     * @param query one unsigned byte per dimension, length ≥ {@code dim}
     * @param docs  the block's code bytes; this vector's codes start at {@code offset}
     * @param offset byte offset of this vector's codes within {@code docs}
     * @param dim    the vector dimension (one code byte per dimension)
     * @return the unsigned-code dot product as a {@code long} (fits: dim·255·255 ≪ 2^63)
     */
    public static long unsignedDot(final byte[] query, final byte[] docs, final int offset, final int dim) {
        long sum = 0;
        int i = 0;
        // Eight bytes at a time: widen each byte to its unsigned value and multiply lane by lane. The
        // per-lane shift+mask is cheap and keeps the two operands' bytes aligned without a gather.
        for (final int bound = dim & -Long.BYTES; i < bound; i += Long.BYTES) {
            final long q = (long) LONG_LE.get(query, i);
            final long d = (long) LONG_LE.get(docs, offset + i);
            for (int lane = 0; lane < Long.BYTES; lane++) {
                final int shift = lane << 3;
                sum += ((q >>> shift) & 0xFF) * ((d >>> shift) & 0xFF);
            }
        }
        for (; i < dim; i++) {
            sum += (long) (query[i] & 0xFF) * (docs[offset + i] & 0xFF);
        }
        return sum;
    }
}
