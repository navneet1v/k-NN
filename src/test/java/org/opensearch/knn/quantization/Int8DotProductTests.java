package org.opensearch.knn.quantization;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Fast, isolated unit tests for {@link Int8DotProduct} (JUnit 5).
 *
 * <p>The kernel reads eight bytes at a time and finishes a scalar tail, so the widths exercise both paths. The
 * expectation comes from a plain byte-by-byte reference — a different formulation from the VarHandle loop — so a
 * lane-shift or offset mistake shows up as a mismatch. Doc codes are indexed from {@code offset}, since a block
 * reader calls this with one buffer and a per-vector offset.
 */
class Int8DotProductTests {

    /** 8 exercises only the eight-at-a-time path, 3 only the tail, 9/13 both; 768/1024 are real dims. */
    @ParameterizedTest(name = "dim {0}")
    @ValueSource(ints = { 1, 3, 8, 9, 13, 16, 768, 1024 })
    void unsignedDot_matchesByteByByteReference(final int dim) {
        final Random random = new Random(dim * 31L);
        final int offset = 5;
        final byte[] query = randomBytes(random, dim);
        final byte[] docs = randomBytes(random, offset + dim);

        final long actual = Int8DotProduct.unsignedDot(query, docs, offset, dim);

        assertEquals(reference(query, docs, offset, dim), actual);
    }

    @Test
    void unsignedDot_allMaxBytes_isDimTimes255Squared() {
        final int dim = 100;
        final byte[] query = new byte[dim];
        final byte[] docs = new byte[dim];
        java.util.Arrays.fill(query, (byte) 0xFF);
        java.util.Arrays.fill(docs, (byte) 0xFF);

        assertEquals((long) dim * 255 * 255, Int8DotProduct.unsignedDot(query, docs, 0, dim));
    }

    @Test
    void unsignedDot_allZero_isZero() {
        final int dim = 64;
        assertEquals(0L, Int8DotProduct.unsignedDot(new byte[dim], new byte[dim], 0, dim));
    }

    private static long reference(final byte[] query, final byte[] docs, final int offset, final int dim) {
        long sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += (long) (query[i] & 0xFF) * (docs[offset + i] & 0xFF);
        }
        return sum;
    }

    private static byte[] randomBytes(final Random random, final int n) {
        final byte[] b = new byte[n];
        random.nextBytes(b);
        return b;
    }
}
