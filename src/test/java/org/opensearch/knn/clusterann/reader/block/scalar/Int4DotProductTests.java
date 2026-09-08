/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Fast, isolated unit tests for {@link Int4DotProduct} (JUnit 5).
 *
 * <p>Both kernels compute a weighted sum of population counts over bit planes, and both are written for speed —
 * eight bytes at a time where they can, with the tail handled separately. The expectations here come from a plain
 * plane-by-plane, byte-by-byte reference below, which is a different formulation rather than the same one twice,
 * so a mistake in the packing or in the plane weights shows up as a mismatch.
 *
 * <p>Query codes are always four planes of {@code len} bytes, which is the transposed layout a 4-bit query is
 * written into. Doc codes are indexed from {@code offset}, since a block reader calls these with one buffer of
 * codes and a per-vector offset into it.
 */
class Int4DotProductTests {

    /** 8 exercises only the eight-at-a-time path, 3 only the tail, 9 and 12 both. */
    @ParameterizedTest(name = "len {0}")
    @ValueSource(ints = { 1, 3, 8, 9, 12, 16 })
    void testBit_thenWeightsEachQueryPlaneByItsPlaceValue(int len) {
        // given
        Random random = new Random(len * 31L);
        byte[] query = randomBytes(random, len * 4);
        byte[] docs = randomBytes(random, len);

        // when
        float actual = Int4DotProduct.bit(query, docs, 0, len);

        // then
        assertEquals(expectedBit(query, docs, 0, len), actual);
    }

    /** Only the docs are offset: the query planes are always read from the start of their own buffer. */
    @ParameterizedTest(name = "offset {0}")
    @ValueSource(ints = { 0, 8, 24 })
    void testBit_whenDocsAreOffset_thenReadsThatVectorsCodes(int offset) {
        // given — three vectors of 8 bytes each in one buffer, as a block of codes is laid out
        int len = 8;
        Random random = new Random(7);
        byte[] query = randomBytes(random, len * 4);
        byte[] docs = randomBytes(random, len * 4);

        // when
        float actual = Int4DotProduct.bit(query, docs, offset, len);

        // then
        assertEquals(expectedBit(query, docs, offset, len), actual);
    }

    @Test
    void testBit_whenDocCodesAreZero_thenContributesNothing() {
        // given
        int len = 8;
        byte[] query = randomBytes(new Random(11), len * 4);

        // when / then — the popcount of anything ANDed with zero is zero, whatever the query is
        assertEquals(0f, Int4DotProduct.bit(query, new byte[len], 0, len));
    }

    /**
     * Every bit set on both sides: each of the eight doc bits matches in all four planes, so the total is
     * 8 × (1 + 2 + 4 + 8) = 120 per byte.
     */
    @ParameterizedTest(name = "len {0}")
    @ValueSource(ints = { 1, 8, 9 })
    void testBit_whenEveryBitIsSet_thenReachesTheMaximum(int len) {
        // given
        byte[] query = filled(len * 4, (byte) 0xFF);
        byte[] docs = filled(len, (byte) 0xFF);

        // when / then
        assertEquals(120f * len, Int4DotProduct.bit(query, docs, 0, len));
    }

    // ---------------------------------------------------------------- dibit

    /** Two-bit doc codes: two stripes, so the length is halved into a low plane and a high plane. */
    @ParameterizedTest(name = "len {0}")
    @ValueSource(ints = { 2, 4, 8, 16 })
    void testDibit_thenWeightsEveryQueryAndDocPlanePair(int len) {
        // given
        Random random = new Random(len * 17L);
        int stripe = len / 2;
        byte[] query = randomBytes(random, stripe * 4);
        byte[] docs = randomBytes(random, len);

        // when
        float actual = Int4DotProduct.dibit(query, docs, 0, len);

        // then
        assertEquals(expectedDibit(query, docs, 0, len), actual);
    }

    @ParameterizedTest(name = "offset {0}")
    @ValueSource(ints = { 0, 8, 16 })
    void testDibit_whenDocsAreOffset_thenReadsThatVectorsCodes(int offset) {
        // given
        int len = 8;
        Random random = new Random(23);
        byte[] query = randomBytes(random, (len / 2) * 4);
        byte[] docs = randomBytes(random, len * 3);

        // when
        float actual = Int4DotProduct.dibit(query, docs, offset, len);

        // then
        assertEquals(expectedDibit(query, docs, offset, len), actual);
    }

    @Test
    void testDibit_whenDocCodesAreZero_thenContributesNothing() {
        // given
        int len = 8;
        byte[] query = randomBytes(new Random(29), (len / 2) * 4);

        // when / then
        assertEquals(0f, Int4DotProduct.dibit(query, new byte[len], 0, len));
    }

    /**
     * Every bit set on both sides: each query plane pairs with both doc planes, so per stripe byte the total is
     * 8 × Σ 2^(i+j) over i in 0..3 and j in 0..1 = 8 × 45 = 360.
     */
    @ParameterizedTest(name = "len {0}")
    @ValueSource(ints = { 2, 8 })
    void testDibit_whenEveryBitIsSet_thenReachesTheMaximum(int len) {
        // given
        int stripe = len / 2;
        byte[] query = filled(stripe * 4, (byte) 0xFF);
        byte[] docs = filled(len, (byte) 0xFF);

        // when / then
        assertEquals(360f * stripe, Int4DotProduct.dibit(query, docs, 0, len));
    }

    // ---------------------------------------------------------------- reference implementations

    /**
     * 1-bit docs against a 4-plane query: plane {@code p} carries place value 2^p, and every doc bit that matches
     * in that plane contributes it. Written plane by plane and byte by byte, which is the definition rather than
     * the optimisation.
     */
    private static float expectedBit(byte[] query, byte[] docs, int offset, int len) {
        long total = 0;
        for (int plane = 0; plane < 4; plane++) {
            long matches = 0;
            for (int b = 0; b < len; b++) {
                matches += Integer.bitCount((query[plane * len + b] & docs[offset + b]) & 0xFF);
            }
            total += matches << plane;
        }
        return total;
    }

    /**
     * 2-bit docs: the doc's two planes carry place values 1 and 2, the query's four carry 1, 2, 4 and 8, and a
     * match between plane {@code i} and plane {@code j} is worth the product — 2^(i+j).
     */
    private static float expectedDibit(byte[] query, byte[] docs, int offset, int len) {
        int stripe = len / 2;
        long total = 0;
        for (int queryPlane = 0; queryPlane < 4; queryPlane++) {
            for (int docPlane = 0; docPlane < 2; docPlane++) {
                long matches = 0;
                for (int b = 0; b < stripe; b++) {
                    int q = query[queryPlane * stripe + b] & 0xFF;
                    int d = docs[offset + docPlane * stripe + b] & 0xFF;
                    matches += Integer.bitCount(q & d);
                }
                total += matches << (queryPlane + docPlane);
            }
        }
        return total;
    }

    private static byte[] randomBytes(Random random, int length) {
        byte[] bytes = new byte[length];
        random.nextBytes(bytes);
        return bytes;
    }

    private static byte[] filled(int length, byte value) {
        byte[] bytes = new byte[length];
        java.util.Arrays.fill(bytes, value);
        return bytes;
    }
}
