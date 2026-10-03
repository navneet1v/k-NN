/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the byte arithmetic {@link ScalarBlockLayout} owns. This is the only place a block byte budget is
 * turned into a vector count, on either side of the format, so these numbers are the on-disk contract: a block
 * written at a given {@code ioFetchBytes} is read back at the same count only because both sides come here.
 */
class ScalarBlockLayoutTests {

    private static final int IO_FETCH_BYTES = 32 * 1024;

    /** Corrective terms are four ints per vector whatever the encoding. */
    private static final int CORRECTIVE_BYTES = 16;

    @ParameterizedTest(name = "{0}-bit codes at dim {1} cost {2} bytes per vector")
    @CsvSource({ "1, 768, 112", "2, 768, 208", "4, 768, 400" })
    void testBytesPerVector_thenCorrectiveTermsPlusPackedCodes(final int docBits, final int dimension, final int expected) {
        final ScalarEncoding encoding = ScalarEncoding.fromNumBits(docBits);
        assertEquals(expected, ScalarBlockLayout.bytesPerVector(encoding, dimension));
        assertEquals(
            expected - CORRECTIVE_BYTES,
            ScalarBlockLayout.packedLength(encoding, dimension),
            "the balance of a vector's cost is its packed codes"
        );
    }

    @ParameterizedTest(name = "{0}-bit codes fit {1} vectors in 32 KB")
    @CsvSource({ "1, 292", "2, 157", "4, 81" })
    void testBlockSize_thenFillsTheByteBudget(final int docBits, final int expected) {
        assertEquals(expected, ScalarBlockLayout.blockSize(IO_FETCH_BYTES, ScalarEncoding.fromNumBits(docBits), 768));
    }

    @Test
    void testBlockSize_thenCountsOnlyWholeVectors() {
        final int bytesPerVector = 112; // dim 768 at 1 bit
        assertEquals(
            3,
            ScalarBlockLayout.blockSize(4 * bytesPerVector - 1, ScalarEncoding.fromNumBits(1), 768),
            "a budget one byte short of four vectors holds three"
        );
        assertEquals(4, ScalarBlockLayout.blockSize(4 * bytesPerVector, ScalarEncoding.fromNumBits(1), 768));
    }

    /**
     * A vector wider than the whole budget still gets a block of its own: a block of zero vectors would make the
     * sequence unwritable, so the floor is a correctness rule rather than a rounding choice.
     */
    @ParameterizedTest(name = "budget {0} bytes")
    @ValueSource(ints = { 0, 1, 64 })
    void testBlockSize_whenAVectorExceedsTheBudget_thenStillOne(final int ioFetchBytes) {
        assertEquals(1, ScalarBlockLayout.blockSize(ioFetchBytes, ScalarEncoding.fromNumBits(2), 4096));
    }

    /** A block never outgrows the budget it was sized for, which is the one promise the reader's fetch rests on. */
    @ParameterizedTest(name = "{0}-bit codes at dim {1}")
    @CsvSource({ "1, 768", "2, 768", "4, 768", "1, 128", "2, 128", "4, 128", "2, 960", "4, 960" })
    void testBlockSize_thenABlockNeverOutgrowsItsBudget(final int docBits, final int dimension) {
        final ScalarEncoding encoding = ScalarEncoding.fromNumBits(docBits);
        final int bytesPerVector = ScalarBlockLayout.bytesPerVector(encoding, dimension);
        final int blockSize = ScalarBlockLayout.blockSize(IO_FETCH_BYTES, encoding, dimension);
        final int blockBytes = blockSize * bytesPerVector;

        assertTrue(
            blockBytes <= IO_FETCH_BYTES,
            "block of " + blockSize + " vectors is " + blockBytes + " bytes, over the " + IO_FETCH_BYTES + " budget"
        );
        assertTrue(
            blockBytes + bytesPerVector > IO_FETCH_BYTES,
            "another whole vector would have fitted, so the budget is not being filled"
        );
    }

    @Test
    void testBytesPerVector_whenTheEncodingCannotBePacked_thenThrows() {
        assertThrows(UnsupportedOperationException.class, () -> ScalarBlockLayout.bytesPerVector(ScalarEncoding.UNSIGNED_BYTE, 768));
    }
}
