/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link ScalarQuantizedBlockReader} (JUnit 5).
 *
 * <p>Bytes come from {@link #write}, an independent encoder of the same layout, so the reader is driven over a
 * real {@link IndexInput} rather than a mock — the arithmetic it does (block stride, partial last block, code
 * offsets) is the whole of its behaviour and a mock would hide it.
 *
 * <p>The layout is unpadded: a block holds only the vectors it has, so the final block is shorter than the
 * rest. Block starts are still a constant stride, because every block <em>before</em> the last is full.
 */
class SQScanContextBlockReaderTests {

    private static final int DIMENSION = 8;
    private static final ScalarEncoding ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
    private static final int PACKED_BYTES = ENCODING.getDocPackedLength(DIMENSION);

    /** 10 vectors over blocks of 4: two full blocks and a partial third, so the short tail is always in play. */
    private static final int BLOCK_SIZE = 4;
    private static final int VECTOR_COUNT = 10;

    private static final String FILE = "quantized";

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- geometry

    @ParameterizedTest(name = "{0} vectors in blocks of {1}")
    @CsvSource({ "10, 4, 3", "8, 4, 2", "1, 4, 1", "4, 4, 1", "5, 4, 2", "0, 4, 0" })
    void testNumBlocks_thenCoversEveryVector(int vectorCount, int blockSize, int expectedBlocks) throws IOException {
        // given / when
        ScalarQuantizedBlockReader reader = readerOver(vectorCount, blockSize);

        // then
        assertEquals(expectedBlocks, reader.numBlocks());
        assertEquals(blockSize, reader.blockSize());
    }

    /** Every block is full but the last, which holds only what is left over. */
    @ParameterizedTest(name = "block {0}")
    @CsvSource({ "0, 4", "1, 4", "2, 2" })
    void testBlockVectorCount_thenTheLastBlockIsShort(int block, int expectedCount) throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);

        // when
        assertTrue(reader.advance(block));

        // then
        assertEquals(expectedCount, reader.blockVectorCount());
    }

    // ---------------------------------------------------------------- advance

    @ParameterizedTest(name = "block {0}")
    @ValueSource(ints = { 0, 1, 2 })
    void testAdvance_whenBlockExists_thenPositionsOnIt(int block) throws IOException {
        // given / when / then
        assertTrue(readerOver(VECTOR_COUNT, BLOCK_SIZE).advance(block));
    }

    /**
     * A walk runs off the end by design — {@code while (advance(i))} — so one past the last block must answer
     * false rather than throw.
     */
    @ParameterizedTest(name = "block {0}")
    @ValueSource(ints = { 3, 4, 99 })
    void testAdvance_whenBlockIsPastTheEnd_thenReturnsFalse(int block) throws IOException {
        // given / when / then
        assertFalse(readerOver(VECTOR_COUNT, BLOCK_SIZE).advance(block));
    }

    /**
     * Positioning carries no state beyond the cursor, so a block behind the current one is reachable and reads
     * correctly — the interface promises a caller may walk to a block and walk away from it freely.
     */
    @Test
    void testAdvance_whenGoingBackwards_thenStillLandsOnThatBlock() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);
        assertTrue(reader.advance(2));

        // when
        assertTrue(reader.advance(0));
        reader.fetchBlock();
        reader.readBlockVectors();

        // then
        assertBlock(reader, 0, BLOCK_SIZE);
    }

    /** Re-reading the block already under the cursor is a fresh read of the same bytes, not a step forward. */
    @Test
    void testAdvance_whenRepeatingTheCurrentBlock_thenReadsItAgain() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);
        assertTrue(reader.advance(1));
        reader.fetchBlock();
        reader.readBlockVectors();

        // when
        assertTrue(reader.advance(1));
        reader.fetchBlock();
        reader.readBlockVectors();

        // then
        assertBlock(reader, 1, BLOCK_SIZE);
    }

    @Test
    void testAdvance_whenThereAreNoVectors_thenReturnsFalse() throws IOException {
        // given / when / then
        assertFalse(readerOver(0, BLOCK_SIZE).advance(0));
    }

    // ---------------------------------------------------------------- reading

    /**
     * The point of the unpadded layout: a full block and the short last block both read their own corrections
     * and their own codes, and neither strays into the other's bytes.
     */
    @ParameterizedTest(name = "block {0}")
    @ValueSource(ints = { 0, 1, 2 })
    void testFetchAndRead_thenEveryBlockYieldsItsOwnValues(int block) throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);

        // when
        assertTrue(reader.advance(block));
        int count = reader.blockVectorCount();
        reader.fetchBlock();
        reader.readBlockVectors();

        // then
        assertBlock(reader, block, count);
    }

    /** Walking the whole sequence forward: the constant stride has to land on each block in turn. */
    @Test
    void testFetchAndRead_whenWalkingEveryBlock_thenEachLandsOnItsOwnData() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);

        // when / then
        int block = 0;
        while (reader.advance(block)) {
            int count = reader.blockVectorCount();
            reader.fetchBlock();
            reader.readBlockVectors();
            assertBlock(reader, block, count);
            block++;
        }
        assertEquals(3, block, "the walk must visit every block and then stop");
    }

    /** Blocks are seekable, not just sequential: the last one reads correctly without touching the earlier two. */
    @Test
    void testFetchAndRead_whenSkippingStraightToTheLastBlock_thenStillLandsOnIt() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);

        // when
        assertTrue(reader.advance(2));
        reader.fetchBlock();
        reader.readBlockVectors();

        // then
        assertBlock(reader, 2, 2);
    }

    // ---------------------------------------------------------------- misuse

    @Test
    void testBlockVectorCount_whenNothingIsPositioned_thenThrows() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);

        // when / then
        assertThrows(IllegalStateException.class, reader::blockVectorCount);
    }

    @Test
    void testFetchBlock_whenNothingIsPositioned_thenThrows() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);

        // when / then
        assertThrows(IllegalStateException.class, reader::fetchBlock);
    }

    @Test
    void testBlockVectorCount_whenTheWalkHasRunOff_thenThrows() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = readerOver(VECTOR_COUNT, BLOCK_SIZE);
        assertFalse(reader.advance(3));

        // when / then
        assertThrows(IllegalStateException.class, reader::blockVectorCount);
    }

    // ---------------------------------------------------------------- prefetch

    /** A hint has to name the block's real start, corrections included, or it warms the wrong bytes. */
    @Test
    void testPrefetchBlock_thenHintsTheWholeBlockFromItsStart() throws IOException {
        // given
        PrefetchRecordingInput input = new PrefetchRecordingInput(open(VECTOR_COUNT, BLOCK_SIZE));
        ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(input, BLOCK_SIZE, VECTOR_COUNT, DIMENSION, ENCODING);
        long blockBytes = fixedBlockBytes(BLOCK_SIZE);

        // when
        assertTrue(reader.advance(0));
        reader.prefetchBlock(1);

        // then
        assertEquals(1, input.prefetches.size());
        assertArrayEquals(new long[] { blockBytes, blockBytes }, input.prefetches.get(0));
    }

    /** The last block is shorter than the stride, so the hint must stop at the end of the file. */
    @Test
    void testPrefetchBlock_whenTheBlockIsTheShortLastOne_thenClampsToTheFileEnd() throws IOException {
        // given
        PrefetchRecordingInput input = new PrefetchRecordingInput(open(VECTOR_COUNT, BLOCK_SIZE));
        ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(input, BLOCK_SIZE, VECTOR_COUNT, DIMENSION, ENCODING);
        long blockBytes = fixedBlockBytes(BLOCK_SIZE);
        long offset = 2 * blockBytes;

        // when
        assertTrue(reader.advance(0));
        reader.prefetchBlock(2);

        // then
        assertEquals(1, input.prefetches.size());
        assertArrayEquals(new long[] { offset, input.length() - offset }, input.prefetches.get(0));
        assertTrue(input.prefetches.get(0)[1] < blockBytes, "the short last block must ask for less than a full stride");
    }

    @Test
    void testPrefetchBlock_whenTheBlockIsPastTheEnd_thenHintsNothing() throws IOException {
        // given
        PrefetchRecordingInput input = new PrefetchRecordingInput(open(VECTOR_COUNT, BLOCK_SIZE));
        ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(input, BLOCK_SIZE, VECTOR_COUNT, DIMENSION, ENCODING);

        // when
        assertTrue(reader.advance(0));
        reader.prefetchBlock(99);

        // then
        assertTrue(input.prefetches.isEmpty(), "there is nothing after the last block to warm");
    }

    // ---------------------------------------------------------------- accounting

    /**
     * The buffers are the reader's whole footprint, and the codes buffer is the bulk of it — so the estimate has
     * to grow with the block size rather than reporting a fixed overhead.
     */
    @Test
    void testRamBytesUsed_thenCountsTheBuffersItHolds() throws IOException {
        // given
        ScalarQuantizedBlockReader small = readerOver(VECTOR_COUNT, 4);
        ScalarQuantizedBlockReader large = readerOver(VECTOR_COUNT, 64);

        // when / then — 64 vectors' worth of codes and corrections cannot cost the same as 4
        assertTrue(large.ramBytesUsed() > small.ramBytesUsed(), "a larger block holds larger buffers");

        long codesAndCorrections = 4L * PACKED_BYTES  // codes
            + 4L * Float.BYTES * 3                    // lower, upper, add
            + 4L * Integer.BYTES * 2;                 // sum, intScratch
        assertTrue(small.ramBytesUsed() >= codesAndCorrections, "every buffer must be counted, codes included");
    }

    // ---------------------------------------------------------------- helpers

    /** Corrections and codes are derived from the global vector ordinal, so any mis-seek shows up as a value. */
    private static float expectedLower(int vector) {
        return vector + 0.5f;
    }

    private static float expectedUpper(int vector) {
        return vector + 100.5f;
    }

    private static float expectedAddCor(int vector) {
        return vector + 200.5f;
    }

    private static int expectedSum(int vector) {
        return vector * 3;
    }

    private static byte expectedCode(int vector, int byteIndex) {
        return (byte) (vector * 10 + byteIndex);
    }

    /** What the reader must hold after fetching and reading {@code block}, which starts at {@code block·blockSize}. */
    private static void assertBlock(ScalarQuantizedBlockReader reader, int block, int count) {
        int base = block * BLOCK_SIZE;
        for (int i = 0; i < count; i++) {
            int vector = base + i;
            assertEquals(expectedLower(vector), reader.lower()[i], "lower of vector " + vector);
            assertEquals(expectedUpper(vector), reader.upper()[i], "upper of vector " + vector);
            assertEquals(expectedAddCor(vector), reader.addCor()[i], "add of vector " + vector);
            assertEquals(expectedSum(vector), reader.sum()[i], "sum of vector " + vector);
            for (int b = 0; b < PACKED_BYTES; b++) {
                assertEquals(expectedCode(vector, b), reader.codes()[i * PACKED_BYTES + b], "code byte " + b + " of vector " + vector);
            }
        }
    }

    private static long fixedBlockBytes(int blockSize) {
        return (long) blockSize * PACKED_BYTES + (long) blockSize * Float.BYTES * 3 + (long) blockSize * Integer.BYTES;
    }

    private ScalarQuantizedBlockReader readerOver(int vectorCount, int blockSize) throws IOException {
        return new ScalarQuantizedBlockReader(open(vectorCount, blockSize), blockSize, vectorCount, DIMENSION, ENCODING);
    }

    private IndexInput open(int vectorCount, int blockSize) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        write(directory, vectorCount, blockSize);
        return directory.openInput(FILE, IOContext.DEFAULT);
    }

    /**
     * Writes the layout the reader reads: per block, the four correction arrays for the vectors that block
     * actually holds, then their packed codes. Nothing is padded, so the last block is shorter.
     */
    private static void write(Directory directory, int vectorCount, int blockSize) throws IOException {
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int first = 0; first < vectorCount; first += blockSize) {
                int count = Math.min(blockSize, vectorCount - first);
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(expectedLower(first + i)));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(expectedUpper(first + i)));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(expectedAddCor(first + i)));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(expectedSum(first + i));
                }
                for (int i = 0; i < count; i++) {
                    for (int b = 0; b < PACKED_BYTES; b++) {
                        out.writeByte(expectedCode(first + i, b));
                    }
                }
            }
        }
    }

    /** Records the hints the reader issues, so a test can assert on the region rather than just the call. */
    private static final class PrefetchRecordingInput extends FilterIndexInput {

        private final List<long[]> prefetches = new ArrayList<>();

        private PrefetchRecordingInput(IndexInput in) {
            super("prefetch-recording(" + in + ")", in);
        }

        @Override
        public void prefetch(long offset, long length) throws IOException {
            prefetches.add(new long[] { offset, length });
            in.prefetch(offset, length);
        }
    }
}
