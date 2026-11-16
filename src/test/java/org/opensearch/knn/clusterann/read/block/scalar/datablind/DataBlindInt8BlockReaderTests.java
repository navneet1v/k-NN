/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the cursor half of {@link DataBlindInt8BlockReader}: the buffers are sized once at construction and
 * {@link DataBlindInt8BlockReader#reset} points them at one cluster after another, so these cover what the reader
 * does on its own — before a reset, across resets, and off the ends of a cluster.
 */
class DataBlindInt8BlockReaderTests {

    private static final int BLOCK_SIZE = 4;
    private static final int DIMENSION = 8;
    private static final int VECTOR_COUNT = 10;
    private static final String FILE = "blocks";

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (final Directory directory : directories) {
            directory.close();
        }
    }

    @Test
    void testReset_whenBlocksAreNull_thenThrows() {
        final DataBlindInt8BlockReader reader = new DataBlindInt8BlockReader(BLOCK_SIZE, DIMENSION);

        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> reader.reset(null, VECTOR_COUNT));
        assertTrue(e.getMessage().contains("blocks"), e.getMessage());
    }

    @Test
    void testReset_whenVectorCountIsNegative_thenThrows() throws IOException {
        final DataBlindInt8BlockReader reader = new DataBlindInt8BlockReader(BLOCK_SIZE, DIMENSION);
        final IndexInput blocks = blocks();

        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> reader.reset(blocks, -1));
        assertTrue(e.getMessage().contains("negative"), e.getMessage());
    }

    @Test
    void testPrefetchBlock_whenNotResetOntoACluster_thenThrows() {
        final DataBlindInt8BlockReader reader = new DataBlindInt8BlockReader(BLOCK_SIZE, DIMENSION);

        assertThrows(IllegalStateException.class, () -> reader.prefetchBlock(0));
    }

    @Test
    void testPrefetchBlock_whenTheBlockIsOutOfRange_thenIsANoOp() throws IOException {
        final DataBlindInt8BlockReader reader = reader();

        // Out of range in both directions: a hint is advisory, so neither is an error.
        reader.prefetchBlock(-1);
        reader.prefetchBlock(reader.numBlocks() + 5);

        assertTrue(reader.advance(0), "a rejected hint must leave the cursor usable");
    }

    @Test
    void testBlockVectorCount_whenTheCursorHasNotAdvanced_thenThrows() throws IOException {
        final DataBlindInt8BlockReader reader = reader();

        final IllegalStateException e = assertThrows(IllegalStateException.class, reader::blockVectorCount);
        assertTrue(e.getMessage().contains("blockCursor"), e.getMessage());
    }

    @Test
    void testAdvance_whenPastTheLastBlock_thenReportsExhausted() throws IOException {
        final DataBlindInt8BlockReader reader = reader();

        assertFalse(reader.advance(reader.numBlocks()), "there is no block at numBlocks()");
    }

    @Test
    void testReset_whenPointedAtASmallerCluster_thenGeometryAndCursorFollowIt() throws IOException {
        final DataBlindInt8BlockReader reader = reader();
        assertTrue(reader.advance(2), "the 10-entry cluster has a third block");

        // A 3-entry cluster is one partial block, so the block count shrinks and the cursor rewinds with it.
        reader.reset(blocks(), 3);

        assertEquals(1, reader.numBlocks());
        assertTrue(reader.advance(0));
        assertEquals(3, reader.blockVectorCount());
        assertFalse(reader.advance(1), "the smaller cluster has no second block");
    }

    @Test
    void testRamBytesUsed_thenCountsTheBuffersItHolds() {
        final DataBlindInt8BlockReader small = new DataBlindInt8BlockReader(BLOCK_SIZE, DIMENSION);
        final DataBlindInt8BlockReader large = new DataBlindInt8BlockReader(BLOCK_SIZE * 8, DIMENSION);

        assertTrue(large.ramBytesUsed() > small.ramBytesUsed(), "a larger block holds larger buffers");
        final long codesAndCorrections = (long) BLOCK_SIZE * DIMENSION   // codes
            + (long) BLOCK_SIZE * Integer.BYTES * 4;                     // scale, sum, sqNorm, intScratch
        assertTrue(small.ramBytesUsed() >= codesAndCorrections, "every buffer must be counted, codes included");
    }

    private DataBlindInt8BlockReader reader() throws IOException {
        final DataBlindInt8BlockReader reader = new DataBlindInt8BlockReader(BLOCK_SIZE, DIMENSION);
        reader.reset(blocks(), VECTOR_COUNT);
        return reader;
    }

    /** Bytes of the right shape for {@link #VECTOR_COUNT} vectors; values are never read back by these tests. */
    private IndexInput blocks() throws IOException {
        final Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        final int blocks = (VECTOR_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int i = 0; i < blocks * BLOCK_SIZE * (3 * Integer.BYTES + DIMENSION); i++) {
                out.writeByte((byte) i);
            }
        }
        return directory.openInput(FILE, IOContext.DEFAULT);
    }
}
