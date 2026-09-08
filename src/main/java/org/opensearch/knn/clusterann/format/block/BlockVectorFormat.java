/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import java.io.IOException;

/**
 * A sequence of vectors split into fixed-size blocks. The block is intended to be a unit of I/O.
 *
 * <p>The format fixes only how a sequence is divided and how a caller walks the divisions. What a block
 * holds is left to the implementation
 */
public interface BlockVectorFormat {

    /**
     * Generic cursor over a set of blocks put in contiguous memory location. It hides the storage-specific arithmetic
     * and access from the reader's consumer.
     *
     * <p>The block layout, metadata it carries and the way to read it stays with the reader's concrete implementation
     *
     * <p>Not thread-safe; one instance per sequence being scanned.
     */
    interface Reader {

        /** Number of blocks in the sequence. Known without positioning on any of them. */
        int numBlocks();

        /**
         * The fixed division: block {@code b} covers positions {@code [b·blockSize, (b+1)·blockSize)}, except
         * the last, which may hold fewer. Known without positioning on any block.
         */
        int blockSize();

        /**
         * Position on {@code blockPos}. Returns {@code false} when {@code blockPos} is out of range.
         *
         * <p>Positioning only — nothing is read, so a caller may walk to a block and walk away from it for
         * free. Everything that costs comes later, in {@link #fetchBlock}.
         */
        boolean advance(int blockPos) throws IOException;

        /**
         * Number of vectors in the current block — the block size for every block but the last, which may be
         * partial.
         */
        int blockVectorCount();

        /**
         * Make the current block's contents available, paying whatever IO that takes.
         * Nothing before this reads.
         *
         * <p>Cheap when {@link #prefetchBlock} already warmed the block: prefetch starts the same work
         * asynchronously, and this waits for it.
         *
         * <p>Call at most once per block, after {@link #advance}.
         */
        void fetchBlock() throws IOException;

        /**
         * Decode the current block's vector payload into this reader, ready for scoring.
         *
         * <p>Returns nothing deliberately: what the payload looks like — packed codes, full precision vectors
         * — stays inside the reader, while the cheap/expensive boundary stays where the caller can control it.
         *
         * <p>Call at most once per block, after {@link #fetchBlock}. Valid until the next {@link #advance}.
         */
        void readBlockVectors() throws IOException;

        /**
         * Advisory hint that {@code block} will be fetched soon, issued while the current block is still being
         * scored so the fetch overlaps useful work — the asynchronous form of {@link #fetchBlock}. Optional; a
         * reader with no prefetch story ignores it.
         *
         * <p>Hint the block you actually intend to visit next, not merely the adjacent one, so hints aren't
         * spent on blocks that end up skipped.
         */
        default void prefetchBlock(int block) throws IOException {}
    }

    /** Write side of the same layout: divides a sequence into blocks. */
    interface Writer {
        // TODO: To be added when writer is being created
    }
}
