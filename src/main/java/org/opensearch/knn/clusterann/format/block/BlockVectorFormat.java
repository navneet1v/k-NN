/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import org.apache.lucene.util.Accountable;

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
    interface Reader extends Accountable {

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

    /**
     * Write side of the same layout: divides a sequence into blocks.
     *
     * <p>Narrower than {@link Reader} on purpose. A reader positions, skips and re-reads because choosing what
     * <em>not</em> to read is its whole job; a writer appends, so it has no cursor, no {@code advance}, and no
     * notion of a block it might revisit. Blocks fill in order and flush when full.
     *
     * <p>What a block holds stays with the implementation, exactly as it does on the read side — the two agree on
     * the layout by being written and read in one place, not by sharing a type.
     *
     * <p>Not thread-safe; one instance per sequence being written.
     */
    interface Writer {

        /**
         * The fixed division this writer produces, which is what makes the reader's positioning arithmetic rather
         * than a walk. Every block but the last holds exactly this many vectors.
         */
        int blockSize();

        /**
         * Append one vector, flushing the current block once it is full.
         *
         * @param vector the vector to store, in the space the field's codes live in
         * @param reference the point the vector is encoded relative to, for a storage family that encodes residuals.
         *     Families that store vectors outright ignore it.
         */
        void addVector(float[] vector, float[] reference) throws IOException;

        /**
         * Flush the final, partial block. Nothing after this may be added.
         *
         * <p>Must be called: the last block is short by definition, so without this the tail of a sequence is
         * buffered and never written.
         */
        void finish() throws IOException;
    }
}
