/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import org.apache.lucene.index.FloatVectorValues;
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
     * Write side of the same layout: divides one sequence of vectors into fixed-size blocks and appends them
     * to the output, in the same division {@link Reader} walks.
     *
     * <p>What a block holds — full-precision floats, or some encoded form — stays with the implementation,
     * exactly as it does on the read side; this interface fixes only the division and the bounded source it
     * pulls from.
     *
     * <p>Not thread-safe; one instance per sequence being written.
     */
    interface Writer {

        /**
         * The fixed division this writer emits: every block holds {@code blockSize()} vectors except the last,
         * which may be partial. Mirrors {@link Reader#blockSize()} so the two sides never disagree on where a
         * block ends.
         */
        int blockSize();

        /**
         * Write {@code source}'s vectors as a run of fixed-size blocks, in the storage order the source
         * presents them. {@code source} is a view bounded to exactly this sequence — {@code size()} is the
         * sequence length and {@code vectorValue(ord)} is defined only for {@code ord} in {@code [0, size())},
         * in storage order — so the writer reads it by position and needs no separate ordinal list.
         *
         * <p>Blocks are full except the last, which carries {@code source.size() % blockSize()} vectors when
         * the count is not a multiple of the block size.
         *
         * <p>An implementation may read and modify in place the array {@code vectorValue(ord)} returns, so the
         * source must hand out a per-call array that is not shared with any backing store. The bounded views
         * the cluster writer builds satisfy this.
         *
         * @param source the sequence's vectors, as a view bounded to exactly this sequence
         * @throws IOException if writing to the underlying output fails
         */
        void writeBlocks(FloatVectorValues source) throws IOException;
    }
}
