/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import java.io.IOException;

/**
 * Generic block cursor over one posting's block storage — the "block vs row" boundary. It positions
 * itself on a block and exposes that block's position range; <em>how</em> a block is laid out on disk
 * (and what per-block metadata it carries) is the concrete reader's business, shared with its matching
 * {@link BlockScorer} and pruners rather than exposed here. {@link BlockPostingScorer} drives it
 * and needs nothing storage-specific — which is what lets one iterator serve any block storage family.
 *
 * <p><b>Random access is the point.</b> {@link #numBlocks()} + {@link #seekToBlock(int)} let the caller
 * decide which blocks to visit and jump straight to them, so a block that is provably hopeless (pruning)
 * or entirely unwanted (filter / SOAR dedup) costs <em>no I/O at all</em> — not even its per-block
 * metadata. Seeks are expected to move forward only, keeping file offsets monotonic and the access
 * pattern sequential-friendly.
 */
public interface BlockReader {

    /** Number of blocks in this posting. Block {@code b} covers positions {@code [b·BS, min((b+1)·BS, count))}. */
    int numBlocks();

    /**
     * Position on {@code block} and read whatever cheap per-block metadata the pruners inspect (and the
     * scorer reuses); returns {@code false} when {@code block} is out of range. Blocks not seeked to are
     * never read. Callers should advance monotonically. A storage family without random access may
     * implement this by skipping forward sequentially.
     */
    boolean seekToBlock(int block) throws IOException;

    /** Convenience: position on the block after the current one. Equivalent to seeking to {@code cur + 1}. */
    boolean nextBlock() throws IOException;

    /** Posting-local index of the first vector in the current block. */
    int blockStart();

    /** Number of vectors in the current block. */
    int blockLength();

    /**
     * Advisory hint that {@code block} will be read soon, issued while the current block is being
     * scored. Optional — a reader with no prefetch story ignores it. Callers hint the block they
     * actually intend to visit next, not merely the adjacent one, so hints aren't spent on skipped
     * blocks.
     */
    default void prefetchBlock(int block) throws IOException {}
}