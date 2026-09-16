/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * {@link PostingScorer} over a posting stored in blocks, for any block storage family. Pure
 * orchestration: iterate blocks via a {@link BlockReader}, prune, mask off the positions not worth scoring,
 * bulk-score the rest via a {@link BlockScorer}, and hand out per-vector {@link #ord()}/{@link #score()}. It
 * knows nothing about how a block is stored or scored — "block vs row" and the quantization family live
 * entirely inside the {@link BlockReader} / {@link BlockScorer} it's given (and the {@link PostingPruner}s).
 *
 * <p>Work escalates in three phases per block, each gate paid for only once the cheaper one passes:
 * <ol>
 *   <li><b>Free</b> — is any vector in the block wanted? A block's position range is arithmetic and the
 *       {@code ordinals} and {@code wanted} {@link Bits} are already in memory, so a block that is
 *       entirely filtered out or already visited is skipped with <em>zero I/O</em> — the scan never
 *       seeks to it, so not even its per-block metadata is read.</li>
 *   <li><b>Cheap</b> — seek to the block (reading its per-block metadata) and let the pruners inspect
 *       it; a {@link PostingPruner.Decision#SKIP}/{@code TERMINATE} avoids the vectors.</li>
 *   <li><b>Expensive</b> — read the block's vectors and score the wanted positions only.</li>
 * </ol>
 * Because phase 1 is free, the scan can also look ahead at no cost to find the block it will actually
 * visit next and prefetch exactly that one, rather than spending the hint on a block it may skip. It is
 * also why block geometry is computed here rather than asked of the reader: phase 1 has to know a block's
 * position range <em>before</em> it decides whether to seek.
 *
 * <p>Holds the posting's {@code ordinals} ({@code pos → ord}), read sequentially up front by the
 * cluster; the {@code ord → doc} hop is the collector's. Not thread-safe.
 */
final class BlockPostingScorer implements PostingScorer {

    private final BlockScorer scorer;
    private final BlockReader reader; // the scorer's own cursor — we position it, it reads from it
    private final PostingPruner[] pruners; // block-skip strategies, run in order (may be empty)
    private final int[] ordinals;          // pos → global vector ord
    private final Bits wanted;             // ord-space "may I score this?" (filter ∩ not-visited); null = all
    private final int numBlocks;

    // Current block's "score me" mask (block-local), and what the scorer reported back. Both reused.
    private final FixedBitSet valid = new FixedBitSet(BLOCK_SIZE);
    private final BlockCandidates candidates = new BlockCandidates();
    private int i = -1;

    private int blockStart;
    private int cursor;                        // next block index to consider
    private int pending = PENDING_UNKNOWN;     // next wanted block, already found by the lookahead

    private static final int PENDING_UNKNOWN = Integer.MIN_VALUE;

    BlockPostingScorer(
        BlockScorer scorer,
        PostingPruner[] pruners,
        int[] ordinals,
        Bits wanted
    ) {
        this.scorer = scorer;
        this.reader = scorer.reader();
        this.pruners = pruners;
        this.ordinals = ordinals;
        this.wanted = wanted;
        this.numBlocks = (ordinals.length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    @Override
    public boolean advance(float minCompetitiveSimilarity) throws IOException {
        // Still inside the current block — advance to the next candidate.
        if (++i < candidates.size) {
            return true;
        }
        while (true) {
            // Phase 1 (free): first block from the cursor holding a wanted vector. Pure arithmetic over
            // in-memory ordinals — blocks entirely filtered/visited are passed over without any I/O.
            // Usually already known: the previous iteration's prefetch lookahead found it.
            int block = pending != PENDING_UNKNOWN ? pending : nextWantedBlock(cursor);
            pending = PENDING_UNKNOWN;
            if (block < 0) {
                return false;
            }
            cursor = block + 1;

            // Phase 2 (cheap): land on the block, reading only its per-block metadata, and prune.
            reader.seekToBlock(block);
            int start = blockStart(block);
            int len = blockLength(start);

            // The next block we intend to visit is knowable for free — prefetch that one, not merely
            // the adjacent one, so hints aren't spent on blocks the scan will skip. Kept for the next
            // iteration so the lookahead is computed once per block, not twice.
            pending = nextWantedBlock(cursor);
            if (pending >= 0) {
                reader.prefetchBlock(pending);
            }

            boolean skip = false;
            for (PostingPruner pruner : pruners) {
                PostingPruner.Decision d = pruner.inspect(start, len, minCompetitiveSimilarity);
                if (d == PostingPruner.Decision.TERMINATE) {
                    return false;     // rest of the cluster is provably hopeless
                }
                if (d == PostingPruner.Decision.SKIP) {
                    skip = true;      // vectors left unread
                    break;
                }
            }
            if (skip) {
                continue;
            }

            // Phase 3 (expensive): read the block's vectors, then score what the mask allows. The mask is
            // rebuilt here rather than reused from the lookahead because `wanted` is live — the searcher
            // marks ordinals visited between advances, so the block may have emptied since. Building it
            // before the read is what keeps an emptied block from paying for one.
            if (!buildMask(start, len)) {
                continue;
            }
            reader.readBlockVectors();
            scorer.scoreBlock(valid, candidates);
            blockStart = start;
            i = 0;
            return true;
        }
    }

    /** Posting-local position of the first vector in {@code block}. */
    private static int blockStart(int block) {
        return block * BLOCK_SIZE;
    }

    /** Number of vectors in the block starting at {@code start} — short only for the last block. */
    private int blockLength(int start) {
        return Math.min(BLOCK_SIZE, ordinals.length - start);
    }

    /**
     * First block at or after {@code from} containing a vector that is still wanted, or {@code -1} if
     * none remain. Costs no I/O: block ranges are arithmetic and both {@code ordinals} and
     * {@code wanted} are in memory. Only tests — {@link #buildMask} builds the mask for the block actually
     * scored.
     */
    private int nextWantedBlock(int from) {
        for (int b = Math.max(from, 0); b < numBlocks; b++) {
            if (wanted == null) {
                return b;
            }
            int start = blockStart(b);
            int len = blockLength(start);
            for (int j = 0; j < len; j++) {
                if (wanted.get(ordinals[start + j])) {
                    return b;
                }
            }
        }
        return -1;
    }

    /**
     * Rebuild {@link #valid} for the block at {@code [start, start+len)}: set the block-local position of
     * every vector still wanted. Returns {@code false} if none are, in which case the block's vectors are
     * never read. With no filter this is a single word-range write rather than a per-position pass.
     */
    private boolean buildMask(int start, int len) {
        valid.clear();
        if (wanted == null) {
            valid.set(0, len);
            return len > 0;
        }
        boolean any = false;
        for (int j = 0; j < len; j++) {
            if (wanted.get(ordinals[start + j])) {
                valid.set(j);
                any = true;
            }
        }
        return any;
    }

    @Override
    public int ord() {
        return ordinals[blockStart + candidates.positions[i]];
    }

    @Override
    public float score() {
        return candidates.scores[i];
    }
}
