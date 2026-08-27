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
 * orchestration: iterate blocks via a {@link BlockReader}, prune, mask out unwanted vectors, bulk-score
 * via a {@link BlockScorer}, and hand out per-vector {@link #ord()}/{@link #score()}. It knows nothing
 * about how a block is stored or scored — "block vs row" and the quantization family live entirely
 * inside the {@link BlockReader} / {@link BlockScorer} it's given (and the {@link PostingPruner}s).
 *
 * <p>Work escalates in three phases per block, each gate paid for only once the cheaper one passes:
 * <ol>
 *   <li><b>Free</b> — is any vector in the block wanted? A block's position range is arithmetic and the
 *       {@code ordinals} and {@code wanted} {@link Bits} are already in memory, so a block that is
 *       entirely filtered out or already visited is skipped with <em>zero I/O</em> — the scan never
 *       seeks to it, so not even its per-block metadata is read.</li>
 *   <li><b>Cheap</b> — seek to the block (reading its per-block metadata) and let the pruners inspect
 *       it; a {@link PostingPruner.Decision#SKIP}/{@code TERMINATE} avoids the codes.</li>
 *   <li><b>Expensive</b> — read and score the codes, for the wanted positions only.</li>
 * </ol>
 * Because phase 1 is free, the scan can also look ahead at no cost to find the block it will actually
 * visit next and prefetch exactly that one, rather than spending the hint on a block it may skip.
 *
 * <p>Holds the posting's {@code ordinals} ({@code pos → ord}), read sequentially up front by the
 * cluster; the {@code ord → doc} hop is the collector's. Not thread-safe.
 */
final class BlockPostingScorer implements PostingScorer {

    private final BlockReader reader;
    private final BlockScorer scorer;
    private final PostingPruner[] pruners; // block-skip strategies, run in order (may be empty)
    private final int[] ordinals;          // pos → global vector ord
    private final Bits wanted;             // ord-space "may I score this?" (filter ∩ not-visited); null = all
    private final float[] scores;          // current block's per-vector scores
    private final FixedBitSet valid;       // current block's per-position "score me" mask (by block position)

    private int blockStart;
    private int blockLen;
    private int posInBlock = -1;
    private int cursor;                        // next block index to consider
    private int pending = PENDING_UNKNOWN;     // next wanted block, already found by the lookahead

    private static final int PENDING_UNKNOWN = Integer.MIN_VALUE;

    BlockPostingScorer(
        BlockReader reader,
        BlockScorer scorer,
        PostingPruner[] pruners,
        int[] ordinals,
        Bits wanted
    ) {
        this.reader = reader;
        this.scorer = scorer;
        this.pruners = pruners;
        this.ordinals = ordinals;
        this.wanted = wanted;
        this.scores = new float[BLOCK_SIZE];
        this.valid = new FixedBitSet(BLOCK_SIZE);
    }

    @Override
    public boolean advance(float minCompetitiveSimilarity) throws IOException {
        // Still inside the current block — advance to the next wanted position.
        if (advanceInBlock()) {
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
            if (!reader.seekToBlock(block)) {
                return false;
            }
            int start = reader.blockStart();
            int len = reader.blockLength();

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
                    skip = true;      // codes left unread
                    break;
                }
            }
            if (skip) {
                continue;
            }

            // Phase 3 (expensive): score the wanted positions; the scorer reads the codes itself.
            // The mask is rebuilt here rather than reused from the lookahead because `wanted` is live —
            // the searcher marks ordinals visited between advances, so the block may have emptied since.
            if (!buildMask(start, len)) {
                continue;
            }
            scorer.scoreBlock(len, valid, scores);
            blockStart = start;
            blockLen = len;
            posInBlock = -1;
            if (advanceInBlock()) {
                return true;          // guaranteed: the mask is non-empty by construction
            }
        }
    }

    /**
     * First block at or after {@code from} containing a vector that is still wanted, or {@code -1} if
     * none remain. Costs no I/O: block ranges are arithmetic and both {@code ordinals} and
     * {@code wanted} are in memory. Only tests — {@link #buildMask} populates the mask for the block
     * actually scored.
     */
    private int nextWantedBlock(int from) {
        int count = ordinals.length;
        for (int b = Math.max(from, 0); b < reader.numBlocks(); b++) {
            int start = b * BLOCK_SIZE;
            int len = Math.min(BLOCK_SIZE, count - start);
            if (wanted == null) {
                return b;
            }
            for (int j = 0; j < len; j++) {
                if (wanted.get(ordinals[start + j])) {
                    return b;
                }
            }
        }
        return -1;
    }

    /**
     * Populate {@link #valid} with the wanted positions of the block at {@code [start, start+len)};
     * returns {@code false} if none are wanted (nothing to score).
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

    /** Advance {@link #posInBlock} to the next wanted position in the current block. */
    private boolean advanceInBlock() {
        while (++posInBlock < blockLen) {
            if (valid.get(posInBlock)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int ord() {
        return ordinals[blockStart + posInBlock];
    }

    @Override
    public float score() {
        return scores[posInBlock];
    }
}
