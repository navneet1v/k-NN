/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.io.IOException;

/**
 * Walks one sequence block by block, scoring the positions it wants and handing them out one at a time.
 *
 * <p>Block geometry — how the sequence is divided and how many vectors the block it is on holds — comes from
 * the {@link BlockVectorFormat.Reader}. Turning a block-local position back into an ordinal is this class's
 * own business, since the ordinal mapping belongs to the posting rather than to the block storage.
 *
 * <p>Which blocks are worth anything is the {@link BlockPostingsPruner}'s business. It is consulted before any
 * IO, and once per block ahead of the current one, so the walk steps over blocks it never positions on and a
 * prefetch hint only ever names a block that will be fetched.
 */
public class BlockPostingScorer implements PostingScorer {

    /** No block left to visit, either because the sequence ran out or because a pruner terminated it. */
    private static final int NO_MORE_BLOCKS = Integer.MAX_VALUE;

    private final BlockVectorScorer scorer;
    private final BlockPostingsPruner pruner;
    private final BlockVectorFormat.Reader reader;
    private final int[] ordinals;
    private final AcceptedPositions positions;
    private final int numBlocks;
    private final int blockSize;

    private final BlockVectorScorer.BlockCandidates candidates = new BlockVectorScorer.BlockCandidates();

    /** Positions of the current block the filter accepts. Sized to a full block; a partial one leaves a tail unset. */
    private final FixedBitSet validPos;

    /**
     * Next block to visit, already tested by the lookahead that named it, so the walk takes it as it stands.
     * {@code -1} until the first {@link #advance}, which has no lookahead behind it and so tests block 0 itself;
     * {@link #NO_MORE_BLOCKS} once the walk is over.
     */
    private int nextBlock = -1;

    /** Set when a pruner says nothing later can compete. The candidates already scored are still handed out. */
    private boolean terminated;

    /**
     * Where the block in {@link #candidates} starts, so {@link #ord()} maps back after {@link #nextBlock}
     * moves on.
     */
    private int scoredBlockVectorOffset;

    /** Cursor into {@link #candidates}. */
    private int cursor = -1;

    /**
     * {@code pruner} is the extra pruning to apply, {@link BlockPostingsPruner#NONE} for none. The filter
     * becomes an {@link AcceptedPositions}, which both masks the positions a block may score and contributes
     * its own block pruning, so the blocks that get skipped and the positions that get scored cannot disagree.
     */
    public BlockPostingScorer(
        final BlockVectorScorer scorer,
        final int[] ordinals,
        final Bits acceptedOrds,
        final BlockPostingsPruner pruner
    ) {
        this.scorer = scorer;
        this.reader = scorer.reader();
        this.ordinals = ordinals;
        this.numBlocks = reader.numBlocks();
        this.blockSize = reader.blockSize();
        this.validPos = new FixedBitSet(blockSize);
        this.positions = AcceptedPositions.of(ordinals, acceptedOrds, blockSize);
        this.pruner = BlockPostingsPruner.of(pruner, positions.pruner());
        candidates.growNoCopy(blockSize);
    }

    @Override
    public boolean advance(float minCompetitiveSimilarity) throws IOException {
        if (cursor != -1 && ++cursor < candidates.getSize()) {
            return true;
        }

        while (!terminated) {
            // nextBlock -1 tests the first block for pruning
            int block = nextBlock == -1 ? seek(0, minCompetitiveSimilarity) : nextBlock;
            if (block == NO_MORE_BLOCKS) {
                break;
            }

            if (!reader.advance(block)) {
                throw new IllegalStateException("BlockPostingScorer can never ask for block outside the range");
            }

            int vectorCount = reader.blockVectorCount();

            // One lookahead, serving as both the prefetch target and the next block to visit: issued before the
            // fetch below so it overlaps this block's IO as well as its scoring.
            nextBlock = seek(block + 1, minCompetitiveSimilarity);
            if (nextBlock != NO_MORE_BLOCKS) {
                reader.prefetchBlock(nextBlock);
            }

            int postingVectorOffset = block * blockSize;
            positions.mask(block, vectorCount, validPos);

            if (validPos.cardinality() != 0) {
                reader.fetchBlock();
                ClusterANNQueryValue.BLOCKS_FETCHED.increment();
                reader.readBlockVectors();

                float maxScore = scorer.scoreBlock(validPos, candidates);
                ClusterANNQueryValue.VECTORS_SCORED.incrementBy(validPos.cardinality());
                scoredBlockVectorOffset = postingVectorOffset;
                cursor = 0;
                if (maxScore > minCompetitiveSimilarity) {
                    return true;
                }
            }
        }

        candidates.setSize(0);
        cursor = -1;
        return false;
    }

    @Override
    public int ord() {
        return ordinals[scoredBlockVectorOffset + candidates.getPositions()[cursor]];
    }

    @Override
    public float score() {
        return candidates.getScores()[cursor];
    }

    /**
     * First block at or after {@code from} the pruner wants scored, or {@link #NO_MORE_BLOCKS} when none is
     * left. Costs no IO, which is what lets it be called for a block the walk has not reached yet.
     */
    private int seek(int from, float minCompetitiveSimilarity) {
        for (int block = from; block < numBlocks; block++) {
            BlockPostingsPruner.Decision decision = pruner.test(block, minCompetitiveSimilarity);
            if (decision == BlockPostingsPruner.Decision.SCORE) {
                ClusterANNQueryValue.PRUNER_SCORE.increment();
                return block;
            }
            if (decision == BlockPostingsPruner.Decision.TERMINATE) {
                ClusterANNQueryValue.PRUNER_TERMINATE.increment();
                // The tail this decision bought: counted here because here is the only place that still knows
                // where the walk stopped, and the blocks beyond are never tested again to be counted later.
                ClusterANNQueryValue.BLOCKS_TERMINATED.incrementBy(numBlocks - block);
                terminated = true;
                return NO_MORE_BLOCKS;
            }
            // Only SKIP reaches here, the loop's own fall-through being what steps over the block.
            ClusterANNQueryValue.PRUNER_SKIP.increment();
        }
        return NO_MORE_BLOCKS;
    }

}
