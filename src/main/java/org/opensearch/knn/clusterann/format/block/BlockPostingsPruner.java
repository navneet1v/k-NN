/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import java.util.ArrayList;
import java.util.List;

/**
 * Decides, from metadata alone, what a block iteration should do with a block: read it, step over it, or stop
 * the posting there.
 *
 * <p>A pruner is asked about blocks the walk may never visit — the block after the current one, so a prefetch
 * hint can name the block that will actually be fetched — so {@link #test} must neither read nor leave a trace.
 */
@FunctionalInterface
public interface BlockPostingsPruner {

    enum Decision {
        /** The block may contain competitive vectors — read and score it. */
        SCORE,
        /** No vector in this block can be competitive — skip it, but keep scanning. */
        SKIP,
        /** Neither this block nor any later block can be competitive — stop the posting. */
        TERMINATE
    }

    /** Always {@link Decision#SCORE}: the pruner to use when there is nothing to prune on. */
    BlockPostingsPruner NONE = (block, minCompetitiveScore) -> Decision.SCORE;

    /**
     * What to do with {@code block} at {@code minCompetitiveScore}. No IO and no side effects, since the caller
     * tests blocks it may never visit.
     *
     * <p>Must be monotone in {@code minCompetitiveScore}: a block pruned at one threshold stays pruned at every
     * higher one. That is what lets a caller decide a block once and trust the decision after the threshold has
     * risen.
     */
    Decision test(int block, float minCompetitiveScore);

    /**
     * The pruners as one, strongest decision winning. Argument order is kept and every pruner runs on every
     * block that is not terminated, so the cheapest — and the one that can terminate — goes first.
     */
    static BlockPostingsPruner of(final BlockPostingsPruner... pruners) {
        final List<BlockPostingsPruner> live = new ArrayList<>(pruners.length);
        for (BlockPostingsPruner pruner : pruners) {
            if (pruner != null && pruner != NONE) {
                live.add(pruner);
            }
        }
        return switch (live.size()) {
            case 0 -> NONE;
            case 1 -> live.get(0);
            default -> new CompositeBlockPostingsPruner(live.toArray(BlockPostingsPruner[]::new));
        };
    }
}
