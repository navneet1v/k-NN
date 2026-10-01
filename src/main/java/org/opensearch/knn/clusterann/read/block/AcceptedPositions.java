/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;

/**
 * Which positions of a block a scan may score, and whether a block holds any at all.
 *
 * <p>The mask is the correctness mechanism — nothing outside it is ever scored — while {@link #pruner()} is only
 * an IO saving, so a strategy with nothing cheap to say may decline to prune. Both answers come from one place
 * so they cannot disagree about what the filter accepts, which is also what leaves room for a strategy that
 * reads the filter differently (a translated bitset, say) without the rest of the walk noticing.
 */
interface AcceptedPositions {

    /** No filter: every position of every block is scoreable, and there is nothing to prune on. */
    AcceptedPositions ALL = new All();

    /** Sets the positions of {@code block} below {@code vectorCount} the filter accepts, clearing the rest. */
    void mask(int block, int vectorCount, FixedBitSet out);

    /** This filter as a block pruner, or {@link BlockPostingsPruner#NONE} when testing blocks cannot pay. */
    BlockPostingsPruner pruner();

    static AcceptedPositions of(final int[] ordinals, final Bits acceptedOrds, final int blockSize) {
        if (acceptedOrds == null) {
            return ALL;
        }
        return new Scanning(ordinals, acceptedOrds, blockSize);
    }

    final class All implements AcceptedPositions {

        @Override
        public void mask(final int block, final int vectorCount, final FixedBitSet out) {
            out.clear();
            out.set(0, vectorCount);
        }

        @Override
        public BlockPostingsPruner pruner() {
            return BlockPostingsPruner.NONE;
        }
    }

    /** Reads the filter per block, so a scan pays only for the blocks it reaches. */
    final class Scanning implements AcceptedPositions, BlockPostingsPruner {

        private final int[] ordinals;
        private final Bits acceptedOrds;
        private final int blockSize;

        Scanning(final int[] ordinals, final Bits acceptedOrds, final int blockSize) {
            this.ordinals = ordinals;
            this.acceptedOrds = acceptedOrds;
            this.blockSize = blockSize;
        }

        @Override
        public Decision test(final int block, final float minCompetitiveScore) {
            int startPos = block * blockSize;
            int endPos = Math.min(startPos + blockSize, ordinals.length);
            for (int pos = startPos; pos < endPos; pos++) {
                if (acceptedOrds.get(ordinals[pos])) {
                    return Decision.SCORE;
                }
            }
            return Decision.SKIP;
        }

        @Override
        public void mask(final int block, final int vectorCount, final FixedBitSet out) {
            out.clear();
            int startPos = block * blockSize;
            for (int pos = 0; pos < vectorCount; pos++) {
                if (acceptedOrds.get(ordinals[startPos + pos])) {
                    out.set(pos);
                }
            }
        }

        @Override
        public BlockPostingsPruner pruner() {
            return this;
        }
    }
}
