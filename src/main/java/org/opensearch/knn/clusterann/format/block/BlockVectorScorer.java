/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IntsRef;

/**
 * Scores blocks of one sequence against a query. The query is fixed at construction time; the
 * vectors come from the block reader this scorer owns.
 *
 * <p>The scorer owns a single cursor and lends it out through {@link #reader()}, The interface is inspired by
 * {@code VectorScorer.Bulk} owns its iterator. This structure helps with having only one cursor — positioned by the
 * caller and read by the scorer
 *
 * <p>The caller seeks to and loads the blocks it wants, then calls {@link #scoreBlock}. This
 * scorer never seeks and never loads.
 */
public interface BlockVectorScorer {

    /**
     * The block storage this scorer reads from, for the caller to position. Same instance every call.
     */
    BlockVectorFormat.Reader reader();

    /**
     * Scores the positions set in {@code validPos} of {@link #reader()}'s current block, appending each
     * one's position and score to {@code out}.
     *
     * <p>The caller chooses which positions it wants by setting bits in the mask; order and
     * uniqueness come for free, so the implementation never has to trust a hand-built list. Bits at
     * or beyond the block's vector count are never set.
     */
    float scoreBlock(FixedBitSet validPos, BlockCandidates out);

    /** Scores from one block, in ascending position order. A caller-owned buffer the scorer fills. */
    @Getter
    @Setter
    final class BlockCandidates {
        private static final float[] EMPTY_FLOATS = new float[0];

        private int[] positions = IntsRef.EMPTY_INTS;
        private float[] scores = EMPTY_FLOATS;
        private int size;

        public void growNoCopy(int minSize) {
            if (positions.length < minSize) {
                positions = ArrayUtil.growNoCopy(positions, minSize);
                scores = new float[positions.length];
            }
        }
    }
}
