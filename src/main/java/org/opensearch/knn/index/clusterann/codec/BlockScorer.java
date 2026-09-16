/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

/**
 * Scores blocks of one posting for a query — the storage-/quantization-specific half of a block scan, which
 * the generic {@link BlockPostingScorer} drives. The query and its quantization parameters are captured at
 * construction; the vectors come from the block storage this scorer owns.
 *
 * <p>It owns the cursor and lends it out through {@link #reader()}, the way a Lucene {@code VectorScorer}
 * owns and hands out its iterator. So there is exactly one cursor, positioned by the caller and read by the
 * scorer — nothing has to keep two of them in step, and a scorer cannot be paired with block storage it
 * doesn't understand. The caller drives {@link #reader()} through the cheap/expensive boundary
 * ({@link BlockReader#seekToBlock}, then {@link BlockReader#readBlockVectors()} only for blocks it decides
 * to score) and then calls {@link #scoreBlock}; the scorer itself never seeks and never loads.
 */
public interface BlockScorer {

    /**
     * The block storage this scorer scores from, for the caller to position. Same instance every call.
     * Exposed as the generic cursor because that is all a caller needs; the concrete type — and the
     * per-block metadata it carries — stays between this scorer and its reader.
     */
    BlockReader reader();

    /**
     * Score the positions set in {@code valid} of {@link #reader()}'s current block, appending each one's
     * position and score to {@code out}.
     *
     * <p>Walk {@code valid} in ascending order and, for the {@code n}th set bit at block-local position
     * {@code j}, write {@code out.positions[n] = j} and {@code out.scores[n] = score}; finish by setting
     * {@code out.size}. Emitting the list while scoring is the point of taking the mask directly — nothing
     * needs a separate pass to turn it into one.
     *
     * <p>Which positions are wanted is entirely the caller's decision, and a mask is how it says so: order
     * and uniqueness are structural, so an implementation never has to trust a hand-built list. Bits at or
     * beyond the block's vector count are never set.
     */
    void scoreBlock(FixedBitSet valid, BlockCandidates out) throws IOException;
}
