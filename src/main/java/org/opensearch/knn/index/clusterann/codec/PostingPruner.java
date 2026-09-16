/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * Optional, pluggable per-block pruning for a posting scan. Given the current block's position range
 * and the competitive threshold, decides whether to score the block, skip it, or terminate the whole
 * posting. Pruners run after {@link BlockReader#seekToBlock(int)} (which has read the block's cheap
 * per-block metadata) but before {@link BlockReader#readBlockVectors()}, so a skip avoids the codes — the
 * bulk of a block — never the (tiny, already-read) corrections.
 *
 * <p>Several independent pruners can be chained; each works off cheap data it was given at
 * construction: the posting's sorted {@code ‖c−v‖} column (geometry, {@link ClipPostingPruner}) or the
 * block's correction columns already loaded by the reader (ADC bound, {@link AdcCorrectionsPruner}).
 * None reads block codes.
 * Pruning must be conservative — a bound at or above threshold must return {@link Decision#SCORE}.
 * {@link Decision#TERMINATE} is only valid when the pruner's metric is monotonic along the posting
 * (i.e. the column it inspects is sorted); otherwise a hopeless block only warrants {@link
 * Decision#SKIP}.
 */
public interface PostingPruner {

    enum Decision {
        /** The block may contain competitive vectors — read and score it. */
        SCORE,
        /** No vector in this block can be competitive — skip it, but keep scanning. */
        SKIP,
        /** Neither this block nor any later block can be competitive — stop the posting. */
        TERMINATE
    }

    /** Decide the block spanning positions {@code [blockStart, blockStart+blockLen)}. */
    Decision inspect(int blockStart, int blockLen, float minCompetitiveSimilarity);
}
