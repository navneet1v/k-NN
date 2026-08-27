/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * Scores the current block of a {@link BlockReader} for a query — the one storage-/quantization-specific
 * step the generic {@link BlockPostingScorer} delegates. All context (the query, the block
 * storage it reads codes/corrections from, quantization parameters) is captured in the implementation's
 * constructor, so scoring is opaque to the iterator: given the block length, a per-position "score me"
 * mask, and an output array, write similarities for the wanted positions.
 */
public interface BlockScorer {

    /**
     * Score the current block's wanted positions into {@code scores}. Only positions with
     * {@code valid.get(j) == true} are scored (others are SOAR duplicates / filtered-out docs);
     * {@code scores[j]} for skipped positions is left untouched and must not be read. The block's codes
     * are read from the storage captured at construction — so a pruned or fully-masked block, whose
     * {@code scoreBlock} is never called, pays no code I/O. Scores are similarities (higher = closer).
     */
    void scoreBlock(int blockLen, Bits valid, float[] scores) throws IOException;
}
