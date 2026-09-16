/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * What a {@link BlockScorer} scored in one block: the positions it visited and their scores, as parallel
 * arrays plus a size — the shape of Lucene's {@code DocAndFloatFeatureBuffer}. Reused for every block of a
 * posting, so nothing here is allocated per block.
 *
 * <p><b>Written entirely by the scorer</b>, including {@link #size}. The caller says <em>which</em>
 * positions it wants through the mask it passes to
 * {@link BlockScorer#scoreBlock(org.apache.lucene.util.FixedBitSet, BlockCandidates)}; the scorer walks that
 * mask and reports back what it did. Nothing has to compact the mask into a list — the scorer emits the
 * list as it goes.
 *
 * <p>Over {@code [0, size)} both arrays are <em>fully</em> defined: {@code scores[i]} is the score of the
 * vector at block-local position {@code positions[i]}, and {@code positions} is ascending because the mask
 * is walked in order. There are no holes to interpret — a position the mask excluded (a SOAR duplicate, a
 * filtered-out doc, an already-visited ordinal) is simply absent. Entries past {@code size} are stale
 * leftovers from an earlier block and mean nothing.
 *
 * <p>Positions are <b>block-local</b>: {@code 0} is the block's first vector, not the posting's. That is
 * the space a block's storage is indexed in; mapping back to a posting position, an ordinal, or a doc is
 * the caller's.
 */
public final class BlockCandidates {

    /** Block-local positions that were scored, ascending, over {@code [0, size)}. */
    public final int[] positions = new int[BLOCK_SIZE];

    /** Similarities (higher = closer) of {@link #positions}, over {@code [0, size)}. */
    public final float[] scores = new float[BLOCK_SIZE];

    /** How many entries of {@link #positions}/{@link #scores} are meaningful. */
    public int size;
}
