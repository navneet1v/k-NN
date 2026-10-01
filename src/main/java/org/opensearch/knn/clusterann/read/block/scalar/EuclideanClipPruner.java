/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;

/**
 * Prunes a EUCLIDEAN posting on how far its shells sit from the query: two points at known distances from one
 * centroid cannot be closer than the difference of those distances, so a shell's radius floors the distance of
 * everything on it.
 *
 * <p>That floor is {@code |‖q−c‖ − ‖c−v‖|} — V-shaped in the radius, not monotone — and the two arms prune
 * differently. Members are stored nearest-first, so a walk crosses the query's own shell once:
 *
 * <ul>
 *   <li><b>Inside it</b> the floor shrinks as the walk goes on, so a block that cannot compete says
 *       {@link Decision#SKIP} — the blocks after it come closer and may well compete. The binding member is the
 *       block's <em>farthest</em>, the one nearest the query's shell.
 *   <li><b>Outside it</b> the floor only grows, so a block that cannot compete says {@link Decision#TERMINATE}
 *       and takes the tail with it. The binding member there is the block's <em>nearest</em>.
 *   <li>A block that straddles the query's shell holds members that could be touching it, and bounds nothing.
 * </ul>
 *
 * <p>The bound is the one {@link ScalarQuantizedBlockScorer} already clamps a quantized estimate to, so a
 * scored vector cannot beat what this pruned.
 */
final class EuclideanClipPruner implements BlockPostingsPruner {

    /** {@code ‖c−v‖²} per member, ascending, as the posting stores it. */
    private final float[] sortedDistances;

    private final int blockSize;

    private final float queryToCentroid;

    EuclideanClipPruner(final float[] sortedDistances, final int blockSize, final float queryToCentroid) {
        this.sortedDistances = sortedDistances;
        this.blockSize = blockSize;
        this.queryToCentroid = queryToCentroid;
    }

    @Override
    public Decision test(final int block, final float minCompetitiveScore) {
        int first = block * blockSize;
        float nearest = (float) Math.sqrt(sortedDistances[first]);
        if (nearest > queryToCentroid) {
            return bound(nearest - queryToCentroid) <= minCompetitiveScore ? Decision.TERMINATE : Decision.SCORE;
        }

        int last = Math.min(first + blockSize, sortedDistances.length) - 1;
        float farthest = (float) Math.sqrt(sortedDistances[last]);
        if (farthest < queryToCentroid) {
            return bound(queryToCentroid - farthest) <= minCompetitiveScore ? Decision.SKIP : Decision.SCORE;
        }

        // The block straddles the query's own shell, so one of its members may be touching the query.
        return Decision.SCORE;
    }

    /**
     * The similarity the scorer would give a vector exactly {@code gap} away — the best any member this far off
     * can do. Not {@code <} at the call sites: the scan keeps a block only on a score strictly above the bar.
     */
    private static float bound(final float gap) {
        return 1f / (1f + gap * gap);
    }
}
