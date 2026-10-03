/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;

/**
 * Prunes a MAXIMUM_INNER_PRODUCT posting on how far its shells sit from the centroid: splitting the score at the
 * centroid leaves one exact term and one the shell's radius caps outright.
 *
 * <pre>
 * ⟨q,v⟩  =  ⟨q,c⟩  +  ⟨q, v−c⟩   ≤   ⟨q,c⟩  +  ‖q‖·‖v−c‖
 * </pre>
 *
 * <p>Cauchy–Schwarz bounds the residual term by the radius, and {@code ⟨q,c⟩} and {@code ‖q‖} are both exact and
 * fixed for the cluster, so the ceiling is a function of the radius alone — and it only <em>grows</em> with it.
 *
 * <p>That is the whole difference from {@link EuclideanClipPruner}. L2's floor is V-shaped in the radius, so a
 * walk crosses the query's own shell and each arm prunes differently; this ceiling is monotone. An inner-product
 * posting stores its members farthest-first ({@code NearestFirstDistanceSorter} reverses the order for every
 * similarity but L2, because an inner product is largest where the residual is longest), so the ceiling falls as
 * the walk goes on. A block that cannot compete therefore takes the whole tail with it: this pruner only ever
 * says {@link Decision#TERMINATE}, never {@link Decision#SKIP}, and the binding member is the block's
 * <em>first</em> — its farthest.
 *
 * <p><b>No quantization margin.</b> The ceiling is exact while {@code minCompetitiveScore} is a k-th best ADC
 * score, so comparing them can prune a block whose true best would have placed — the margin of Appendix B.5
 * exists for precisely this. It is deliberately not applied: the margin belongs to the vector holding k-th
 * place, whose interval width lives in a block scored several clusters ago, and the design puts the resulting
 * recall loss at ≈0 for 4-bit codes and ≈0.08 at 1 bit. So this is sound where it is used and lossy at the
 * narrow widths, and the component test is what measures which.
 */
final class MaximumInnerProductClipPruner implements BlockPostingsPruner {

    /** {@code ‖c−v‖²} per member, descending, as an inner-product posting stores it. */
    private final float[] sortedDistances;

    private final int blockSize;

    /** {@code ⟨q,c⟩}, exact: computed at query time against the full-precision centroid. */
    private final float queryDotCentroid;

    /** {@code ‖q‖}, which converts a shell's radius into the most its residual term can contribute. */
    private final float queryNorm;

    MaximumInnerProductClipPruner(final float[] sortedDistances, final int blockSize, final float queryDotCentroid, final float queryNorm) {
        this.sortedDistances = sortedDistances;
        this.blockSize = blockSize;
        this.queryDotCentroid = queryDotCentroid;
        this.queryNorm = queryNorm;
    }

    @Override
    public Decision test(final int block, final float minCompetitiveScore) {
        // Farthest-first, so the block's first member is its longest residual and carries its highest ceiling.
        // Every later block is nearer the centroid and so bounded lower, which is what makes this a tail cut.
        final float farthest = (float) Math.sqrt(sortedDistances[block * blockSize]);
        return bound(farthest) <= minCompetitiveScore ? Decision.TERMINATE : Decision.SCORE;
    }

    /**
     * The similarity the scorer would give a vector whose residual is {@code radius} long and points straight at
     * the query — the best anything on that shell can do. Mapped through the same scaling
     * {@link ScalarQuantizedBlockScorer} applies, since the threshold is a number in that space.
     *
     * <p>Not {@code <} at the call site: the scan keeps a block only on a score strictly above the bar.
     */
    private float bound(final float radius) {
        final float dot = queryDotCentroid + queryNorm * radius;
        return dot >= 0 ? dot + 1 : 1f / (1f - dot);
    }
}
