/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * ADC {@link PostingPruner}: bounds a block's best achievable similarity from the
 * {@link org.apache.lucene.util.quantization.OptimizedScalarQuantizer} correction columns (which the
 * {@link ScalarQuantizedBlockReader} has already read for the current block) plus the shared
 * {@link AdcQueryContext}, and skips the block when that bound cannot beat the threshold — no codes
 * read.
 *
 * <p>The bound is a per-vector maximum over the block using absolute terms (a conservative
 * over-estimate of the dot product), converted to a MIP-style similarity that upper-bounds the
 * DOT/cosine similarity too. It only ever {@link Decision#SKIP}s, never terminates: correction
 * columns are not ordered across blocks, so a hopeless block says nothing about later ones. For
 * {@code EUCLIDEAN} the corrections give no useful bound, so this never skips — L2's block-skip is
 * the geometry {@link ClipPostingPruner} instead.
 */
final class AdcCorrectionsPruner implements PostingPruner {

    private final boolean euclidean;
    private final int dimension;
    private final int packedBytes;
    private final float docBitScale;
    private final float queryLower;
    private final float queryScale;
    private final float queryComponentSum;
    private final float dpMinusNorm; // ⟨q,c⟩ − ‖c‖²

    // The reader's reused per-block correction buffers, filled by nextBlock() before each inspect().
    private final float[] lower;
    private final float[] upper;
    private final float[] add;
    private final int[] sum;

    AdcCorrectionsPruner(
        AdcQueryContext ctx,
        ScalarBitEncoding encoding,
        VectorSimilarityFunction sim,
        int dimension,
        ScalarQuantizedBlockReader reader
    ) {
        this.euclidean = sim == VectorSimilarityFunction.EUCLIDEAN;
        this.dimension = dimension;
        this.packedBytes = encoding.docPackedBytes(dimension);
        this.docBitScale = encoding.docBitScale();
        this.queryLower = ctx.queryLower;
        this.queryScale = ctx.queryScale;
        this.queryComponentSum = ctx.queryComponentSum;
        this.dpMinusNorm = ctx.centroidDp - ctx.centroidNormSq;
        this.lower = reader.lower();
        this.upper = reader.upper();
        this.add = reader.add();
        this.sum = reader.sum();
    }

    @Override
    public PostingPruner.Decision inspect(int blockStart, int blockLen, float minCompetitiveSimilarity) {
        if (euclidean) {
            return PostingPruner.Decision.SCORE; // corrections give no useful bound for L2
        }
        final float absQLower = Math.abs(queryLower);
        final float absQScale = Math.abs(queryScale);
        final float absQCompSum = Math.abs(queryComponentSum);
        float maxUpperBound = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < blockLen; j++) {
            float docScale = (upper[j] - lower[j]) * docBitScale;
            float maxScore = lower[j] * queryLower * dimension
                + absQLower * docScale * Math.abs(sum[j])
                + Math.abs(lower[j]) * absQScale * absQCompSum
                + docScale * absQScale * packedBytes * 4f;
            float upperBound = maxScore + add[j] + dpMinusNorm;
            if (upperBound > maxUpperBound) {
                maxUpperBound = upperBound;
            }
        }
        float maxSim = maxUpperBound >= 0 ? maxUpperBound + 1 : 1f / (1f - maxUpperBound);
        return maxSim < minCompetitiveSimilarity ? Decision.SKIP : Decision.SCORE;
    }
}
