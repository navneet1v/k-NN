/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * {@link BlockScorer} over scalar-quantized blocks: each doc vector's codes are dotted against the
 * already-quantized query ({@link AdcQueryContext}) and corrected into a similarity.
 *
 * <p>It does not quantize. The query arrives quantized because that work is shared by every posting written
 * against the same reference centroid, so it belongs to the scan, not to one posting. Construction is
 * therefore free, which is what lets a posting be built and then abandoned by a pruner at no cost.
 *
 * <p>The dot kernel is chosen from the <b>pair</b> {@code (queryBits, docBits)}, which is what decides
 * whether a scan is asymmetric (ADC, {@code queryBits > docBits} — a finer query against coarse codes) or
 * symmetric (SDC, equal widths — cheapest, e.g. 1-bit against 1-bit reduces to popcount). Everything
 * else — the corrective terms and the four-term score — is identical either way, which is why one scorer
 * covers both and only {@link #dotProduct} varies. Query width is validated once, where the query is
 * quantized.
 *
 * <p>It owns the cursor and lends it out through {@link #reader()}. Codes and correction columns come from
 * that reader's current block, which the caller has already positioned and loaded, so a pruned block — or
 * one holding no candidates — never pays for its codes.
 */
final class ADCBlockScorer implements BlockScorer {

    private final ScalarQuantizedBlockReader reader; // owned; lent to the caller via reader()
    private final AdcQueryContext query;
    private final ScalarQuantizedLayout layout;

    ADCBlockScorer(ScalarQuantizedBlockReader reader, AdcQueryContext query, ScalarQuantizedLayout layout) {
        this.reader = reader;
        this.query = query;
        this.layout = layout;
    }

    @Override
    public BlockReader reader() {
        return reader;
    }

    /**
     * This scan's ADC block bound, over the same quantized query the scores use — which is exactly why it
     * comes from here: a bound computed from different query state than the scores it is compared against
     * would not be conservative.
     */
    PostingPruner correctionsBound() {
        return new AdcCorrectionsPruner(
            reader,
            layout.sim() == VectorSimilarityFunction.EUCLIDEAN,
            query.dimension,
            layout.packedBytes(),
            layout.docBitScale(),
            query.queryLower,
            query.queryScale,
            query.queryComponentSum,
            query.dpMinusNorm
        );
    }

    @Override
    public void scoreBlock(FixedBitSet valid, BlockCandidates out) throws IOException {
        final int[] positions = out.positions;
        final float[] scores = out.scores;

        // Columns and codes of the reader's current block; the codes were loaded by the caller's
        // readBlockVectors(), which is the phase this scoring belongs to.
        final byte[] codes = reader.codes();
        final float[] lower = reader.lower();
        final float[] upper = reader.upper();
        final float[] add = reader.add();
        final int[] sum = reader.sum();

        final int packedBytes = layout.packedBytes();
        final float docBitScale = layout.docBitScale();
        final boolean euclidean = layout.sim() == VectorSimilarityFunction.EUCLIDEAN;
        // Constant over the whole reference, not just this block — precomputed with the quantized query.
        final float qLowerDim = query.qLowerDim;
        final float qScaleCompSum = query.qScaleCompSum;
        final float dpMinusNorm = query.dpMinusNorm;
        final float queryCorrection = query.queryCorrection;

        // Walk the mask and emit as we go: one pass produces both the position list and the scores.
        int n = 0;
        for (int j = valid.nextSetBit(0); j != DocIdSetIterator.NO_MORE_DOCS; ) {
            float rawDot = dotProduct(codes, j * packedBytes);
            float docScale = (upper[j] - lower[j]) * docBitScale;

            // ⟨q−c, v−c⟩ from the four-term expansion of Σᵢ(l_q + s_q·bᵢ)(l_d + s_d·aᵢ). Three of the four
            // terms are O(1) — only the code dot product touches the block's bytes, which is what `sum`
            // (Σaᵢ) and queryComponentSum (Σbᵢ) are stored for.
            float score = lower[j] * qLowerDim
                + query.queryLower * docScale * sum[j]
                + lower[j] * qScaleCompSum
                + docScale * query.queryScale * rawDot;

            float adc;
            if (euclidean) {
                // ‖q−v‖² = ‖q−c‖² + ‖v−c‖² − 2⟨q−c, v−c⟩. The centroid cancels outright.
                float dist = queryCorrection + add[j] - 2f * score;
                // A squared distance can't be negative. When it is, the ADC estimate has suffered
                // catastrophic cancellation (all three terms large — a posting far from the query). The
                // estimate is unusable, so treat the vector as farthest (adc = 0) rather than clamping the
                // distance to 0, which would make a bogus estimate look like the nearest neighbor and
                // pollute the top-k / any downstream exact rescore or filter. Genuine near neighbours
                // (small ‖q−c‖) yield a small positive distance and are unaffected.
                adc = dist < 0f ? 0f : 1.0f / (1.0f + dist);
            } else {
                // ⟨q,v⟩ = ⟨q−c, v−c⟩ + ⟨v,c⟩ + ⟨q,c⟩ − ‖c‖². Here the centroid does not cancel: two query
                // constants and the stored ⟨v,c⟩ survive.
                float dot = score + add[j] + dpMinusNorm;
                if (layout.sim() == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
                    adc = dot >= 0 ? dot + 1 : 1f / (1f - dot);
                } else {
                    // DOT_PRODUCT and COSINE alike: both sides are unit-norm, so cosine is this dot product.
                    adc = Math.max((1.0f + dot) / 2.0f, 0f);
                }
            }

            positions[n] = j;
            scores[n] = adc;
            n++;
            if (++j == BLOCK_SIZE) {
                break;
            }
            j = valid.nextSetBit(j);
        }
        out.size = n;
    }

    /**
     * The {@code (queryBits, docBits)} kernel. The query packing and the kernel must agree, so this is the
     * one place the pairing is resolved; the query width itself was validated where it was quantized.
     */
    private float dotProduct(byte[] codes, int offset) {
        int packedBytes = layout.packedBytes();
        switch (layout.docBits()) {
            case 1:
                return Int4DotProduct.bit(query.transposedQuery, codes, offset, packedBytes);
            case 2:
                return Int4DotProduct.dibit(query.transposedQuery, codes, offset, packedBytes);
            default:
                return Int4DotProduct.nibble(query.transposedQuery, codes, offset, packedBytes);
        }
    }
}
