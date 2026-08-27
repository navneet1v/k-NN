/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * {@link ScalarQuantizedBlockScorer} over scalar-quantized blocks: using the query already quantized
 * relative to this posting's centroid ({@link AdcQueryContext}), each doc vector's codes are dotted
 * against it and corrected into a similarity. Pure scoring — block-skip lives in the pruning strategies.
 *
 * <p>The dot kernel is chosen from the <b>pair</b> {@code (queryBits, docBits)}, which is what decides
 * whether a scan is asymmetric (ADC, {@code queryBits > docBits} — a finer query against coarse codes) or
 * symmetric (SDC, equal widths — cheapest, e.g. 1-bit against 1-bit reduces to popcount). Everything
 * else — the corrective terms and the four-term score — is identical either way, which is why one scorer
 * covers both and only {@link #dotProduct} varies. Doc width comes from the field's
 * {@link ScalarBitEncoding}; query width comes from the query ({@link ScanParams#queryBits()}). Only the
 * 4-bit-query kernels exist today ({@link Int4DotProduct}); other pairings are rejected rather than
 * scored against a mismatched packing.
 *
 * <p>All context is captured at construction: the query quantization state and the
 * {@link ScalarQuantizedBlockReader} it reads each block's codes + corrections from. So
 * {@link #scoreBlock} takes no columns — it pulls them from the reader's current block, which is why a
 * pruned/masked block (never scored) never reads its codes.
 */
final class ADCBlockScorer implements ScalarQuantizedBlockScorer {

    private final ScalarQuantizedBlockReader reader; // block storage this scorer reads codes/corrections from
    private final VectorSimilarityFunction sim;
    private final int dimension;
    private final int docBits;   // stored width (field layout)
    private final int queryBits; // query-side width (per-query knob) — with docBits, selects the kernel
    private final int packedBytes;
    private final float docBitScale;

    // Query quantization state (relative to this posting's centroid), shared with the pruner.
    private final byte[] transposedQuery;
    private final float queryLower;
    private final float queryScale;
    private final float queryComponentSum;
    private final float queryAdditionalCorrection;
    private final float centroidNormSq;
    private final float centroidDp; // ⟨q,c⟩ — used for inner-product metrics only

    ADCBlockScorer(
        ScalarQuantizedBlockReader reader,
        AdcQueryContext ctx,
        ScalarBitEncoding encoding,
        VectorSimilarityFunction sim,
        int dimension,
        int queryBits
    ) {
        this.reader = reader;
        this.sim = sim;
        this.dimension = dimension;
        this.docBits = encoding.docBits();
        this.queryBits = queryBits;
        this.packedBytes = encoding.docPackedBytes(dimension);
        this.docBitScale = encoding.docBitScale();

        this.transposedQuery = ctx.transposedQuery;
        this.queryLower = ctx.queryLower;
        this.queryScale = ctx.queryScale;
        this.queryComponentSum = ctx.queryComponentSum;
        this.queryAdditionalCorrection = ctx.queryAdditionalCorrection;
        this.centroidNormSq = ctx.centroidNormSq;
        this.centroidDp = ctx.centroidDp;
    }

    @Override
    public void scoreBlock(int blockLen, Bits valid, float[] scores) throws IOException {
        // Read this block's codes + corrections from the storage captured at construction.
        byte[] codes = reader.readCodes();
        float[] lower = reader.lower();
        float[] upper = reader.upper();
        float[] add = reader.add();
        int[] sum = reader.sum();

        final float qLowerDim = queryLower * dimension;
        final float qScaleCompSum = queryScale * queryComponentSum;
        final float dpMinusNorm = centroidDp - centroidNormSq;
        final boolean euclidean = sim == VectorSimilarityFunction.EUCLIDEAN;

        for (int j = 0; j < blockLen; j++) {
            if (!valid.get(j)) {
                continue;
            }
            float rawDot = dotProduct(codes, j * packedBytes);
            float docScale = (upper[j] - lower[j]) * docBitScale;
            float score = lower[j] * qLowerDim
                + queryLower * docScale * sum[j]
                + lower[j] * qScaleCompSum
                + docScale * queryScale * rawDot;

            float adc;
            if (euclidean) {
                float dist = queryAdditionalCorrection + add[j] - 2f * score;
                // A squared distance can't be negative. When it is, the ADC estimate has suffered
                // catastrophic cancellation (‖q−c‖² + ‖v−c‖² − 2·⟨q−c,v−c⟩ with all terms large — a
                // posting far from the query). The estimate is unusable, so treat the vector as
                // farthest (adc = 0) rather than clamping the distance to 0, which would make a bogus
                // estimate look like the nearest neighbor and pollute the top-k / any downstream
                // exact rescore or filter. Genuine near neighbours (small ‖q−c‖) yield a small
                // positive distance and are unaffected.
                adc = dist < 0f ? 0f : 1.0f / (1.0f + dist);
            } else {
                float dot = score + add[j] + dpMinusNorm;
                if (sim == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
                    adc = dot >= 0 ? dot + 1 : 1f / (1f - dot);
                } else {
                    adc = Math.max((1.0f + dot) / 2.0f, 0f);
                }
            }
            scores[j] = adc;
        }
    }

    /**
     * The {@code (queryBits, docBits)} kernel. The query packing and the kernel must agree, so this is
     * the one place the pairing is resolved. Today only a 4-bit query is packed
     * ({@link AdcQueryContext}), giving the three asymmetric/symmetric-at-4 kernels below; a narrower
     * query (e.g. symmetric 1-bit, an AND+popcount) needs its own packing and kernel, so it fails loudly
     * here rather than being scored against a layout it doesn't match.
     */
    private float dotProduct(byte[] codes, int offset) {
        if (queryBits != 4) {
            throw new UnsupportedOperationException(
                "No dot kernel for queryBits=" + queryBits + " with docBits=" + docBits + " (only a 4-bit query is implemented)");
        }
        switch (docBits) {
            case 1:
                return Int4DotProduct.bit(transposedQuery, codes, offset, packedBytes);
            case 2:
                return Int4DotProduct.dibit(transposedQuery, codes, offset, packedBytes);
            default:
                return Int4DotProduct.nibble(transposedQuery, codes, offset, packedBytes);
        }
    }
}
