/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.read.ComputeScoreContext;
import org.opensearch.knn.clusterann.read.ScoreComputer;
import org.opensearch.knn.quantization.Int4DotProduct;

/**
 * The format's own {@link ScoreComputer} for scalar-quantized codes: asymmetric distance computation in plain Java.
 *
 * <p>Scores a query against the codes without decompressing them. The query was quantized once per cluster, against
 * the cluster's centroid, and each vector's score falls out of a dot product over its packed codes plus corrections
 * that cost O(1) per vector. The codes are lossy, so every score is an estimate of the similarity the full-precision
 * vectors would give.
 *
 * <p>The four-term expansion never depended on how the code dot product {@code Σᵢaᵢbᵢ} was obtained, so the only thing
 * the encoding changes is {@link #dotProduct}: a 1- or 2-bit document code is bit-plane transposed and popcounted
 * against the 4-bit query ({@code SINGLE_BIT_QUERY_NIBBLE}, {@code DIBIT_QUERY_NIBBLE}), while 4-bit
 * {@code PACKED_NIBBLE} stores whole codes and takes a plain integer dot product.
 *
 * <p>Stateless; use {@link #INSTANCE}.
 */
public final class JavaAdcScoreComputer implements ScoreComputer {

    public static final JavaAdcScoreComputer INSTANCE = new JavaAdcScoreComputer();

    private JavaAdcScoreComputer() {}

    @Override
    public float score(final ComputeScoreContext context, final int[] positions, final int count, final float[] scores) {
        final SQComputeContext block = (SQComputeContext) context;
        final SQScanContext quantizedQuery = block.query();
        final VectorSimilarityFunction sim = quantizedQuery.similarityFunction();
        final ScalarEncoding encoding = block.encoding();
        final int dimension = quantizedQuery.query().length;
        final int docBits = encoding.getDocBitsPerDim();
        final boolean asymmetric = encoding.isAsymmetric();
        final int packedBytes = block.packedBytes();
        final byte[] queryCodes = quantizedQuery.transposed();

        final byte[] codes = block.codes();
        final float[] lower = block.lower();
        final float[] upper = block.upper();
        final float[] addCor = block.addCor();
        final int[] sum = block.sum();

        // these are pre-computed purely to avoid computes in the loop
        final float step = 1f / ((1 << docBits) - 1);
        final float qLowerDim = quantizedQuery.lower() * dimension;  // l_q * dim
        final float qScaleCompSum = quantizedQuery.scale() * quantizedQuery.componentSum(); // s_q * Σᵢbᵢ
        final float centroidDotProductMinusNorm = quantizedQuery.correction() - quantizedQuery.centroidNormSq(); // ⟨q,c⟩ − ‖c‖²
        final float queryNorm = sim == VectorSimilarityFunction.EUCLIDEAN ? (float) Math.sqrt(quantizedQuery.correction()) : 0f;

        float maxScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            final int index = positions[i];
            float rawDot = dotProduct(asymmetric, docBits, queryCodes, codes, index * packedBytes, packedBytes); // Σᵢaᵢbᵢ
            float docScale = (upper[index] - lower[index]) * step;    // s_d = (u_d - l_d)/ 2^(bits-1);

            // ⟨q−c, v−c⟩ from the four-term expansion of Σᵢ(l_q + s_q·bᵢ)(l_d + s_d·aᵢ). Three of the four
            // terms are O(1) — only the code dot product touches the block's bytes, which is what `sum`
            // (Σaᵢ) and queryComponentSum (Σbᵢ) are stored for.
            float score = lower[index] * qLowerDim                   // l_d * l_q * dim
                + quantizedQuery.lower() * docScale * sum[index]   // l_q * s_d * Σᵢaᵢ
                + lower[index] * qScaleCompSum                   // l_d * s_q * Σᵢbᵢ
                + docScale * quantizedQuery.scale() * rawDot;      // s_d * s_q * Σᵢaᵢbᵢ

            float adc = switch (sim) {
                case EUCLIDEAN -> {
                    // ‖q−v‖² = ‖q−c‖² + ‖v−c‖² − 2⟨q−c, v−c⟩
                    float distance = quantizedQuery.correction() + addCor[index] - 2f * score;
                    if (distance < 0f) {
                        // Both norms are exact, so the triangle inequality gives a provable floor on the true
                        // squared distance: two points at known distances from the centroid cannot be closer
                        // than the difference of those distances. An estimate below it is quantization error, and
                        // clamping there is sound in both directions — unlike 0 (claims infinitely far) or
                        // max(distance, 0) (claims touching, which then poisons minCompetitiveSimilarity).
                        float gap = queryNorm - (float) Math.sqrt(addCor[index]);
                        distance = gap * gap;
                    }
                    yield 1.0f / (1.0f + distance);
                }
                case MAXIMUM_INNER_PRODUCT -> {
                    // ⟨q,v⟩ = ⟨q−c, v−c⟩ + ⟨v,c⟩ + ⟨q,c⟩ − ‖c‖²
                    float dot = score + addCor[index] + centroidDotProductMinusNorm;
                    yield dot >= 0 ? dot + 1 : 1f / (1f - dot);
                }
                case COSINE -> {
                    float dot = score + addCor[index] + centroidDotProductMinusNorm;
                    yield Math.max((1.0f + dot) / 2.0f, 0f);
                }
                case DOT_PRODUCT -> throw new IllegalStateException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
            };
            scores[i] = adc;
            if (adc > maxScore) {
                maxScore = adc;
            }
        }
        return maxScore;
    }

    /**
     * {@code Σᵢaᵢbᵢ} over one vector's codes and the query's.
     *
     * <p>Every width reads the vector's codes where they lie: the narrow ones popcount 4-bit query planes against the
     * document planes, the packed nibble multiplies whole codes. All three take an offset, so no candidate's codes are
     * copied anywhere to be scored.
     */
    private static float dotProduct(boolean asymmetric, int docBits, byte[] queryCodes, byte[] codes, int offset, int packedBytes) {
        if (asymmetric) {
            return docBits == 1
                ? Int4DotProduct.bit(queryCodes, codes, offset, packedBytes)
                : Int4DotProduct.dibit(queryCodes, codes, offset, packedBytes);
        }
        return Int4DotProduct.nibble(queryCodes, codes, offset, packedBytes);
    }
}
