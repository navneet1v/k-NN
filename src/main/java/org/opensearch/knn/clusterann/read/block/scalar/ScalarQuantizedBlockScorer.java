/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;

/**
 * {@link BlockVectorScorer} that scores a query against scalar-quantized codes without decompressing them: the query is
 * quantized once at construction, against the cluster's centroid, and each vector's score then falls out of a dot
 * product over the packed codes plus corrections that cost O(1) per vector.
 *
 * <p>Both sides being centered on one centroid is what ties an instance to a single cluster and a single query. The
 * codes are lossy, so every score is an estimate of the similarity the full-precision vectors would give.
 *
 * <p><b>Asymmetric and symmetric encodings alike.</b> The four-term expansion below never depended on how the code dot
 * product {@code Σᵢaᵢbᵢ} was obtained, so the only thing the encoding changes is {@link #dotProduct}: a 1- or 2-bit
 * document code is bit-plane transposed and popcounted against the 4-bit query ({@code SINGLE_BIT_QUERY_NIBBLE},
 * {@code DIBIT_QUERY_NIBBLE}), while 4-bit {@code PACKED_NIBBLE} stores whole codes and takes a plain integer dot
 * product. This class was once named for the asymmetric case alone, which stopped being true at 4 bits.
 */
public class ScalarQuantizedBlockScorer implements BlockVectorScorer {

    private final ScalarQuantizedBlockReader reader;
    private final VectorSimilarityFunction sim;
    private final int dimension;
    private final int docBits;
    private final boolean asymmetric;
    private final int packedBytes;
    private final int queryPackedBytes;
    private final float queryNorm;

    private final SQScanContext quantizedQuery;

    public ScalarQuantizedBlockScorer(
        ScalarQuantizedBlockReader reader,
        SQScanContext queryContext,
        ScalarEncoding encoding,
        VectorSimilarityFunction sim
    ) {
        this.reader = reader;
        this.sim = sim;
        this.dimension = queryContext.query().length;
        this.docBits = encoding.getDocBitsPerDim();
        this.asymmetric = encoding.isAsymmetric();

        switch (encoding) {
            case SINGLE_BIT_QUERY_NIBBLE, DIBIT_QUERY_NIBBLE, PACKED_NIBBLE -> {
            }
            default -> throw new IllegalArgumentException(
                "Unsupported encoding "
                    + encoding
                    + " ("
                    + docBits
                    + " document bits); supported are 1 and 2 bit"
                    + " transposed codes and 4-bit packed nibbles"
            );
        }
        this.packedBytes = encoding.getDocPackedLength(dimension);
        this.queryPackedBytes = encoding.getQueryPackedLength(dimension);
        this.quantizedQuery = queryContext;

        this.queryNorm = sim == VectorSimilarityFunction.EUCLIDEAN ? (float) Math.sqrt(queryContext.correction()) : 0f;
    }

    @Override
    public BlockVectorFormat.Reader reader() {
        return reader;
    }

    @Override
    public float scoreBlock(final FixedBitSet validPos, final BlockCandidates out) {
        if (validPos == null || out == null) {
            throw new IllegalArgumentException("validPos and out must not be null");
        }
        final int[] positions = out.getPositions();
        final float[] scores = out.getScores();

        byte[] codes = reader.codes();
        float[] lower = reader.lower();
        float[] upper = reader.upper();
        float[] addCor = reader.addCor();
        int[] sum = reader.sum();

        // these are pre-computed purely to avoid computes in the loop
        final float qLowerDim = quantizedQuery.lower() * dimension;  // l_q * dim
        final float qScaleCompSum = quantizedQuery.scale() * quantizedQuery.componentSum(); // s_q * Σᵢbᵢ
        final float centroidDotProductMinusNorm = quantizedQuery.correction() - quantizedQuery.centroidNormSq(); // ⟨q,c⟩ − ‖c‖²

        int scored = 0;
        int index = validPos.nextSetBit(0);
        float maxScore = Float.NEGATIVE_INFINITY;
        while (index != DocIdSetIterator.NO_MORE_DOCS) {
            float rawDot = dotProduct(codes, index * packedBytes);    // Σᵢaᵢbᵢ
            float docScale = (upper[index] - lower[index]) * step();  // s_d = (u_d - l_d)/ 2^(bits-1);

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
            positions[scored] = index;
            scores[scored] = adc;
            scored++;

            if (adc > maxScore) {
                maxScore = adc;
            }

            // nextSetBit asserts its argument is inside the set, so the last position needs the explicit stop.
            int next = index + 1;
            index = next < validPos.length() ? validPos.nextSetBit(next) : DocIdSetIterator.NO_MORE_DOCS;
        }

        out.setSize(scored);
        return maxScore;
    }

    /**
     * {@code Σᵢaᵢbᵢ} over this vector's codes and the query's.
     *
     * <p>Every width reads the vector's codes where they lie: the narrow ones popcount 4-bit query planes against the
     * document planes, the packed nibble multiplies whole codes. All three take an offset, so no candidate's codes are
     * copied anywhere to be scored.
     */
    private float dotProduct(byte[] codes, int offset) {
        if (asymmetric) {
            return docBits == 1
                ? Int4DotProduct.bit(quantizedQuery.transposed(), codes, offset, packedBytes)
                : Int4DotProduct.dibit(quantizedQuery.transposed(), codes, offset, packedBytes);
        }
        return Int4DotProduct.nibble(quantizedQuery.transposed(), codes, offset, packedBytes);
    }

    /**
     * The ceiling over the block's corrective terms, with only the code dot product left unknown.
     *
     * <p>Of the four terms the estimate is assembled from, three are already exact here: {@code lower} and
     * {@code upper} give the document's interval, {@code sum} gives {@code Σᵢaᵢ}, and the query contributes
     * {@code Σᵢbᵢ} and its own interval. Only {@code Σᵢaᵢbᵢ} needs the codes, and it is bounded two ways —
     * all of the document's code mass landing on the query's largest component, or the reverse — so the
     * smaller of the two is used.
     *
     * <p>That is the loose part: it assumes an alignment between query and document codes that a real pair
     * almost never has, so this ceiling sits well above what the block will really score. Sound, but how often
     * it is low enough to act on is an empirical question, which is what the skip counter measures.
     */
    @Override
    public float blockCeiling(final FixedBitSet validPos) {
        if (validPos == null) {
            throw new IllegalArgumentException("validPos must not be null");
        }

        final float[] lower = reader.lower();
        final float[] upper = reader.upper();
        final float[] addCor = reader.addCor();
        final int[] sum = reader.sum();

        final float qLower = quantizedQuery.lower();
        final float qScale = quantizedQuery.scale();
        final float qComponentSum = quantizedQuery.componentSum();
        final float qLowerDim = qLower * dimension;
        final float qScaleCompSum = qScale * qComponentSum;
        final float centroidDotProductMinusNorm = quantizedQuery.correction() - quantizedQuery.centroidNormSq();

        // Largest value a single code can take on each side, which is what caps the unknown dot product.
        final int maxDocCode = (1 << docBits) - 1;
        final int maxQueryCode = (1 << quantizedQuery.queryBitsPerDimension()) - 1;

        float ceiling = Float.NEGATIVE_INFINITY;
        int index = validPos.nextSetBit(0);
        while (index != DocIdSetIterator.NO_MORE_DOCS) {
            final float docScale = (upper[index] - lower[index]) * step();
            // Σᵢaᵢbᵢ ≤ Σᵢaᵢ · max(b), and ≤ max(a) · Σᵢbᵢ; neither needs a code read.
            final float maxRawDot = Math.min((float) sum[index] * maxQueryCode, (float) maxDocCode * qComponentSum);

            final float maxScore = lower[index] * qLowerDim + qLower * docScale * sum[index] + lower[index] * qScaleCompSum + docScale
                * qScale * maxRawDot;

            ceiling = Math.max(ceiling, similarityCeiling(maxScore, addCor[index], centroidDotProductMinusNorm));
            index = index + 1 >= validPos.length() ? DocIdSetIterator.NO_MORE_DOCS : validPos.nextSetBit(index + 1);
        }
        return ceiling;
    }

    /**
     * The best similarity a residual dot of at most {@code maxScore} can turn into, in the same space
     * {@link #scoreBlock} reports. L2 inverts, so the largest dot gives the smallest distance; the geometric
     * floor still applies, since the scorer clamps to it and so can never report above it.
     */
    private float similarityCeiling(final float maxScore, final float addCor, final float centroidDotProductMinusNorm) {
        return switch (sim) {
            case EUCLIDEAN -> {
                float distance = quantizedQuery.correction() + addCor - 2f * maxScore;
                final float gap = queryNorm - (float) Math.sqrt(addCor);
                yield 1.0f / (1.0f + Math.max(distance, gap * gap));
            }
            case MAXIMUM_INNER_PRODUCT -> {
                final float dot = maxScore + addCor + centroidDotProductMinusNorm;
                yield dot >= 0 ? dot + 1 : 1f / (1f - dot);
            }
            case COSINE -> {
                final float dot = maxScore + addCor + centroidDotProductMinusNorm;
                yield Math.max((1.0f + dot) / 2.0f, 0f);
            }
            case DOT_PRODUCT -> throw new IllegalStateException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
        };
    }

    private float step() {
        return 1f / ((1 << docBits) - 1);
    }
}
