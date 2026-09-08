/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.reader.Centroid;
import org.opensearch.knn.clusterann.reader.ScanParams;

/**
 * {@link BlockVectorScorer} that scores a query against scalar-quantized codes without decompressing them
 * (asymmetric distance computation): the query is quantized once at construction, against the cluster's
 * centroid, and each vector's score then falls out of a dot product over the packed codes plus corrections
 * that cost O(1) per vector.
 *
 * <p>Both sides being centered on one centroid is what ties an instance to a single cluster and a single
 * query. The codes are lossy, so every score is an estimate of the similarity the full-precision vectors
 * would give.
 */
public class ADCScalarQuantizedBlockScorer implements BlockVectorScorer {

    private final ScalarQuantizedBlockReader reader;
    private final VectorSimilarityFunction sim;
    private final int dimension;
    private final int docBits;
    private final int queryBits;
    private final int packedBytes;
    private final int queryPackedBytes;
    private final float centroidNormSq;
    private final float queryNorm;

    private final Quantized quantizedQuery;

    public ADCScalarQuantizedBlockScorer(
        final ScalarQuantizedBlockReader reader,
        final ScanParams scanParams,
        final Centroid centroid,
        final OptimizedScalarQuantizer quantizer,
        final ScalarEncoding encoding,
        final VectorSimilarityFunction sim
    ) {
        this.reader = reader;
        this.sim = sim;
        this.dimension = scanParams.query().length;
        this.docBits = encoding.getDocBitsPerDim();
        if (docBits != 1 && docBits != 2) {
            throw new IllegalArgumentException("Unsupported docBits: " + docBits);
        }
        this.queryBits = scanParams.queryBits();
        if (queryBits != ScanParams.DEFAULT_QUERY_BITS) {
            throw new IllegalArgumentException("Unsupported queryBits: " + queryBits);
        }
        this.packedBytes = encoding.getDocPackedLength(dimension);
        this.queryPackedBytes = encoding.getQueryPackedLength(dimension);
        this.centroidNormSq = centroid.normSq();

        this.quantizedQuery = quantize(scanParams.query(), centroid.vector(), quantizer);

        // Note: This is ‖q−c‖² and NOT ‖q‖²
        this.queryNorm = sim == VectorSimilarityFunction.EUCLIDEAN ? (float) Math.sqrt(quantizedQuery.correction) : 0f;
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
        final float qLowerDim = quantizedQuery.lower * dimension;  // l_q * dim
        final float qScaleCompSum = quantizedQuery.scale * quantizedQuery.componentSum; // s_q * Σᵢbᵢ
        final float centroidDotProductMinusNorm = quantizedQuery.correction - centroidNormSq; // ⟨q,c⟩ − ‖c‖²

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
                + quantizedQuery.lower * docScale * sum[index]   // l_q * s_d * Σᵢaᵢ
                + lower[index] * qScaleCompSum                   // l_d * s_q * Σᵢbᵢ
                + docScale * quantizedQuery.scale * rawDot;      // s_d * s_q * Σᵢaᵢbᵢ

            float adc = switch (sim) {
                case EUCLIDEAN -> {
                    // ‖q−v‖² = ‖q−c‖² + ‖v−c‖² − 2⟨q−c, v−c⟩
                    float distance = quantizedQuery.correction + addCor[index] - 2f * score;
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
                case DOT_PRODUCT, COSINE -> {
                    // Same dot product; both sides are unit-norm, so cosine is that dot product.
                    float dot = score + addCor[index] + centroidDotProductMinusNorm;
                    yield Math.max((1.0f + dot) / 2.0f, 0f);
                }
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

    private float dotProduct(byte[] codes, int offset) {
        switch (docBits) {
            case 1:
                return Int4DotProduct.bit(quantizedQuery.transposed, codes, offset, packedBytes);
            case 2:
                return Int4DotProduct.dibit(quantizedQuery.transposed, codes, offset, packedBytes);
            default:
                throw new IllegalArgumentException("Unsupported docBits: " + docBits);
        }
    }

    private float step() {
        return 1f / ((1 << docBits) - 1);
    }

    private record Quantized(byte[] transposed, float lower, float scale, float componentSum, float correction) {
    }

    private Quantized quantize(float[] query, float[] centroid, OptimizedScalarQuantizer quantizer) {
        int dimension = query.length;

        byte[] scratch = new byte[dimension];
        float[] queryCopy = query.clone(); // multiScalarQuantize centers in place

        OptimizedScalarQuantizer.QuantizationResult q = quantizer.multiScalarQuantize(
            queryCopy,
            new byte[][] { scratch },
            new byte[] { (byte) queryBits },
            centroid
        )[0];

        // Derived from the encoding rather than assumed, so it stays right for a dimension the layout has had to
        // round up — the same reason the doc side asks for its packed length instead of computing one.
        byte[] transposed = new byte[queryPackedBytes];
        OptimizedScalarQuantizer.transposeHalfByte(scratch, transposed);

        float lower = q.lowerInterval();
        float scale = (q.upperInterval() - lower) / ((1 << queryBits) - 1);
        return new Quantized(transposed, lower, scale, q.quantizedComponentSum(), q.additionalCorrection());
    }
}
