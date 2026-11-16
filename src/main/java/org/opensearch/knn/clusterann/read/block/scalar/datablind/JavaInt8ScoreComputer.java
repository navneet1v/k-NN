/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.opensearch.knn.clusterann.read.ComputeScoreContext;
import org.opensearch.knn.clusterann.read.ScoreComputer;
import org.opensearch.knn.quantization.DataBlindInt8Quantizer;
import org.opensearch.knn.quantization.Int8DotProduct;

/**
 * The format's own {@link ScoreComputer} for data-blind int8 codes, in plain Java.
 *
 * <p>Each vector's score is the unsigned code dot {@code Σ(qᵤ·dᵤ)} corrected to a signed dot and run through the metric
 * transform, all delegated to {@link DataBlindInt8Quantizer#score} so the scoring math lives in one place and the write
 * and read sides cannot drift. The scheme is absolute, so there is no centroid term.
 *
 * <p>Stateless; use {@link #INSTANCE}.
 */
public final class JavaInt8ScoreComputer implements ScoreComputer {

    public static final JavaInt8ScoreComputer INSTANCE = new JavaInt8ScoreComputer();

    private JavaInt8ScoreComputer() {}

    @Override
    public float score(final ComputeScoreContext context, final int[] positions, final int count, final float[] scores) {
        final DataBlindInt8ComputeContext block = (DataBlindInt8ComputeContext) context;
        final DataBlindInt8ScanContext query = block.query();
        final byte[] codes = block.codes();
        final float[] scale = block.scale();
        final int[] sum = block.sum();
        final float[] sqNorm = block.sqNorm();
        final int dimension = block.dimension();
        final byte[] queryCode = query.queryCode();
        final DataBlindInt8Quantizer.QueryTerms terms = query.terms();

        float maxScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            final int index = positions[i];
            final long unsignedDot = Int8DotProduct.unsignedDot(queryCode, codes, index * dimension, dimension);
            final float score = DataBlindInt8Quantizer.score(
                unsignedDot,
                terms,
                scale[index],
                sum[index],
                sqNorm[index],
                query.similarityFunction()
            );
            scores[i] = score;
            if (score > maxScore) {
                maxScore = score;
            }
        }
        return maxScore;
    }
}
