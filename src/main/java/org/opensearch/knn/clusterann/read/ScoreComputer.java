/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

/**
 * Scores the vectors of one block against a prepared query.
 *
 * <p>The block to score is the one whose data the context holds; the caller has already fetched its corrections and
 * codes into those arrays. {@code positions[0..count)} are the in-block positions to score, in
 * ascending order. One score is written per position into {@code scores[0..count)}, and the maximum written is
 * returned; with {@code count == 0} nothing is written and {@link Float#NEGATIVE_INFINITY} is returned.
 *
 * <p>Everything specific to a quantizer travels in the context: the block's arrays, and through
 * {@link ComputeScoreContext#queryContext()} the quantized query, its corrections and the field's similarity. An
 * implementation casts the context to the compute context of the quantizer it was created for, e.g.
 * {@code SQComputeContext} for optimized scalar quantization, and reads what it needs from there. The common
 * parameters are the same for every quantizer.
 *
 * <p>An implementation is stateless: one instance is shared by every segment, field, cluster and thread that scores
 * with it. The format supplies a Java implementation per quantizer; a {@link ScoreComputerFactory} handed to the
 * format's constructor may replace it per field.
 */
public interface ScoreComputer {

    /**
     * Scores the listed positions of the current block and reports the best of them.
     *
     * @param context the block to score and the query prepared for its cluster
     * @param positions in-block positions to score, ascending; only the first {@code count} are read
     * @param count how many positions to score
     * @param scores receives one score per position, parallel to {@code positions}; at least {@code count} long
     * @return the maximum of the scores written, or {@link Float#NEGATIVE_INFINITY} when {@code count == 0}
     */
    float score(ComputeScoreContext context, int[] positions, int count, float[] scores);
}
