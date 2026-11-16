/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.opensearch.knn.clusterann.read.ComputeScoreContext;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

/**
 * The data-blind int8 block a {@link org.opensearch.knn.clusterann.read.ScoreComputer} scores, plus the
 * query it is scored against. The arrays are the scan's block buffers, allocated once by
 * {@link DataBlindInt8BlockReader} and refilled in place for every block, so one instance serves a whole scan of one
 * cluster; the query is the segment-wide context the scan shares across clusters.
 *
 * <p>Those buffers outlive one cluster — the scan resets the same reader onto the next cluster — so the arrays alone
 * cannot say which cluster they currently hold. {@link DataBlindInt8BlockScorer} tracks that against the reader
 * directly, which is why nothing about it appears here.
 *
 * @param codes     the block's codes, one unsigned byte per dimension; vector {@code i} starts at {@code i * dimension}
 * @param scale     per-vector {@code maxAbs/127}
 * @param sum       per-vector signed component sum {@code Σq}
 * @param sqNorm    per-vector {@code ‖v‖²} of the rotated vector
 * @param dimension the field's vector dimension, which is also one vector's code length
 * @param query     the query encoded once for the whole scan
 */
public record DataBlindInt8ComputeContext(byte[] codes, float[] scale, int[] sum, float[] sqNorm, int dimension,
    DataBlindInt8ScanContext query) implements ComputeScoreContext {
    @Override
    public ScanContext queryContext() {
        return query;
    }
}
