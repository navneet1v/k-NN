package org.opensearch.knn.clusterann.read.block.scalar;

import org.opensearch.knn.clusterann.read.ComputeScoreContext;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

/**
 * The scalar-quantized block a {@link org.opensearch.knn.clusterann.read.ScoreComputer} scores, plus the
 * query it is scored against. The arrays are the scan's block buffers, refilled in place for every block, so one
 * instance serves a whole scan of one cluster.
 *
 * <p>Those buffers outlive one cluster — the scan resets the same reader onto the next cluster — so the arrays alone
 * cannot say which cluster they currently hold. {@link ScalarQuantizedBlockScorer} tracks that against the reader
 * directly, which is why nothing about it appears here.
 *
 * @param codes the block's packed codes; vector {@code i} starts at {@code i * packedBytes}
 * @param lower per-vector lower interval {@code l_d}
 * @param upper per-vector upper interval {@code u_d}
 * @param addCor per-vector {@code ‖v−c‖²} for EUCLIDEAN, {@code ⟨v,c⟩} otherwise
 * @param sum per-vector {@code Σᵢaᵢ} over the quantized codes
 * @param encoding the stored code width and layout
 * @param packedBytes bytes one vector's codes occupy in {@code codes}
 * @param query the query prepared for this cluster
 */
public record SQComputeContext(byte[] codes, float[] lower, float[] upper, float[] addCor, int[] sum, ScalarEncoding encoding,
    int packedBytes, SQScanContext query) implements ComputeScoreContext {
    @Override
    public ScanContext queryContext() {
        return query;
    }
}
