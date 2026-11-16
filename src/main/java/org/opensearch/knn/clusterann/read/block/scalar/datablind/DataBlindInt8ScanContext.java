/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;
import org.opensearch.knn.quantization.DataBlindInt8Quantizer;

/**
 * The query state a data-blind int8 scan carries: the original query (for the orchestrator), the query encoded
 * once — the same {@code maxAbs/127} absolute encoding the stored codes use — into unsigned byte codes, and the
 * query-invariant score terms precomputed once ({@link DataBlindInt8Quantizer.QueryTerms}) so the per-candidate
 * score is a single multiply-add plus the metric transform.
 *
 * <p>Unlike {@link org.opensearch.knn.clusterann.read.block.scalar.SQScanContext} there is
 * <em>no centroid</em>: the scheme is absolute, so the query is rotated
 * and quantized a single time for the whole segment, independent of which cluster it lands in.
 *
 * @param query     the original (already rotation-prepared) query, for the orchestrator
 * @param queryCode the query's unsigned int8 codes, one byte per dimension
 * @param terms     the query-invariant score terms (scale, {@code ‖q‖²}, offset residual)
 * @param similarityFunction the field's similarity, which selects the score transform
 */
public record DataBlindInt8ScanContext(float[] query, byte[] queryCode, DataBlindInt8Quantizer.QueryTerms terms,
    VectorSimilarityFunction similarityFunction) implements ScanContext.Float {

    @Override
    public float[] query() {
        return query;
    }

    /**
     * {@code true}: the scheme is absolute, so this encoding of the query is what every cluster of the scan is
     * scored against, whichever one the query routed to.
     */
    @Override
    public boolean isQueryScoped() {
        return true;
    }
}
