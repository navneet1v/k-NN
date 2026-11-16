/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

/**
 * A query prepared for one scalar-quantized cluster: the query side of what a
 * {@link org.opensearch.knn.clusterann.read.ScoreComputer} needs to score that cluster's blocks.
 *
 * @param query the full-precision query
 * @param queryBitsPerDimension the width the query was quantized at
 * @param centroidNormSq {@code ‖c‖²} of the cluster's centroid
 * @param transposed the quantized query: bit-plane transposed for a 1- or 2-bit document code, one code per
 *     dimension for 4-bit packed nibbles
 * @param lower the query's lower interval {@code l_q}
 * @param scale the query's step {@code s_q}
 * @param componentSum {@code Σᵢbᵢ} over the quantized query codes
 * @param correction {@code ‖q−c‖²} for EUCLIDEAN, {@code ⟨q,c⟩} otherwise
 * @param similarityFunction the field's similarity, which selects the ADC transform
 */
public record SQScanContext(float[] query, int queryBitsPerDimension, float centroidNormSq, byte[] transposed, float lower, float scale,
    float componentSum, float correction, VectorSimilarityFunction similarityFunction) implements ScanContext.Float {
    @Override
    public float[] query() {
        return query;
    }
}
