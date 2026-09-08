/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.opensearch.knn.clusterann.reader.orchestration.ScanContext;

public record SQScanContext(float[] query, int queryBitsPerDimension, float centroidNormSq, byte[] transposed, float lower, float scale,
    float componentSum, float correction) implements ScanContext.Float {
    @Override
    public float[] query() {
        return query;
    }
}
