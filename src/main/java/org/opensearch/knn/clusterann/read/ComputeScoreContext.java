/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

/**
 * What a {@link ScoreComputer} scores one block from: the block's data and the query it is scored against. Each
 * quantizer has its own implementation holding that quantizer's block arrays, e.g. {@code SQComputeContext}; the
 * query side is reached through {@link #queryContext()}.
 */
public interface ComputeScoreContext {

    /** The query prepared for the cluster the block belongs to. */
    ScanContext queryContext();
}
