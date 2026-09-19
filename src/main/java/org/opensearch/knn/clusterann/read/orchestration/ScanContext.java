/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

public interface ScanContext {

    interface Float extends ScanContext {
        float[] query();
    }
}
