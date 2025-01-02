/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.s3;

public interface ProgressCallback {
    void onProgress(long bytes);

    void onComplete(long bytes);

    void onStarted(long bytes);
}
