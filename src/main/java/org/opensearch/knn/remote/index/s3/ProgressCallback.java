/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.s3;

public interface ProgressCallback {
    void onProgress(long bytesUploaded);

    void onComplete(long totalBytesUploaded);

    void onUploadStarted(long bytesUploaded);
}
