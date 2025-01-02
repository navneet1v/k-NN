/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.s3;

import lombok.Builder;
import lombok.Value;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Value
@Builder
public class SimpleS3ProgressCallback implements ProgressCallback {

    String key;
    String bucket;
    String operationName;

    @Override
    public void onProgress(long bytes) {
        log.info("{}ed {} bytes for key {}, bucket: {}", operationName, bytes, key, bucket);
    }

    @Override
    public void onStarted(long bytes) {
        log.info("{} started for {} bytes for key {}, bucket: {}", operationName, bytes, key, bucket);
    }

    @Override
    public void onComplete(long bytes) {
        log.info("{}ed {} bytes in total for key {}, bucket: {}", operationName, bytes, key, bucket);
    }
}
