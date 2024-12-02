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

    @Override
    public void onProgress(long bytesUploaded) {
        log.info("Uploaded {} bytes for key {}, bucket: {}", bytesUploaded, key, bucket);
    }

    @Override
    public void onUploadStarted(long bytesUploaded) {
        log.info("Upload started for {} bytes for key {}, bucket: {}", bytesUploaded, key, bucket);
    }

    @Override
    public void onComplete(long totalBytesUploaded) {
        log.info("Uploaded {} bytes in total for key {}, bucket: {}", totalBytesUploaded, key, bucket);
    }
}
