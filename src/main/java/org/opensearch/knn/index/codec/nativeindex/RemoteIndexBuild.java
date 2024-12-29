/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.VectorValuesInputStream;
import org.opensearch.knn.remote.index.client.IndexBuildServiceClient;
import org.opensearch.knn.remote.index.model.CreateIndexRequest;
import org.opensearch.knn.remote.index.model.CreateIndexResponse;
import org.opensearch.knn.remote.index.model.GetJobRequest;
import org.opensearch.knn.remote.index.model.GetJobResponse;
import org.opensearch.knn.remote.index.s3.S3Client;

import java.io.InputStream;
import java.util.UUID;
import java.util.function.Supplier;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;

@Log4j2
public class RemoteIndexBuild {
    private static final String COMPLETED_STATUS = "completed";
    private static final String FAILED_STATUS = "failed";
    private final S3Client s3Client;
    private final IndexBuildServiceClient indexBuildServiceClient;
    private final String indexUUID;
    private final SegmentWriteState segmentWriteState;

    public RemoteIndexBuild(final String indexUUID, final SegmentWriteState segmentWriteState) {
        this.indexUUID = indexUUID;
        try {
            this.s3Client = S3Client.getInstance();
            this.indexBuildServiceClient = IndexBuildServiceClient.getInstance();
            this.segmentWriteState = segmentWriteState;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void buildIndexRemotely(FieldInfo fieldInfo, Supplier<KNNVectorValues<?>> knnVectorValuesSupplier, int totalLiveDocs) {
        try {
            // First upload all the vectors to S3
            String objectKey = uploadToS3(fieldInfo, knnVectorValuesSupplier);
            log.info("Creating the IndexRequest...");
            CreateIndexRequest createIndexRequest = buildCreateIndexRequest(fieldInfo, totalLiveDocs, objectKey);
            log.info("Submitting request to remote indexbuildService");
            // call the CreateIndex api to kick off the index creation
            CreateIndexResponse response = indexBuildServiceClient.createIndex(createIndexRequest);
            log.info("Request completed with response : {}", response);
            // wait for create index job to be completed
            final GetJobResponse getJobResponse = isIndexBuildCompletedWithoutErrors(response);
            downloadGraphFileFromS3(getJobResponse, fieldInfo);

        } catch (Exception e) {
            log.error("Failed to create the index remotely: {}", fieldInfo, e);
        }
    }

    private GetJobResponse isIndexBuildCompletedWithoutErrors(final CreateIndexResponse response) {
        try {
            GetJobRequest getJobRequest = GetJobRequest.builder().jobId(response.getIndexCreationRequestId()).build();
            log.info("Waiting for index build to be completed: {}", response.getIndexCreationRequestId());
            GetJobResponse getJobResponse = indexBuildServiceClient.getJob(getJobRequest);
            if (COMPLETED_STATUS.equals(getJobResponse.getStatus()) || FAILED_STATUS.equals(getJobResponse.getStatus())) {
                log.info("Remote Index build completed with status: {}", getJobResponse.getStatus());
                return getJobResponse;
            } else {
                log.info("Index build is still in progress. Current status: {}", getJobResponse.getStatus());
                // I am using the same merge thread to ensure that we are not completing merge early.
                Thread.sleep(KNNSettings.getIndexBuildStatusWaitTime());
                return isIndexBuildCompletedWithoutErrors(response);
            }
        } catch (Exception e) {
            log.error("Failed to wait for remote index build to be completed: {}", response.getIndexCreationRequestId(), e);
        }
        return GetJobResponse.builder().status("errored").build();
    }

    private String uploadToS3(final FieldInfo fieldInfo, final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier) {
        // s3 uploader
        String s3Key = createObjectKey(fieldInfo);
        try (InputStream vectorInputStream = new VectorValuesInputStream((KNNFloatVectorValues) knnVectorValuesSupplier.get())) {
            StopWatch stopWatch = new StopWatch().start();
            // Lets upload data to s3.
            long totalBytesUploaded = s3Client.uploadWithProgress(vectorInputStream, s3Key);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            log.info(
                "Time taken to upload vector for segment : {}, field: {}, totalBytes: {}, dimension: {} is : {}ms",
                segmentWriteState.segmentInfo.name,
                fieldInfo.getName(),
                totalBytesUploaded,
                fieldInfo.getVectorDimension(),
                time_in_millis
            );
        } catch (Exception e) {
            // logging here as this is in internal error
            log.error("Error while uploading data to s3.", e);
        }
        return s3Key;
    }

    private void downloadGraphFileFromS3(GetJobResponse getJobResponse, FieldInfo fieldInfo) {
        try {
            String graphFileLocation = getJobResponse.getResult().getGraphFileLocation();
            String bucketName = getJobResponse.getResult().getBucketName();
            final String engineFileName = buildEngineFileName(
                segmentWriteState.segmentInfo.name,
                KNNEngine.FAISS.getVersion(),
                fieldInfo.name,
                KNNEngine.FAISS.getExtension()
            );
            log.info("Downloading and writing the index file for field: {}", fieldInfo.name);
            try (IndexOutput indexOutput = segmentWriteState.directory.createOutput(engineFileName, segmentWriteState.context)) {
                final NativeIndexOutputStream outputStream = NativeIndexOutputStream.builder().indexOutput(indexOutput).build();
                s3Client.downloadFile(bucketName, graphFileLocation, outputStream);
                CodecUtil.writeFooter(indexOutput);
            }
            log.info("Remote Index file downloaded and written successfully for field: {}", fieldInfo.name);
        } catch (Exception e) {
            log.error("Error while downloading graph file from s3.", e);
        }
    }

    private CreateIndexRequest buildCreateIndexRequest(final FieldInfo fieldInfo, int totalLiveDocs, String objectKey) {
        int dimension = fieldInfo.getVectorDimension();
        return CreateIndexRequest.builder()
            .bucketName(S3Client.BUCKET_NAME)
            .objectLocation(objectKey)
            .dimensions(dimension)
            .numberOfVectors(totalLiveDocs)
            .build();
    }

    private String createObjectKey(FieldInfo fieldInfo) {
        String fieldName = fieldInfo.getName();
        // shard information will also be needed to ensure that we can correct paths.
        // We need to see what we need to replace this UUID with.
        return indexUUID + "_" + UUID.randomUUID() + segmentWriteState.segmentInfo.name + "_" + fieldName + ".s3vec";
    }
}
