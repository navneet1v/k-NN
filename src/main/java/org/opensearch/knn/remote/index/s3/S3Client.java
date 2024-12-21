/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.s3;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.StringUtils;
import org.opensearch.common.SuppressForbidden;
import org.opensearch.knn.index.KNNSettings;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentials;
import software.amazon.awssdk.auth.credentials.AwsSessionCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.async.AsyncRequestBody;
import software.amazon.awssdk.core.client.config.ClientOverrideConfiguration;
import software.amazon.awssdk.http.nio.netty.NettyNioAsyncHttpClient;
import software.amazon.awssdk.profiles.ProfileFileSystemSetting;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.model.CompleteMultipartUploadRequest;
import software.amazon.awssdk.services.s3.model.CompleteMultipartUploadResponse;
import software.amazon.awssdk.services.s3.model.CompletedMultipartUpload;
import software.amazon.awssdk.services.s3.model.CompletedPart;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.CreateBucketResponse;
import software.amazon.awssdk.services.s3.model.CreateMultipartUploadRequest;
import software.amazon.awssdk.services.s3.model.CreateMultipartUploadResponse;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.NoSuchBucketException;
import software.amazon.awssdk.services.s3.model.S3Exception;
import software.amazon.awssdk.services.s3.model.UploadPartRequest;
import software.amazon.awssdk.services.s3.model.UploadPartResponse;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * A simple S3 client to upload the data to S3 bucket. The class is thread
 */
@Log4j2
public class S3Client {
    private static volatile S3Client INSTANCE;
    private static final int CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunk size
    public static final String BUCKET_NAME = "remote-knn-index-build-navneet";

    private static final Region REGION = Region.US_WEST_2;
    private static software.amazon.awssdk.services.s3.S3AsyncClient s3AsyncClient;

    public static S3Client getInstance() throws IOException {
        S3Client result = INSTANCE;
        if (result == null) {
            synchronized (S3Client.class) {
                result = INSTANCE;
                if (result == null) {
                    INSTANCE = result = new S3Client();
                }
            }
        }
        return result;
    }

    @SuppressForbidden(reason = "Need to provide this override to v2 SDK so that path does not default to home path")
    private S3Client() {
        SocketAccess.doPrivilegedException(() -> {
            if (ProfileFileSystemSetting.AWS_SHARED_CREDENTIALS_FILE.getStringValue().isEmpty()) {
                System.setProperty(
                    ProfileFileSystemSetting.AWS_SHARED_CREDENTIALS_FILE.property(),
                    System.getProperty("opensearch.path.conf")
                );
            }
            if (ProfileFileSystemSetting.AWS_CONFIG_FILE.getStringValue().isEmpty()) {
                System.setProperty(ProfileFileSystemSetting.AWS_CONFIG_FILE.property(), System.getProperty("opensearch.path.conf"));
            }
            final String accessKey = KNNSettings.getKnnS3AccessKey();
            final String secretKey = KNNSettings.getKnnS3SecretKey();
            final String sessionToken = KNNSettings.getKnnS3Token();
            final AwsCredentials credentials;
            if (StringUtils.isEmpty(sessionToken)) {
                // create basic credentials
                credentials = AwsBasicCredentials.create(accessKey, secretKey);
            } else {
                // create a session credentials
                // these credentials should be updated.
                credentials = AwsSessionCredentials.create(accessKey, secretKey, sessionToken);
            }
            log.debug("**********Credentials are: {}", credentials.toString());

            software.amazon.awssdk.services.s3.S3AsyncClientBuilder builder = software.amazon.awssdk.services.s3.S3AsyncClient.builder()
                .region(REGION)
                .httpClientBuilder(NettyNioAsyncHttpClient.builder())
                .credentialsProvider(StaticCredentialsProvider.create(credentials))
                .overrideConfiguration(ClientOverrideConfiguration.builder().defaultProfileFile(null).defaultProfileName(null).build());

            // SpecialPermission.check();

            s3AsyncClient = builder.build();
            if (doesBucketExist(BUCKET_NAME) == false) {
                CreateBucketRequest createBucketRequest = CreateBucketRequest.builder().bucket(BUCKET_NAME).build();
                CreateBucketResponse response = s3AsyncClient.createBucket(createBucketRequest).get();
                log.debug("**********Response is: {}", response.toString());
            }
            return null;
        });

    }

    public void downloadFile() {
        // TODO: Implement this later
    }

    public long uploadWithProgress(final InputStream inputStream, final String key) {
        ProgressCallback progressCallback = SimpleS3ProgressCallback.builder().bucket(BUCKET_NAME).key(key).build();
        long totalBytesUploaded = 0;
        long totalBytesStarted = 0;
        try {
            CreateMultipartUploadRequest createMultipartUploadRequest = CreateMultipartUploadRequest.builder()
                .bucket(BUCKET_NAME)
                .key(key)
                .build();

            CreateMultipartUploadResponse multipartUpload = SocketAccess.doPrivilegedException(
                () -> s3AsyncClient.createMultipartUpload(createMultipartUploadRequest).get()
            );

            String uploadId = multipartUpload.uploadId();
            List<CompletedPart> completedParts = new ArrayList<>();

            byte[] buffer = new byte[CHUNK_SIZE];
            int partNumber = 1;
            int bytesRead;
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            List<CompletableFuture<UploadPartResponse>> completableFutureList = new ArrayList<>();
            // Read and upload parts
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                baos.write(buffer, 0, bytesRead);

                // Upload when we have at least 5MB or reached end of stream
                if (baos.size() >= CHUNK_SIZE) {
                    byte[] partData = baos.toByteArray();

                    // Upload the part
                    UploadPartRequest uploadPartRequest = UploadPartRequest.builder()
                        .bucket(BUCKET_NAME)
                        .key(key)
                        .uploadId(uploadId)
                        .partNumber(partNumber)
                        .build();

                    CompletableFuture<UploadPartResponse> uploadPartResponse = SocketAccess.doPrivilegedException(
                        () -> s3AsyncClient.uploadPart(uploadPartRequest, AsyncRequestBody.fromBytes(partData))
                    );
                    completableFutureList.add(uploadPartResponse);
                    // CompletedPart part = CompletedPart.builder()
                    // .partNumber(partNumber)
                    // .eTag((uploadPartResponse).eTag())
                    // .build();

                    // completedParts.add(part);
                    partNumber++;
                    baos.reset();
                    totalBytesStarted += bytesRead;
                    progressCallback.onUploadStarted(totalBytesStarted);
                    totalBytesUploaded += bytesRead;
                }
            }

            if (baos.size() > 0) {
                byte[] partData = baos.toByteArray();

                // Upload the part
                UploadPartRequest uploadPartRequest = UploadPartRequest.builder()
                    .bucket(BUCKET_NAME)
                    .key(key)
                    .uploadId(uploadId)
                    .partNumber(partNumber)
                    .build();

                CompletableFuture<UploadPartResponse> uploadPartResponse = SocketAccess.doPrivileged(
                    () -> s3AsyncClient.uploadPart(uploadPartRequest, AsyncRequestBody.fromBytes(partData))
                );
                completableFutureList.add(uploadPartResponse);
                totalBytesUploaded += partData.length;
            }

            for (CompletableFuture<UploadPartResponse> future : completableFutureList) {
                UploadPartResponse response = future.get();
                CompletedPart part = CompletedPart.builder().partNumber(partNumber).eTag((response).eTag()).build();
                completedParts.add(part);
                // TODO: Fix this, add the way to find how many bytes are uploaded
                // totalBytesUploaded += partData.length;
                // progressCallback.onProgress(totalBytesUploaded);
            }

            // Complete the multipart upload
            CompletedMultipartUpload completedMultipartUpload = CompletedMultipartUpload.builder().parts(completedParts).build();

            CompleteMultipartUploadRequest completeRequest = CompleteMultipartUploadRequest.builder()
                .bucket(BUCKET_NAME)
                .key(key)
                .uploadId(uploadId)
                .multipartUpload(completedMultipartUpload)
                .build();

            CompleteMultipartUploadResponse response = SocketAccess.doPrivilegedException(
                () -> s3AsyncClient.completeMultipartUpload(completeRequest).get()
            );
            log.debug("********** CompleteMultipartUploadResponse : {} **************", response);
            progressCallback.onComplete(totalBytesUploaded);
            return totalBytesUploaded;
        } catch (Exception e) {
            // Handle exceptions appropriately
            throw new RuntimeException("Failed to upload file", e);
        }
    }

    public boolean doesBucketExist(String bucketName) throws ExecutionException, InterruptedException {
        try {
            s3AsyncClient.headBucket(HeadBucketRequest.builder().bucket(bucketName).build()).get();
            return true;
        } catch (NoSuchBucketException e) {
            return false;
        } catch (S3Exception e) {
            if (e.statusCode() == 404) {
                return false;
            }
            throw e;
        }
    }

}
