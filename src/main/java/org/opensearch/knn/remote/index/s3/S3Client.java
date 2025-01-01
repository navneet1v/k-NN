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
import software.amazon.awssdk.core.async.AsyncResponseTransformer;
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
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectResponse;
import software.amazon.awssdk.services.s3.model.NoSuchBucketException;
import software.amazon.awssdk.services.s3.model.UploadPartRequest;
import software.amazon.awssdk.services.s3.model.UploadPartResponse;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * A simple S3 client to upload the data to S3 bucket. The class is thread
 */
@Log4j2
public class S3Client {
    private static volatile S3Client INSTANCE;
    private static final int CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunk size
    public static String BUCKET_NAME;

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
            BUCKET_NAME = KNNSettings.getKnnS3BucketName();
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

            s3AsyncClient = builder.build();
            if (doesBucketExist(BUCKET_NAME) == false) {
                CreateBucketRequest createBucketRequest = CreateBucketRequest.builder().bucket(BUCKET_NAME).build();
                CreateBucketResponse response = s3AsyncClient.createBucket(createBucketRequest).get();
                log.debug("**********Response is: {}", response.toString());
            }
            return null;
        });

    }

    public void downloadFile(final String bucketName, final String s3Object, final OutputStream outputStream) {
        SocketAccess.doPrivilegedException(() -> {
            ExecutorService executorService = null;
            try {
                // Get the s3Object size
                HeadObjectResponse objectHead = s3AsyncClient.headObject(
                    HeadObjectRequest.builder().bucket(bucketName).key(s3Object).build()
                ).get();

                long objectSize = objectHead.contentLength();
                long partSize = 10 * 1024 * 1024L; // 10MB parts
                int numParts = (int) Math.ceil((double) objectSize / partSize);

                // Create a thread pool for parallel downloads
                executorService = Executors.newFixedThreadPool(Math.min(numParts, 10));

                // Create a blocking queue to maintain order of parts
                BlockingQueue<byte[]> partsQueue = new LinkedBlockingQueue<>();
                List<CompletableFuture<Void>> futures = new ArrayList<>();
                // Download parts in parallel
                for (int i = 0; i < numParts; i++) {
                    final int partNumber = i;
                    long rangeStart = i * partSize;
                    long rangeEnd = Math.min(rangeStart + partSize - 1, objectSize - 1);

                    CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                        try {
                            // Download the part
                            GetObjectRequest getRequest = GetObjectRequest.builder()
                                .bucket(bucketName)
                                .key(s3Object)
                                .range("bytes=" + rangeStart + "-" + rangeEnd)
                                .build();

                            byte[] partData = s3AsyncClient.getObject(getRequest, AsyncResponseTransformer.toBytes()).get().asByteArray();

                            // Add to queue in order
                            partsQueue.put(partData);
                        } catch (Exception e) {
                            throw new CompletionException(e);
                        }
                    }, executorService);
                    futures.add(future);
                }

                // Start a separate thread to write to the output stream in order
                CompletableFuture<Void> writerFuture = CompletableFuture.runAsync(() -> {
                    try {
                        for (int i = 0; i < numParts; i++) {
                            byte[] partData = partsQueue.take();
                            outputStream.write(partData);
                            outputStream.flush();
                        }
                    } catch (Exception e) {
                        throw new CompletionException(e);
                    }
                }, executorService);

                // Wait for all downloads and writing to complete
                CompletableFuture.allOf(CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])), writerFuture).join();

            } finally {
                assert executorService != null;
                executorService.shutdown();
            }
            return null;
        });
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
        } catch (ExecutionException e) {
            if (e.getCause() instanceof NoSuchBucketException) {
                return false;
            } else {
                throw e;
            }
        }
    }

}
