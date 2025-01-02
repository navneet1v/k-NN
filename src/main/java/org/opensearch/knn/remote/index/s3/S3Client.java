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
import software.amazon.awssdk.core.retry.RetryMode;
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
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
    private static final ExecutorService executorService = Executors.newFixedThreadPool(
        (int) (Runtime.getRuntime().availableProcessors() * 1.5f)
    );

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
                .httpClientBuilder(
                    NettyNioAsyncHttpClient.builder()
                        .writeTimeout(Duration.ofMinutes(5))
                        .maxConcurrency(100)
                        .readTimeout(Duration.ofMinutes(5))
                        .connectionMaxIdleTime(Duration.ofMinutes(5))        // Max idle connection time
                        .connectionTimeToLive(Duration.ofMinutes(5))        // Max connection lifetime
                        .maxPendingConnectionAcquires(10000)                 // Max queued requests
                        .connectionTimeout(Duration.ofMinutes(5)) // Connection establishment timeout
                        .tcpKeepAlive(true)
                )
                .credentialsProvider(StaticCredentialsProvider.create(credentials))
                .overrideConfiguration(
                    ClientOverrideConfiguration.builder()
                        .retryPolicy(RetryMode.ADAPTIVE)
                        .defaultProfileFile(null)
                        .defaultProfileName(null)
                        .build()
                );

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
        ProgressCallback progressCallback = SimpleS3ProgressCallback.builder()
            .operationName("Download")
            .bucket(BUCKET_NAME)
            .key(s3Object)
            .build();
        SocketAccess.doPrivilegedException(() -> {
            // Get the s3Object size
            HeadObjectResponse objectHead = s3AsyncClient.headObject(HeadObjectRequest.builder().bucket(bucketName).key(s3Object).build())
                .get();

            long objectSize = objectHead.contentLength();
            long partSize = CHUNK_SIZE;
            int numParts = (int) Math.ceil((double) objectSize / partSize);

            final Map<Integer, byte[]> downloadPartsMap = new ConcurrentHashMap<>();

            List<CompletableFuture<Void>> futures = new ArrayList<>();
            // Download parts in parallel
            for (int i = 0; i < numParts; i++) {
                final int partNumber = i;
                long rangeStart = i * partSize;
                long rangeEnd = Math.min(rangeStart + partSize - 1, objectSize - 1);

                CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                    try {
                        // we need to create this local variable to ensure that different threads don't change this value
                        int localPartNumber = partNumber;
                        // Download the part
                        GetObjectRequest getRequest = GetObjectRequest.builder()
                            .bucket(bucketName)
                            .key(s3Object)
                            .range("bytes=" + rangeStart + "-" + rangeEnd)
                            .build();
                        progressCallback.onStarted(CHUNK_SIZE);
                        byte[] partData = s3AsyncClient.getObject(getRequest, AsyncResponseTransformer.toBytes()).get().asByteArray();
                        progressCallback.onProgress(CHUNK_SIZE);
                        // add the part in the map, so that we can build the whole file later.
                        // TODO: We need to find a way in which we can reduce the heap consumption here
                        downloadPartsMap.put(localPartNumber, partData);
                    } catch (Exception e) {
                        throw new CompletionException(e);
                    }
                }, executorService);
                futures.add(future);
            }

            int i = 0;
            long totalBytes = 0;
            for (Future future : futures) {
                // wait for download to be completed and put into the map
                future.get();
                byte[] partData = downloadPartsMap.get(i);
                totalBytes += partData.length;
                progressCallback.onComplete(totalBytes);
                i++;
                outputStream.write(partData);
                outputStream.flush();
            }
            return null;
        });
    }

    public long uploadWithProgress(final InputStream inputStream, final String key) {
        ProgressCallback progressCallback = SimpleS3ProgressCallback.builder().operationName("Upload").bucket(BUCKET_NAME).key(key).build();
        long totalBytesUploaded = 0;
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
                    partNumber++;
                    progressCallback.onStarted(baos.size());
                    baos.reset();
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
                log.info("Last chunk upload started to s3.");
                progressCallback.onStarted(baos.size());
            }

            int responsePartNumbers = 1;
            for (CompletableFuture<UploadPartResponse> future : completableFutureList) {
                UploadPartResponse response = future.get();
                CompletedPart part = CompletedPart.builder().partNumber(responsePartNumbers).eTag((response).eTag()).build();
                completedParts.add(part);
                // TODO: Fix this, add the way to find how many bytes are uploaded
                totalBytesUploaded += CHUNK_SIZE;
                progressCallback.onProgress(totalBytesUploaded);
                responsePartNumbers++;
            }
            completedParts.sort(Comparator.comparingInt(CompletedPart::partNumber));
            // List<Integer> myList = completedParts.stream().map(CompletedPart::partNumber).collect(Collectors.toList());
            // log.info("********** CompletedParts : {} **************", myList);
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
