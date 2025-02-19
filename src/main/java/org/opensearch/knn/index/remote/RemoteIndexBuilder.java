/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.action.LatchedActionListener;
import org.opensearch.common.CheckedTriFunction;
import org.opensearch.common.StopWatch;
import org.opensearch.common.StreamContext;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.stream.write.WriteContext;
import org.opensearch.common.blobstore.stream.write.WritePriority;
import org.opensearch.common.io.InputStreamContainer;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.remote.client.IndexBuildServiceClient;
import org.opensearch.knn.index.remote.model.CreateIndexRequest;
import org.opensearch.knn.index.remote.model.CreateIndexResponse;
import org.opensearch.knn.index.remote.model.GetJobRequest;
import org.opensearch.knn.index.remote.model.GetJobResponse;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.Repository;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.concurrent.CountDownLatch;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * This class orchestrates building vector indices. It handles uploading data to a repository, submitting a remote
 * build request, awaiting upon the build request to complete, and finally downloading the data from a repository.
 * This class is intended to be used by {@link org.opensearch.knn.index.codec.nativeindex.RemoteNativeIndexWriter}.
 * This class is created per-index, so we should not store per-segment information as class fields.
 */
@Log4j2
@ExperimentalApi
public class RemoteIndexBuilder {
    private static final String COMPLETED_STATUS = "completed";
    private static final String FAILED_STATUS = "failed";
    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final IndexSettings indexSettings;
    private static final String VECTOR_BLOB_FILE_EXTENSION = ".knnvec";
    private static final String DOC_ID_FILE_EXTENSION = ".knndid";
    private static final String GRAPH_FILE_EXTENSION = ".faiss";
    private IndexBuildServiceClient indexBuildServiceClient;

    /**
     * Public constructor
     *
     * @param repositoriesServiceSupplier   A supplier for {@link RepositoriesService} used for interacting with repository
     * @param indexSettings
     */
    public RemoteIndexBuilder(Supplier<RepositoriesService> repositoriesServiceSupplier, IndexSettings indexSettings) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.indexSettings = indexSettings;
        try {
            this.indexBuildServiceClient = IndexBuildServiceClient.getInstance();
        } catch (IOException e) {
            log.error("Error while creating the IndexBuildService client. ", e);
        }
    }

    /**
     * @return whether to use the remote build feature
     */
    public boolean shouldBuildIndexRemotely(KNNVectorValues<?> knnVectorValues) throws IOException {
        String vectorRepo = KNNSettings.state().getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey());
        initializeVectorValues(knnVectorValues);
        int liveDocs = 0;
        while (knnVectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }

        if (liveDocs < KNNSettings.getRemoteIndexBuildThreshold()) {
            log.debug(
                "Total docs {} is less than threshold {}, skipping remote build",
                liveDocs,
                KNNSettings.getRemoteIndexBuildThreshold()
            );
            return false;
        }

        return KNNFeatureFlags.isKNNRemoteVectorBuildEnabled()
            && indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)
            && vectorRepo != null
            && !vectorRepo.isEmpty()
            // ensure that index build service client is not null.
            && indexBuildServiceClient != null;
    }

    /**
     * Gets the KNN repository container from the repository service.
     *
     * @return {@link RepositoriesService}
     * @throws RepositoryMissingException if repository is not registered or if {@link KNN_REMOTE_VECTOR_REPO_SETTING} is not set
     */
    private BlobStoreRepository getRepository() throws RepositoryMissingException {
        RepositoriesService repositoriesService = repositoriesServiceSupplier.get();
        assert repositoriesService != null;
        String vectorRepo = KNNSettings.state().getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey());
        if (vectorRepo == null || vectorRepo.isEmpty()) {
            throw new RepositoryMissingException("Vector repository " + KNN_REMOTE_VECTOR_REPO_SETTING.getKey() + " is not registered");
        }
        final Repository repository = repositoriesService.repository(vectorRepo);
        assert repository instanceof BlobStoreRepository : "Repository should be instance of BlobStoreRepository";
        return (BlobStoreRepository) repository;
    }

    /**
     * 1. upload files, 2. trigger build and wait for completion, 3. download the graph, 4. write to indexoutput
     *
     * @param fieldInfo
     * @param knnVectorValuesSupplier
     * @param totalLiveDocs
     */
    public void buildIndexRemotely(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs,
        SegmentWriteState segmentWriteState
    ) throws IOException, InterruptedException, URISyntaxException {
        StopWatch stopWatch;
        long time_in_millis;

        stopWatch = new StopWatch().start();
        writeToRepository(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, segmentWriteState);
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.info("Repository write took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        stopWatch = new StopWatch().start();
        String blobName = indexSettings.getUUID() + "_" + fieldInfo.getName() + "_" + segmentWriteState.segmentInfo.name;
        String vectorsFilePath = getRepository().basePath().buildAsString() + blobName;
        CreateIndexResponse createIndexResponse = submitVectorBuild(
            fieldInfo,
            totalLiveDocs,
            vectorsFilePath,
            getRepository().getMetadata().settings().get("bucket")
        );
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.info("Submit vector build took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        stopWatch = new StopWatch().start();
        boolean isIndexBuildCompleted = isVectorIndexBuildCompleted(createIndexResponse);
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.info("Await vector build took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
        if (isIndexBuildCompleted == false) {
            log.error("Job status with jobId : {} errored out. Returning.", createIndexResponse.getIndexCreationRequestId());
            throw new RuntimeException("Index Build job status was not completed.");
        }

        stopWatch = new StopWatch().start();
        readFromRepository(fieldInfo, segmentWriteState);
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.info("Repository read took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    /**
     * Write relevant vector data to repository
     *
     * @param fieldInfo
     * @param knnVectorValuesSupplier
     * @param totalLiveDocs
     * @param segmentWriteState
     * @throws IOException
     */
    private void writeToRepository(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs,
        SegmentWriteState segmentWriteState
    ) throws IOException, InterruptedException {
        String indexUUID = indexSettings.getUUID();
        // TODO: Blob Path
        BlobContainer blobContainer = getRepository().blobStore().blobContainer(getRepository().basePath());
        String blobName = indexUUID + "_" + fieldInfo.getName() + "_" + segmentWriteState.segmentInfo.name;

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
        initializeVectorValues(knnVectorValues);
        long vectorBlobLength = (long) knnVectorValues.bytesPerVector() * totalLiveDocs;

        // First upload vector blob
        if (blobContainer instanceof AsyncMultiStreamBlobContainer) {
            log.info("Repository {} Supports Parallel Blob Upload", getRepository());
            StopWatch stopWatchVector = new StopWatch().start();

            WriteContext writeContext = new WriteContext.Builder().fileName(blobName + VECTOR_BLOB_FILE_EXTENSION)
                .streamContextSupplier((partSize) -> getStreamContext(partSize, vectorBlobLength, knnVectorValuesSupplier, vectorDataType))
                .fileSize(vectorBlobLength)
                .failIfAlreadyExists(true)
                .writePriority(WritePriority.NORMAL)
                // TODO: finalizer, checksum, integrity check all in final PR
                .uploadFinalizer((bool) -> {})
                .doRemoteDataIntegrityCheck(false)
                .expectedChecksum(null)
                .metadata(null)
                .build();

            final CountDownLatch latch = new CountDownLatch(1);
            ((AsyncMultiStreamBlobContainer) blobContainer).asyncBlobUpload(
                writeContext,
                new LatchedActionListener<>(new ActionListener<>() {
                    @Override
                    public void onResponse(Void unused) {
                        log.info("Parallel Upload Completed");
                        long time_in_millis = stopWatchVector.stop().totalTime().millis();
                        log.info(
                            "Parallel vector blob upload took {} ms for vector field [{}], uploaded {} bytes",
                            time_in_millis,
                            fieldInfo.getName(),
                            vectorBlobLength
                        );
                    }

                    @Override
                    public void onFailure(Exception e) {
                        log.info("Parallel Upload Failed", e);
                    }
                }, latch)
            );
            StopWatch stopWatch = new StopWatch().start();
            // Then upload doc id blob
            InputStream docStream = new BufferedInputStream(new DocIdInputStream(knnVectorValuesSupplier.get()));
            long docBlobLength = totalLiveDocs * 4L;
            blobContainer.writeBlob(blobName + DOC_ID_FILE_EXTENSION, docStream, docBlobLength, false);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            log.info(
                "Sequential docid blob upload took {} ms for vector field [{}], uploaded {} bytes",
                time_in_millis,
                fieldInfo.getName(),
                docBlobLength
            );
            latch.await();
        } else {
            StopWatch stopWatch = new StopWatch().start();
            log.info("Repository {} Does Not Support Parallel Blob Upload", getRepository());
            // Write Vectors
            InputStream vectorStream = new BufferedInputStream(new VectorValuesInputStream(knnVectorValuesSupplier.get(), vectorDataType));
            log.info("Writing {} bytes for {} docs to {}", vectorBlobLength, totalLiveDocs, blobName + VECTOR_BLOB_FILE_EXTENSION);
            blobContainer.writeBlob(blobName + VECTOR_BLOB_FILE_EXTENSION, vectorStream, vectorBlobLength, false);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            log.info(
                "Sequential vector blob upload took {} ms for vector field [{}], uploaded {} bytes",
                time_in_millis,
                fieldInfo.getName(),
                vectorBlobLength
            );

            stopWatch = new StopWatch().start();
            // Then upload doc id blob
            InputStream docStream = new BufferedInputStream(new DocIdInputStream(knnVectorValuesSupplier.get()));
            long docBlobLength = totalLiveDocs * 4L;
            blobContainer.writeBlob(blobName + DOC_ID_FILE_EXTENSION, docStream, docBlobLength, false);
            time_in_millis = stopWatch.stop().totalTime().millis();
            log.info(
                "Sequential docid blob upload took {} ms for vector field [{}], uploaded {} bytes",
                time_in_millis,
                fieldInfo.getName(),
                docBlobLength
            );
        }
    }

    private CheckedTriFunction<Integer, Long, Long, InputStreamContainer, IOException> getTransferPartStreamSupplier(
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        VectorDataType vectorDataType
    ) {
        return ((partNo, size, position) -> {
            log.info("Creating InputStream for partNo: {}, size: {}, position: {}", partNo, size, position);
            VectorValuesInputStream vectorValuesInputStream = new VectorValuesInputStream(
                knnVectorValuesSupplier.get(),
                vectorDataType,
                position,
                size
            );
            return new InputStreamContainer(vectorValuesInputStream, size, position);
        });
    }

    private StreamContext getStreamContext(
        long partSize,
        long vectorBlobLength,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        VectorDataType vectorDataType
    ) {
        long lastPartSize = (vectorBlobLength % partSize) != 0 ? vectorBlobLength % partSize : partSize;
        int numberOfParts = (int) ((vectorBlobLength % partSize) == 0 ? vectorBlobLength / partSize : (vectorBlobLength / partSize) + 1);
        log.info(
            "Performing parallel upload with partSize: {}, numberOfParts: {}, lastPartSize: {}",
            partSize,
            numberOfParts,
            lastPartSize
        );
        return new StreamContext(
            getTransferPartStreamSupplier(knnVectorValuesSupplier, vectorDataType),
            partSize,
            lastPartSize,
            numberOfParts
        );
    }

    /**
     * Submit vector build request to remote vector build service
     *
     */
    private CreateIndexResponse submitVectorBuild(
        final FieldInfo fieldInfo,
        int totalLiveDocs,
        final String vectorsFilePath,
        final String bucketName
    ) throws IOException, URISyntaxException {
        return indexBuildServiceClient.createIndex(buildCreateIndexRequest(fieldInfo, totalLiveDocs, vectorsFilePath, bucketName));
    }

    private CreateIndexRequest buildCreateIndexRequest(
        final FieldInfo fieldInfo,
        int totalLiveDocs,
        final String vectorsFilePath,
        final String bucketName
    ) {
        int dimension = fieldInfo.getVectorDimension();
        String spaceType = fieldInfo.attributes().getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue());
        return CreateIndexRequest.builder()
            .bucketName(bucketName)
            .objectLocation(vectorsFilePath)
            .dimensions(dimension)
            .spaceType(spaceType)
            .numberOfVectors(totalLiveDocs)
            .build();
    }

    /**
     * Wait on remote vector build to complete
     */
    private boolean isVectorIndexBuildCompleted(CreateIndexResponse createIndexResponse) {
        final GetJobResponse response = isIndexBuildCompletedWithoutErrors(createIndexResponse);
        return COMPLETED_STATUS.equals(response.getStatus());
    }

    private GetJobResponse isIndexBuildCompletedWithoutErrors(final CreateIndexResponse response) {
        while (true) {
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
                }
            } catch (Exception e) {
                log.error("Failed to wait for remote index build to be completed: {}", response.getIndexCreationRequestId(), e);
                break;
            }
        }
        return GetJobResponse.builder().status("errored").build();
    }

    /**
     * Read constructed vector file from remote repository and write to IndexOutput
     */
    private void readFromRepository(FieldInfo fieldInfo, SegmentWriteState segmentWriteState) throws IOException {
        BlobContainer blobContainer = getRepository().blobStore().blobContainer(getRepository().basePath());
        String indexUUID = indexSettings.getUUID();
        String blobName = indexUUID + "_" + fieldInfo.getName() + "_" + segmentWriteState.segmentInfo.name;
        InputStream graphStream = blobContainer.readBlob(blobName + GRAPH_FILE_EXTENSION);
        final String engineFileName = buildEngineFileName(
            segmentWriteState.segmentInfo.name,
            KNNEngine.FAISS.getVersion(),
            fieldInfo.name,
            KNNEngine.FAISS.getExtension()
        );
        IndexOutput indexOutput = segmentWriteState.directory.createOutput(engineFileName, segmentWriteState.context);
        int bytes = graphStream.available();
        // TODO: Write to indexOutput in a buffered manner
        indexOutput.writeBytes(graphStream.readAllBytes(), bytes);
        CodecUtil.writeFooter(indexOutput);
    }
}
