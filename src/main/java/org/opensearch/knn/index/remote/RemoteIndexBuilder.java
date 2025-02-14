/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.NotImplementedException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.common.StopWatch;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.Repository;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;

/**
 * This class orchestrates building vector indices. It handles uploading data to a repository, submitting a remote
 * build request, awaiting upon the build request to complete, and finally downloading the data from a repository.
 * This class is intended to be used by {@link org.opensearch.knn.index.codec.nativeindex.RemoteNativeIndexWriter}.
 * This class is created per-index, so we should not store per-segment information as class fields.
 */
@Log4j2
@ExperimentalApi
public class RemoteIndexBuilder {

    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final IndexSettings indexSettings;
    private static final String VECTOR_BLOB_FILE_EXTENSION = ".knnvec";
    private static final String DOC_ID_FILE_EXTENSION = ".knndid";
    private static final String GRAPH_FILE_EXTENSION = ".knngraph";

    /**
     * Public constructor
     *
     * @param repositoriesServiceSupplier   A supplier for {@link RepositoriesService} used for interacting with repository
     * @param indexSettings
     */
    public RemoteIndexBuilder(Supplier<RepositoriesService> repositoriesServiceSupplier, IndexSettings indexSettings) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.indexSettings = indexSettings;
    }

    /**
     * @return whether to use the remote build feature
     */
    public boolean shouldBuildIndexRemotely() {
        String vectorRepo = KNNSettings.state().getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey());
        return KNNFeatureFlags.isKNNRemoteVectorBuildEnabled()
            && indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)
            && vectorRepo != null
            && !vectorRepo.isEmpty();
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
    ) throws IOException, InterruptedException {
        StopWatch stopWatch;
        long time_in_millis;

        stopWatch = new StopWatch().start();
        writeToRepository(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, segmentWriteState);
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.debug("Repository write took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        stopWatch = new StopWatch().start();
        submitVectorBuild();
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.debug("Submit vector build took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        stopWatch = new StopWatch().start();
        awaitVectorBuild();
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.debug("Await vector build took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        stopWatch = new StopWatch().start();
        readFromRepository();
        time_in_millis = stopWatch.stop().totalTime().millis();
        log.debug("Repository read took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
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
        throw new NotImplementedException();
    }

    /**
     * Submit vector build request to remote vector build service
     *
     */
    private void submitVectorBuild() {
        throw new NotImplementedException();
    }

    /**
     * Wait on remote vector build to complete
     */
    private void awaitVectorBuild() {
        throw new NotImplementedException();
    }

    /**
     * Read constructed vector file from remote repository and write to IndexOutput
     */
    private void readFromRepository() {
        throw new NotImplementedException();
    }
}
