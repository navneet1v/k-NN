/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.index.remote.RemoteIndexBuilder;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.function.Supplier;

/**
 * Writes KNN Index for a field in a segment. This is intended to be used for native engines. This class uses a remote index build service for building segments.
 * See {@link LocalNativeIndexWriter} for local vector index build path.
 */
@Log4j2
@ExperimentalApi
public class RemoteNativeIndexWriter implements NativeIndexWriter {

    private final FieldInfo fieldInfo;
    private final SegmentWriteState segmentWriteState;
    private final NativeIndexWriter fallbackWriter;
    private final RemoteIndexBuilder remoteIndexBuilder;
    private final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier;

    public RemoteNativeIndexWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState segmentWriteState,
        final NativeIndexWriter fallbackWriter,
        final RemoteIndexBuilder remoteIndexBuilder,
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier
    ) {
        this.fieldInfo = fieldInfo;
        this.segmentWriteState = segmentWriteState;
        this.fallbackWriter = fallbackWriter;
        this.remoteIndexBuilder = remoteIndexBuilder;
        this.knnVectorValuesSupplier = knnVectorValuesSupplier;
    }

    @Override
    public void flushIndex(KNNVectorValues<?> knnVectorValues, int totalLiveDocs) throws IOException {
        try {
            remoteIndexBuilder.buildIndexRemotely(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, segmentWriteState);
        } catch (Exception e) {
            log.info("Failed to flush index remotely", e);
            fallbackWriter.flushIndex(knnVectorValues, totalLiveDocs);
        }
    }

    @Override
    public void mergeIndex(KNNVectorValues<?> knnVectorValues, int totalLiveDocs) throws IOException {
        try {
            remoteIndexBuilder.buildIndexRemotely(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, segmentWriteState);
        } catch (Exception e) {
            log.info("Failed to merge index remotely", e);
            fallbackWriter.mergeIndex(knnVectorValues, totalLiveDocs);
        }
    }
}
