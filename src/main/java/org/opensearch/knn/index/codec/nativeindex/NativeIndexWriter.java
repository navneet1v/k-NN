/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.remote.RemoteIndexBuilder;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Interface for writing a KNN index field in a segment. This is intended to be used for native engines.
 */
public interface NativeIndexWriter {

    /**
     * flushes the index
     *
     * @param knnVectorValues
     * @throws IOException
     */
    void flushIndex(KNNVectorValues<?> knnVectorValues, int totalLiveDocs) throws IOException;

    /**
     * Merges kNN index
     * @param knnVectorValues
     * @throws IOException
     */
    void mergeIndex(KNNVectorValues<?> knnVectorValues, int totalLiveDocs) throws IOException;

    /**
     * This method either gets a {@link RemoteIndexBuilder} or a {@link LocalNativeIndexWriter}. This is expected to be called
     * from {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter} instead of any other getWriter method.
     *
     * @param fieldInfo
     * @param segmentWriteState
     * @param quantizationState
     * @param remoteIndexBuilder
     * @param knnVectorValuesSupplier
     * @return {@link RemoteNativeIndexWriter} if {@link RemoteIndexBuilder} is available configured and configured properly. Uses {@link LocalNativeIndexWriter} as fallback.
     */
    static NativeIndexWriter getWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState segmentWriteState,
        final QuantizationState quantizationState,
        final RemoteIndexBuilder remoteIndexBuilder,
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier
    ) {
        // TODO: We will add threshold settings for using this featuer here as well, see:
        // https://github.com/opensearch-project/k-NN/issues/2391
        if (remoteIndexBuilder != null && remoteIndexBuilder.shouldBuildIndexRemotely()) {
            return new RemoteNativeIndexWriter(
                fieldInfo,
                segmentWriteState,
                createWriter(fieldInfo, segmentWriteState, quantizationState),
                remoteIndexBuilder,
                knnVectorValuesSupplier
            );
        } else {
            return createWriter(fieldInfo, segmentWriteState, quantizationState);
        }
    }

    /**
     * Gets the correct writer type from fieldInfo
     *
     * @param fieldInfo
     * @return correct NativeIndexWriter to make index specified in fieldInfo
     */
    static NativeIndexWriter getWriter(final FieldInfo fieldInfo, SegmentWriteState state) {
        return createWriter(fieldInfo, state, null);
    }

    /**
     * Gets the correct writer type for the specified field, using a given QuantizationModel.
     *
     * This method returns a NativeIndexWriter instance that is tailored to the specific characteristics
     * of the field described by the provided FieldInfo. It determines whether to use a template-based
     * writer or an iterative approach based on the engine type and whether the field is associated with a template.
     *
     * If quantization is required, the QuantizationModel is passed to the writer to facilitate the quantization process.
     *
     * @param fieldInfo          The FieldInfo object containing metadata about the field for which the writer is needed.
     * @param state              The SegmentWriteState representing the current segment's writing context.
     * @param quantizationState  The QuantizationState that contains  quantization state required for quantization
     * @return                   A NativeIndexWriter instance appropriate for the specified field, configured with or without quantization.
     */
    static NativeIndexWriter getWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        final QuantizationState quantizationState
    ) {
        return createWriter(fieldInfo, state, quantizationState);
    }

    /**
     * Helper method to create the appropriate NativeIndexWriter based on the field info and quantization state.
     *
     * @param fieldInfo          The FieldInfo object containing metadata about the field for which the writer is needed.
     * @param state              The SegmentWriteState representing the current segment's writing context.
     * @param quantizationState  The QuantizationState that contains quantization state required for quantization, can be null.
     * @return                   A NativeIndexWriter instance appropriate for the specified field, configured with or without quantization.
     */
    private static NativeIndexWriter createWriter(
        final FieldInfo fieldInfo,
        final SegmentWriteState state,
        @Nullable final QuantizationState quantizationState
    ) {
        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        boolean isTemplate = fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = !isTemplate && KNNEngine.FAISS == knnEngine;
        NativeIndexBuildStrategy strategy = iterative
            ? MemOptimizedNativeIndexBuildStrategy.getInstance()
            : DefaultIndexBuildStrategy.getInstance();
        return new LocalNativeIndexWriter(state, fieldInfo, strategy, quantizationState);
    }
}
