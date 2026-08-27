/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsReader;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

/**
 * Dedicated format for Faiss SQ vector fields.
 *
 * <p>Uses {@link KNN1040LocalityAwareSQVectorsFormat} with 1-bit quantization
 * ({@link ScalarEncoding#SINGLE_BIT_QUERY_NIBBLE}) for flat vector storage (.vec/.veq files),
 * while HNSW graph construction is delegated to the native Faiss engine (.faiss files).
 *
 * <p>A field is routed to this format when its method parameters contain
 * {@code "encoder": {"name": "sq", "bits": 1}}. See {@code FaissCodecFormatResolver} for the
 * routing logic in {@code BasePerFieldKnnVectorsFormat.getKnnVectorsFormatForField}.
 *
 * @see LocalityOrderedQuantizedVectorsWriter for reordering
 * @see LocalityOrderedQuantizedVectorsReader for reordering
 */
@Log4j2
public class Faiss1040SQHNSWReorderedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040SQHNSWReorderedKnnVectorsFormat";

    // TODO : We have to make it scalable for other encoding types, not limit this on `ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE`.
    private static final KNN1040ScalarQuantizedVectorsFormat faissSqFlatFormat = new KNN1040ScalarQuantizedVectorsFormat(
        ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
    );

    private static final Lucene99FlatVectorsFormat RAW_VECTOR_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()
    );

    private static final KNN1040ScalarQuantizedVectorScorer KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER = FlatVectorsScorerProvider
        .getKNN1040ScalarQuantizedVectorScorer(FlatVectorsScorerProvider.getLucene99FlatVectorsScorer());

    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    /**
     * Constructor needed for SPI
     */
    public Faiss1040SQHNSWReorderedKnnVectorsFormat() {
        this(new NativeIndexBuildStrategyFactory());
    }

    public Faiss1040SQHNSWReorderedKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        super(FORMAT_NAME);
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Faiss1040SQHNSWReorderedWriter(
            state,
            faissSqFlatFormat.fieldsWriter(state),
            faissSqFlatFormat::fieldsReader,
            new LocalityOrderedQuantizedVectorsWriter(state, ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE),
            nativeIndexBuildStrategyFactory
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new Faiss1040SQHNSWReorderedReader(
            state,
            new LocalityOrderedQuantizedVectorsReader(state, RAW_VECTOR_FORMAT.fieldsReader(state), KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER)
        );
    }

    /**
     * Uses Faiss max dimension since the HNSW graph is built by native Faiss.
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(name=" + this.getClass().getSimpleName() + ")";
    }
}
