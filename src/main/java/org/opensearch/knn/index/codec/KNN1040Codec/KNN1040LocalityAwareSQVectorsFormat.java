/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsReader;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

public final class KNN1040LocalityAwareSQVectorsFormat extends FlatVectorsFormat {

    public static final String NAME = "KNN1040LocalityAwareSQVectorsFormat";

    private static final KNN1040ScalarQuantizedVectorScorer KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER = FlatVectorsScorerProvider
        .getKNN1040ScalarQuantizedVectorScorer(FlatVectorsScorerProvider.getLucene99FlatVectorsScorer());

    // Must use the default Lucene scorer here, not KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER.
    // KNN1040ScalarQuantizedVectorScorer.getRandomVectorScorer(float[]) always assumes quantized
    // vectors and will fail (NPE/exception) when called with raw OffHeapFloatVectorValues.
    private static final Lucene99FlatVectorsFormat RAW_VECTOR_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()
    );

    private final QuantizedByteVectorValues.ScalarEncoding encoding;

    public KNN1040LocalityAwareSQVectorsFormat() {
        this(QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE);
    }

    public KNN1040LocalityAwareSQVectorsFormat(final QuantizedByteVectorValues.ScalarEncoding encoding) {
        super(NAME);
        this.encoding = encoding;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return null;
    }

    @Override
    public String toString() {
        return String.format(
            "%s(encoding=%s, scorer=%s, rawVectorFormat=%s)",
            getClass().getSimpleName(),
            encoding,
            KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER,
            RAW_VECTOR_FORMAT
        );
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new LocalityOrderedQuantizedVectorsReader(
            state,
            RAW_VECTOR_FORMAT.fieldsReader(state),
            KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER
        );
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String getName() {
        return NAME;
    }
}
