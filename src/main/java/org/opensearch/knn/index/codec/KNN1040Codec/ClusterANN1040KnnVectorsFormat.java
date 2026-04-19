/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.codec.scorer.NativeEngines990KnnVectorsScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.MAX_DIMENSIONS_SUPPORTED_BY_KNN_VECTOR_SEARCH;

/**
 * Placeholder {@link KnnVectorsFormat} for the engine-less cluster-based ANN algorithm.
 *
 * <p>This format uses a flat vectors scorer backed by {@link Lucene99FlatVectorsFormat} with
 * prefetch-aware scoring for SIMD-friendly distance computation. The actual cluster index
 * building and search logic will be implemented in a follow-up.
 *
 * <p>Currently, all methods return null/zero as placeholders. Once the codec layer is
 * implemented, {@link #fieldsWriter} will build the cluster index structure during segment
 * flush/merge, and {@link #fieldsReader} will read it for cluster-based search.
 */
public class ClusterANN1040KnnVectorsFormat extends KnnVectorsFormat {

    private static final FlatVectorsFormat flatVectorsFormat = new Lucene99FlatVectorsFormat(
        new PrefetchableFlatVectorScorer(new NativeEngines990KnnVectorsScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()))
    );
    private static final String FORMAT_NAME = "ClusterANN1040KnnVectorsFormat";

    /**
     * Constructs a ClusterANN1040KnnVectorsFormat with the default format name.
     */
    public ClusterANN1040KnnVectorsFormat() {
        this(FORMAT_NAME);
    }

    /**
     * Constructs a ClusterANN1040KnnVectorsFormat with the given name.
     *
     * @param name the format name used for codec registration
     */
    public ClusterANN1040KnnVectorsFormat(final String name) {
        super(name);
    }

    /**
     * Returns a writer that stores vectors using the flat vectors format.
     * The flat format writes raw vectors without any index structure. The cluster
     * index building logic will be layered on top in a follow-up.
     *
     * @param state the segment write state
     * @return a {@link ClusterANN1040KnnVectorsWriter} backed by {@link Lucene99FlatVectorsFormat}
     * @throws IOException if an I/O error occurs
     */
    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new ClusterANN1040KnnVectorsWriter(flatVectorsFormat.fieldsWriter(state));
    }

    /**
     * Returns a reader that loads vectors using the flat vectors format.
     * The flat format reads raw vectors stored by {@link ClusterANN1040KnnVectorsWriter}.
     * The cluster index search logic will be layered on top in a follow-up.
     *
     * @param state the segment read state
     * @return a {@link ClusterANN1040KnnVectorsReader} backed by {@link Lucene99FlatVectorsFormat}
     * @throws IOException if an I/O error occurs
     */
    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new ClusterANN1040KnnVectorsReader(flatVectorsFormat.fieldsReader(state), state);
    }

    /**
     * Returns the maximum number of vector dimensions supported by this format.
     *
     * @param fieldName the field name
     * @return the maximum dimensions supported
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return MAX_DIMENSIONS_SUPPORTED_BY_KNN_VECTOR_SEARCH;
    }
}
