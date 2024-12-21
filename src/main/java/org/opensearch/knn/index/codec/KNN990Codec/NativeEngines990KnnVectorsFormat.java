/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;

/**
 * This is a Vector format that will be used for Native engines like Faiss and Nmslib for reading and writing vector
 * related data structures.
 */
public class NativeEngines990KnnVectorsFormat extends KnnVectorsFormat {
    /** The format for storing, reading, merging vectors on disk */
    private static FlatVectorsFormat flatVectorsFormat;
    private static final String FORMAT_NAME = "NativeEngines990KnnVectorsFormat";
    private static int approximateThreshold;
    private String indexUUID;

    public NativeEngines990KnnVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public NativeEngines990KnnVectorsFormat(int approximateThreshold) {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()), approximateThreshold);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat) {
        this(flatVectorsFormat, KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat, int approximateThreshold) {
        super(FORMAT_NAME);
        NativeEngines990KnnVectorsFormat.flatVectorsFormat = flatVectorsFormat;
        NativeEngines990KnnVectorsFormat.approximateThreshold = approximateThreshold;
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat, int approximateThreshold, final String indexUUID) {
        super(FORMAT_NAME);
        NativeEngines990KnnVectorsFormat.flatVectorsFormat = flatVectorsFormat;
        NativeEngines990KnnVectorsFormat.approximateThreshold = approximateThreshold;
        this.indexUUID = indexUUID;
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new NativeEngines990KnnVectorsWriter(state, flatVectorsFormat.fieldsWriter(state), approximateThreshold, indexUUID);
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new NativeEngines990KnnVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    @Override
    public String toString() {
        return "NativeEngines99KnnVectorsFormat(name="
            + this.getClass().getSimpleName()
            + ", flatVectorsFormat="
            + flatVectorsFormat
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }
}
