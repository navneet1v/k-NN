/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * FlatVectors format for writing the vectors only. No search k-nn search data structures are built in this format.
 */
public class KNN990FlatVectorsFormat extends FlatVectorsFormat {

    private final Lucene99FlatVectorsFormat flatVectorsFormat;
    private static final String NAME = "KNN990FlatVectorsFormat";

    public KNN990FlatVectorsFormat() {
        super(NAME);
        flatVectorsFormat = new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer());
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return flatVectorsFormat.fieldsWriter(state);
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return flatVectorsFormat.fieldsReader(state);
    }
}
