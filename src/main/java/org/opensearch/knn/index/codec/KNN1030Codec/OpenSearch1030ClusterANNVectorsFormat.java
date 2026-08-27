/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

public class OpenSearch1030ClusterANNVectorsFormat extends KnnVectorsFormat {
    /**
     * Sole constructor
     *
     * @param name
     */
    protected OpenSearch1030ClusterANNVectorsFormat(String name) {
        super(name);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return null;
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return null;
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return 16000;
    }
}
