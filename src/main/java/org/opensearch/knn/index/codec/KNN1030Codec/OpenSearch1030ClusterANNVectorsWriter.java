/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Sorter;

import java.io.IOException;

public class OpenSearch1030ClusterANNVectorsWriter extends KnnVectorsWriter {
    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        return null;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {

    }

    @Override
    public void finish() throws IOException {

    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }
}
