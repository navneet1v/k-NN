/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsReader;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;

import java.io.IOException;

public class Faiss1040SQHNSWReorderedReader extends AbstractNativeEnginesKnnVectorsReader {

    public Faiss1040SQHNSWReorderedReader(final SegmentReadState segmentReadState, final LocalityOrderedQuantizedVectorsReader reader)
        throws IOException {
        super(segmentReadState, reader);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss scalar quantized format Reordered reader");
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfos.fieldInfo(field));

        if (memoryOptimizedSearcher == null) {
            throw new IllegalStateException(
                "Faiss scalar quantized format requires memory optimized search but searcher could not be loaded for field [" + field + "]"
            );
        }

        memoryOptimizedSearcher.search(target, knnCollector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {

    }

    @Override
    public void warmUp(String fieldName) throws IOException {

    }
}
