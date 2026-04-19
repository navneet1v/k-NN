/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;

/**
 * Writer for the cluster-based ANN format. Delegates vector storage to a {@link FlatVectorsWriter}
 * which writes raw vectors without any index structure. The cluster index building logic
 * will be layered on top in a follow-up.
 */
@Log4j2
public class ClusterANN1040KnnVectorsWriter extends KnnVectorsWriter {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(ClusterANN1040KnnVectorsWriter.class);

    private final FlatVectorsWriter flatVectorsWriter;

    /**
     * Creates a new writer that delegates vector storage to the given flat vectors writer.
     *
     * @param flatVectorsWriter the underlying flat vectors writer for raw vector storage
     */
    public ClusterANN1040KnnVectorsWriter(FlatVectorsWriter flatVectorsWriter) {
        this.flatVectorsWriter = flatVectorsWriter;
    }

    /**
     * Adds a new field for indexing, delegating to the flat vectors writer.
     *
     * @param fieldInfo the field info
     * @return a writer for the field's vectors
     * @throws IOException if an I/O error occurs
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        log.info("[ClusterANN] addField: {}", fieldInfo.getName());
        return flatVectorsWriter.addField(fieldInfo);
    }

    /**
     * Flushes buffered vectors to disk via the flat vectors writer.
     *
     * @param maxDoc the maximum document number
     * @param sortMap the sort map, or null if unsorted
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        log.info("[ClusterANN] flush: maxDoc={}", maxDoc);
        flatVectorsWriter.flush(maxDoc, sortMap);
    }

    /**
     * Merges vectors from multiple segments via the flat vectors writer.
     *
     * @param mergeState the merge state
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        log.info("[ClusterANN] mergeOneField: {}", fieldInfo.getName());
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed();
    }

    /**
     * Finishes writing vectors for this segment, delegating to the flat vectors writer.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void finish() throws IOException {
        log.info("[ClusterANN] finish");
        flatVectorsWriter.finish();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter);
    }
}
