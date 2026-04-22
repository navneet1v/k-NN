/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.index.warmup.WarmableReader;
import org.opensearch.knn.index.query.metrics.SearchMetricsContext;

import java.io.IOException;

/**
 * Reader for the cluster-based ANN format. Delegates vector reads to a {@link FlatVectorsReader}
 * which reads raw vectors stored by {@link ClusterANN1040KnnVectorsWriter}. The cluster index
 * search logic will be layered on top in a follow-up.
 */
@Log4j2
public class ClusterANN1040KnnVectorsReader extends KnnVectorsReader implements WarmableReader {

    private final FlatVectorsReader flatVectorsReader;

    /**
     * Creates a new reader that delegates vector reads to the given flat vectors reader.
     *
     * @param flatVectorsReader the underlying flat vectors reader
     * @param state the segment read state
     */
    public ClusterANN1040KnnVectorsReader(FlatVectorsReader flatVectorsReader, SegmentReadState state) {
        this.flatVectorsReader = flatVectorsReader;
        log.info("[ClusterANN] reader created for segment: {}", state.segmentInfo.name);
    }

    @Override
    public void checkIntegrity() throws IOException {
        flatVectorsReader.checkIntegrity();
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        log.info("[ClusterANN] getFloatVectorValues: {}", field);
        return flatVectorsReader.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return flatVectorsReader.getByteVectorValues(field);
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer scorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (scorer == null) {
            return;
        }
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        int maxOrd = scorer.maxOrd();
        long vectorBytesRead = 0;
        long vectorByteSize = target.length;
        for (int ord = 0; ord < maxOrd; ord++) {
            int docId = scorer.ordToDoc(ord);
            if (acceptBits != null && !acceptBits.get(docId)) {
                continue;
            }
            knnCollector.collect(docId, scorer.score(ord));
            knnCollector.incVisitedCount(1);
            vectorBytesRead += vectorByteSize;
        }
        SearchMetricsContext.current().addVectorBytesPrefetched(vectorBytesRead);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer scorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (scorer == null) {
            return;
        }
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        int maxOrd = scorer.maxOrd();
        for (int ord = 0; ord < maxOrd; ord++) {
            int docId = scorer.ordToDoc(ord);
            if (acceptBits != null && !acceptBits.get(docId)) {
                continue;
            }
            knnCollector.collect(docId, scorer.score(ord));
            knnCollector.incVisitedCount(1);
        }
    }

    /**
     * Warms up the flat vector data for the given field by reading it into the OS page cache.
     * Once the cluster index is implemented, this will also warm up the cluster structures.
     *
     * @param fieldName the name of the vector field to warm up
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void warmUp(String fieldName) throws IOException {
        log.info("[ClusterANN] warmUp: {}", fieldName);
        FloatVectorValues values = flatVectorsReader.getFloatVectorValues(fieldName);
        if (values != null) {
            WarmupUtil.readAll(values);
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsReader);
    }
}
