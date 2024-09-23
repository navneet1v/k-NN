/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.Bits;
import org.opensearch.common.StopWatch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class to get merged VectorValues from MergeState
 */
@Log4j2
public final class KNNMergeVectorValuesUtil {

    /**
     * Gets list of {@link KNNVectorValuesSub} for {@link FloatVectorValues} from a merge state and returns the iterator which
     * iterates over live docs from all segments while mapping docIds.
     *
     * @param fieldInfo
     * @param mergeState
     * @return List of KNNVectorSub
     * @throws IOException
     */
    public static FloatVectorValues mergeFloatVectorValues(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        assert fieldInfo != null && fieldInfo.hasVectorValues();
        if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new UnsupportedOperationException("Cannot merge vectors encoded as [" + fieldInfo.getVectorEncoding() + "] as FLOAT32");
        }
        final List<KNNVectorValuesSub<FloatVectorValues>> subs = new ArrayList<>();
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
            if (knnVectorsReader != null) {
                FloatVectorValues values = knnVectorsReader.getFloatVectorValues(fieldInfo.getName());
                if (values != null) {
                    final Bits liveDocs = mergeState.liveDocs[i];
                    StopWatch stopWatch = new StopWatch().start();
                    // live docs cardinality is needed to make sure deletedDocs are not included in final count for merge
                    final int liveDocsInt;
                    if (liveDocs != null) {
                        liveDocsInt = cardinality(values, liveDocs);
                        values = knnVectorsReader.getFloatVectorValues(fieldInfo.getName());
                    } else {
                        liveDocsInt = Math.toIntExact(values.cost());
                    }
                    stopWatch.stop();
                    log.debug("[FloatVectorValues] Time to compute live docs cardinality {} ms", stopWatch.totalTime().millis());
                    subs.add(new KNNVectorValuesSub<>(mergeState.docMaps[i], values, liveDocsInt));
                }
            }
        }
        return new MergeFloat32VectorValues(subs, mergeState);
    }

    /**
     * Gets list of {@link KNNVectorValuesSub} for {@link ByteVectorValues} from a merge state. This can be further
     * used to create an iterator for getting the docs and its vector values
     * @param fieldInfo
     * @param mergeState
     * @return List of KNNVectorSub
     * @throws IOException
     */
    public static ByteVectorValues mergeByteVectorValues(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        assert fieldInfo != null && fieldInfo.hasVectorValues();
        if (fieldInfo.getVectorEncoding() != VectorEncoding.BYTE) {
            throw new UnsupportedOperationException("Cannot merge vectors encoded as [" + fieldInfo.getVectorEncoding() + "] as BYTE");
        }
        final List<KNNVectorValuesSub<ByteVectorValues>> subs = new ArrayList<>();
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
            if (knnVectorsReader != null) {
                ByteVectorValues values = knnVectorsReader.getByteVectorValues(fieldInfo.getName());
                if (values != null) {
                    final Bits liveDocs = mergeState.liveDocs[i];
                    StopWatch stopWatch = new StopWatch().start();
                    // live docs cardinality is needed to make sure deletedDocs are not included in final count for merge
                    final int liveDocsInt;
                    if (liveDocs != null) {
                        liveDocsInt = cardinality(values, liveDocs);
                        values = knnVectorsReader.getByteVectorValues(fieldInfo.getName());
                    } else {
                        liveDocsInt = Math.toIntExact(values.cost());
                    }
                    stopWatch.stop();
                    log.debug("[ByteVectorValues] Time to compute live docs cardinality {} ms", stopWatch.totalTime().millis());
                    subs.add(new KNNVectorValuesSub<>(mergeState.docMaps[i], values, liveDocsInt));
                }
            }
        }
        return new MergeByteVectorValues(subs, mergeState);
    }

    private static int cardinality(final DocIdSetIterator iterator, final Bits liveDocs) throws IOException {
        int count = 0;
        while (iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            if (liveDocs.get(iterator.docID())) {
                count++;
            }
        }
        return count;
    }

    private static class KNNVectorValuesSub<T extends DocIdSetIterator> extends DocIDMerger.Sub {
        final T values;
        final int liveDocs;

        KNNVectorValuesSub(MergeState.DocMap docMap, T values, int liveDocs) {
            super(docMap);
            this.values = values;
            this.liveDocs = liveDocs;
        }

        @Override
        public int nextDoc() throws IOException {
            return values.nextDoc();
        }
    }

    /**
     * Iterator to get mapped docsIds from MergeState
     */
    private static class MergeFloat32VectorValues extends FloatVectorValues {

        private final DocIDMerger<KNNMergeVectorValuesUtil.KNNVectorValuesSub<FloatVectorValues>> docIdMerger;
        private final int liveDocs;
        private int docId;
        private final List<KNNMergeVectorValuesUtil.KNNVectorValuesSub<FloatVectorValues>> subs;
        private KNNMergeVectorValuesUtil.KNNVectorValuesSub<FloatVectorValues> current;

        MergeFloat32VectorValues(
            final List<KNNMergeVectorValuesUtil.KNNVectorValuesSub<FloatVectorValues>> subs,
            final MergeState mergeState
        ) throws IOException {
            this.subs = subs;
            this.docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
            int totalSize = 0;
            for (KNNMergeVectorValuesUtil.KNNVectorValuesSub<FloatVectorValues> sub : subs) {
                totalSize += sub.liveDocs;
            }
            this.liveDocs = totalSize;
            this.docId = -1;
        }

        @Override
        public int docID() {
            return docId;
        }

        @Override
        public int nextDoc() throws IOException {
            current = docIdMerger.next();
            if (current == null) {
                docId = NO_MORE_DOCS;
            } else {
                docId = current.mappedDocID;
            }
            return docId;
        }

        @Override
        public float[] vectorValue() throws IOException {
            return current.values.vectorValue();
        }

        @Override
        public int advance(int target) {
            throw new UnsupportedOperationException("call to advance for MergeFloat32VectorValues not supported");
        }

        @Override
        public int size() {
            return liveDocs;
        }

        @Override
        public int dimension() {
            return subs.get(0).values.dimension();
        }

        @Override
        public VectorScorer scorer(float[] target) {
            throw new UnsupportedOperationException("call to scorer for MergeFloat32VectorValues is not supported");
        }
    }

    /**
     * Iterator to get mapped docsIds from MergeState
     */
    @VisibleForTesting
    private static class MergeByteVectorValues extends ByteVectorValues {

        private final DocIDMerger<KNNMergeVectorValuesUtil.KNNVectorValuesSub<ByteVectorValues>> docIdMerger;
        private final int liveDocs;
        private int docId;
        private final List<KNNMergeVectorValuesUtil.KNNVectorValuesSub<ByteVectorValues>> subs;
        private KNNMergeVectorValuesUtil.KNNVectorValuesSub<ByteVectorValues> current;

        MergeByteVectorValues(final List<KNNMergeVectorValuesUtil.KNNVectorValuesSub<ByteVectorValues>> subs, final MergeState mergeState)
            throws IOException {

            this.subs = subs;
            this.docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
            int totalSize = 0;
            for (KNNMergeVectorValuesUtil.KNNVectorValuesSub<ByteVectorValues> sub : subs) {
                totalSize += sub.liveDocs;
            }
            this.liveDocs = totalSize;
            this.docId = -1;
        }

        @Override
        public int docID() {
            return docId;
        }

        @Override
        public int nextDoc() throws IOException {
            current = docIdMerger.next();
            if (current == null) {
                docId = NO_MORE_DOCS;
            } else {
                docId = current.mappedDocID;
            }
            return docId;
        }

        @Override
        public byte[] vectorValue() throws IOException {
            return current.values.vectorValue();
        }

        @Override
        public int advance(int target) {
            throw new UnsupportedOperationException("call to advance for MergeByteVectorValues not supported");
        }

        @Override
        public int size() {
            return liveDocs;
        }

        @Override
        public int dimension() {
            return subs.get(0).values.dimension();
        }

        @Override
        public VectorScorer scorer(byte[] target) {
            throw new UnsupportedOperationException("call to scorer for MergeByteVectorValues is not supported");
        }
    }
}
