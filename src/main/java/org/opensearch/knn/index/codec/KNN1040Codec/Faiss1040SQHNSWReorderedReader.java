/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
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

        // Seed the HNSW search from the hub closest to the query (scored against the stored hub records)
        // instead of CAGRA's random entry points. Wrapping the collector as Seeded makes the searcher
        // honor our entry point and skip its random-seed selection.
        memoryOptimizedSearcher.search(target, maybeSeedWithClosestHub(field, target, knnCollector), acceptDocs);
    }

    /**
     * If the field has hubs, returns a collector that seeds the search from the single hub closest to
     * {@code target}; otherwise returns {@code knnCollector} unchanged (falling back to the searcher's
     * default entry-point selection). A collector already carrying a {@link KnnSearchStrategy.Seeded}
     * (e.g. from a seed query) is left untouched.
     */
    private KnnCollector maybeSeedWithClosestHub(final String field, final float[] target, final KnnCollector knnCollector)
        throws IOException {
        if (knnCollector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded) {
            return knnCollector;
        }
        if ((flatVectorsReader instanceof LocalityOrderedQuantizedVectorsReader) == false) {
            return knnCollector;
        }
        final int bestHub = ((LocalityOrderedQuantizedVectorsReader) flatVectorsReader).selectBestHubOrdinal(field, target);
        if (bestHub < 0) {
            return knnCollector;
        }
        return new KnnCollector.Decorator(knnCollector) {
            @Override
            public KnnSearchStrategy getSearchStrategy() {
                // Seeded delegates block iteration to the wrapped strategy, so it must be non-null.
                final KnnSearchStrategy original = knnCollector.getSearchStrategy();
                return new KnnSearchStrategy.Seeded(
                    singleEntryPoint(bestHub),
                    1,
                    original != null ? original : KnnSearchStrategy.Hnsw.DEFAULT
                );
            }
        };
    }

    /** A one-shot {@link DocIdSetIterator} yielding a single ordinal — the chosen hub entry point. */
    private static DocIdSetIterator singleEntryPoint(final int ordinal) {
        return new DocIdSetIterator() {
            private int doc = -1;
            private boolean consumed = false;

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                if (consumed) {
                    doc = NO_MORE_DOCS;
                } else {
                    consumed = true;
                    doc = ordinal;
                }
                return doc;
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }

            @Override
            public long cost() {
                return 1L;
            }
        };
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Byte vector based search is not supported.");
    }

    @Override
    public void warmUp(String fieldName) throws IOException {
        throw new UnsupportedOperationException("warmUp is not supported");
    }
}
