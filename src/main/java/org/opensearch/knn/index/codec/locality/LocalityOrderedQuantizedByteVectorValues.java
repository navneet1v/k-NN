/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.knn.index.codec.scorer.PrefetchHelper;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * A {@link QuantizedByteVectorValues} view over a {@code .veqo} record region. Callers address it
 * by <b>original</b> (insertion-order) ordinal, matching the ordinal space of the raw flat vectors
 * reader. Each accessor maps that original ordinal to its physical position via
 * {@code ordToPhysicalOrdMap} and then seeks {@code physicalOrd * recordSize} into the record slice,
 * so the SIMD scorer path (which extracts the base address from {@link #getSlice()}) works unchanged.
 *
 * <p>{@link #ordToDoc(int)} and {@link #iterator()} are delegated to the raw {@link FloatVectorValues}
 * (the {@code .vec} reader), which owns the {@code originalOrdinal -> docId} mapping and the
 * doc-ascending iterator. Nothing docId-related is stored in {@code .veqo}.
 *
 * <p><b>Identity (physical-addressed) mode:</b> when {@code ordToPhysicalOrdMap} is {@code null},
 * the incoming ordinal is treated as the physical position directly (no mapping). This view is used
 * only by the scorer path, where records must be read by physical position (see
 * {@code LocalityOrderedQuantizedVectorsReader.PhysicalOrdinalTranslatingScorer}); {@link #ordToDoc(int)}
 * / {@link #iterator()} are not meaningful in this mode and must not be used.
 */
@Log4j2
public final class LocalityOrderedQuantizedByteVectorValues extends QuantizedByteVectorValues {

    private final int dimension;
    private final int size;
    private final int codeLength;
    private final int recordSize;
    private final float[] centroid;
    private final float centroidDp;
    private final OptimizedScalarQuantizer quantizer;
    private final ScalarEncoding encoding;
    private final VectorSimilarityFunction similarity;
    // The reader's shared quantized (ADC) scorer, reused for the exact-search scorer() path.
    private final FlatVectorsScorer quantizedVectorScorer;
    private final IndexInput slice;

    private final byte[] vector;
    private final float[] correctionScratch = new float[3];
    private int lastOrd = -1;
    final ByteBuffer byteBuffer;
    private int quantizedComponentSum;

    private final FloatVectorValues floatVectorValues;
    private final int[] ordToPhysicalOrdMap;
    // Scratch for translating ords -> physical positions in prefetch(); grown on demand since the
    // number of ords per bulk-score batch is bounded by ef_search / neighbor count, not a fixed 64.
    private int[] ordToPrefetchScratch = new int[0];

    LocalityOrderedQuantizedByteVectorValues(
        final int dimension,
        final int size,
        final int codeLength,
        final int recordSize,
        final float[] centroid,
        final float centroidDp,
        final OptimizedScalarQuantizer quantizer,
        final ScalarEncoding encoding,
        final VectorSimilarityFunction similarity,
        final FlatVectorsScorer quantizedVectorScorer,
        final int[] ordToPhysicalOrdMap,
        final IndexInput slice,
        final FloatVectorValues floatVectorValues
    ) {
        this.dimension = dimension;
        this.size = size;
        this.codeLength = codeLength;
        this.recordSize = recordSize;
        this.centroid = centroid;
        this.centroidDp = centroidDp;
        this.quantizer = quantizer;
        this.encoding = encoding;
        this.similarity = similarity;
        this.quantizedVectorScorer = quantizedVectorScorer;
        this.ordToPhysicalOrdMap = ordToPhysicalOrdMap;
        this.slice = slice;
        this.byteBuffer = ByteBuffer.allocate(codeLength);
        this.vector = byteBuffer.array();
        this.floatVectorValues = floatVectorValues;
    }

    /** Maps a logical ordinal to its physical record position, or identity when no map is set. */
    private int toPhysical(final int ord) {
        return ordToPhysicalOrdMap == null ? ord : ordToPhysicalOrdMap[ord];
    }

    @Override
    public byte[] vectorValue(final int originalOrd) throws IOException {
        final int physicalOrd = toPhysical(originalOrd);
        if (lastOrd == physicalOrd) {
            return vector;
        }
        slice.seek((long) physicalOrd * recordSize);
        slice.readBytes(byteBuffer.array(), byteBuffer.arrayOffset(), vector.length);
        slice.readFloats(correctionScratch, 0, 3);
        quantizedComponentSum = slice.readInt();
        lastOrd = physicalOrd;
        return vector;
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(final int originalOrd) throws IOException {
        final int physicalOrd = toPhysical(originalOrd);
        if (lastOrd == physicalOrd) {
            return new OptimizedScalarQuantizer.QuantizationResult(
                correctionScratch[0],
                correctionScratch[1],
                correctionScratch[2],
                quantizedComponentSum
            );
        }

        slice.seek((long) physicalOrd * recordSize + codeLength);
        slice.readFloats(correctionScratch, 0, 3);
        quantizedComponentSum = slice.readInt();
        return new OptimizedScalarQuantizer.QuantizationResult(
            correctionScratch[0],
            correctionScratch[1],
            correctionScratch[2],
            quantizedComponentSum
        );
    }

    @Override
    public OptimizedScalarQuantizer getQuantizer() {
        return quantizer;
    }

    @Override
    public ScalarEncoding getScalarEncoding() {
        return encoding;
    }

    @Override
    public float[] getCentroid() {
        return centroid;
    }

    @Override
    public float getCentroidDP() {
        return centroidDp;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public VectorEncoding getEncoding() {
        return VectorEncoding.BYTE;
    }

    @Override
    public int ordToDoc(final int originalOrd) {
        return floatVectorValues.ordToDoc(originalOrd);
    }

    @Override
    public void prefetch(int[] ordsToPrefetch, int numOrds) throws IOException {
        if (ordsToPrefetch == null) {
            return;
        }
        int finalNumOrds = Math.min(numOrds, ordsToPrefetch.length);
        if (finalNumOrds <= 1) {
            return;
        }
        if (ordToPrefetchScratch.length < finalNumOrds) {
            ordToPrefetchScratch = new int[finalNumOrds];
        }
        for (int i = 0; i < finalNumOrds; i++) {
            ordToPrefetchScratch[i] = toPhysical(ordsToPrefetch[i]);
        }
        PrefetchHelper.prefetch(getSlice(), 0, recordSize, ordToPrefetchScratch, finalNumOrds);
    }

    @Override
    public IndexInput getSlice() {
        return slice;
    }

    @Override
    public DocIndexIterator iterator() {
        return floatVectorValues.iterator();
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
        // Exact-search scorer: score `query` against every record, in doc order, reusing the reader's
        // shared quantized (ADC) scorer. That scorer addresses records PHYSICALLY (base + ord*recordSize
        // via the SIMD path), so build it over a physical-addressed (identity) view and translate each
        // original ordinal to its physical position before delegating — mirroring the reader's
        // getRandomVectorScorer / PhysicalOrdinalTranslatingScorer.
        final LocalityOrderedQuantizedByteVectorValues physicalView = new LocalityOrderedQuantizedByteVectorValues(
            dimension,
            size,
            codeLength,
            recordSize,
            centroid,
            centroidDp,
            quantizer,
            encoding,
            similarity,
            quantizedVectorScorer,
            null,
            slice.clone(),
            floatVectorValues
        );
        final RandomVectorScorer physicalDelegate = quantizedVectorScorer.getRandomVectorScorer(similarity, physicalView, query);

        // Translating scorer: stays in ORIGINAL-ordinal space (backed by `this`, so maxOrd/ordToDoc are
        // correct) and maps each ordinal to its physical position before delegating. Its bulkScore
        // pre-translates the batch into physical ordinals in one pass so the delegate's SIMD bulk path
        // reads the right records — same shape as the reader's PhysicalOrdinalTranslatingScorer.
        final RandomVectorScorer translating = new RandomVectorScorer.AbstractRandomVectorScorer(this) {
            private int[] scratch = new int[0];

            @Override
            public float score(final int node) throws IOException {
                return physicalDelegate.score(toPhysical(node));
            }

            @Override
            public float bulkScore(final int[] nodes, final float[] scores, final int numNodes) throws IOException {
                if (scratch.length < numNodes) {
                    scratch = new int[numNodes];
                }
                for (int i = 0; i < numNodes; i++) {
                    scratch[i] = toPhysical(nodes[i]);
                }
                return physicalDelegate.bulkScore(scratch, scores, numNodes);
            }
        };

        final DocIndexIterator iterator = iterator();
        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return translating.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                // Sparse: the field may be sparse (some docs have no vector), matching createADCScorer.
                return Bulk.fromRandomScorerSparse(translating, iterator, matchingDocs);
            }
        };
    }

    @Override
    public int getVectorByteLength() {
        return vector.length;
    }

    @Override
    public QuantizedByteVectorValues copy() throws IOException {
        return new LocalityOrderedQuantizedByteVectorValues(
            dimension,
            size,
            codeLength,
            recordSize,
            centroid,
            centroidDp,
            quantizer,
            encoding,
            similarity,
            quantizedVectorScorer,
            ordToPhysicalOrdMap,
            slice.clone(),
            floatVectorValues.copy()
        );

    }
}
