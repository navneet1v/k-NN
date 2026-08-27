/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

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
final class LocalityOrderedQuantizedByteVectorValues extends QuantizedByteVectorValues {

    private final int dimension;
    private final int size;
    private final int codeLength;
    private final int recordSize;
    private final float[] centroid;
    private final float centroidDp;
    private final OptimizedScalarQuantizer quantizer;
    private final ScalarEncoding encoding;
    private final IndexInput slice;

    private final byte[] vector;
    private final float[] correctionScratch = new float[3];
    private int lastOrd = -1;
    final ByteBuffer byteBuffer;
    private int quantizedComponentSum;

    // Physical ordinals sorted by ascending docId; built lazily when iterator() is first used.
    // Needed because the physical (locality) ordering is not doc-ascending, but a DocIdSetIterator
    // must return docIDs in ascending order.
    private int[] docSortedOrds;
    private final FloatVectorValues floatVectorValues;
    private final int[] ordToPhysicalOrdMap;

    LocalityOrderedQuantizedByteVectorValues(
        final int dimension,
        final int size,
        final int codeLength,
        final int recordSize,
        final float[] centroid,
        final float centroidDp,
        final OptimizedScalarQuantizer quantizer,
        final ScalarEncoding encoding,
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

        // 1. calculate offset and prefetch immediately
        for (int i = 0; i < finalNumOrds; i++) {
            long offset = (long) toPhysical(ordsToPrefetch[i]) * recordSize;
            slice.prefetch(offset, recordSize);
        }
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
        throw new RuntimeException("Not implemented right now");
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
            ordToPhysicalOrdMap,
            slice.clone(),
            floatVectorValues.copy()
        );

    }
}
