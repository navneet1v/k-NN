/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

import java.io.IOException;

/**
 * An in-heap {@link QuantizedByteVectorValues} — the records are fully materialized in memory (no
 * {@link IndexInput} / mmap slice). Used for small, hot record sets like the per-field hub entry-point
 * candidates: parsed once at reader-open time into this form, then handed to a scorer whenever needed.
 *
 * <p>Records are addressed by index {@code 0..size-1}. Each holds a 1-bit code ({@code codeLength}
 * bytes) plus its four correction terms, in the same layout the store uses. Because {@link #getSlice()}
 * returns {@code null}, callers must score with a scorer that reads through the values API
 * ({@link #vectorValue}/{@link #getCorrectiveTerms}) — e.g. the plain
 * {@code Lucene104ScalarQuantizedVectorScorer} (Java ADC path) — rather than the SIMD path that
 * requires an off-heap slice.
 *
 * <p>Doc-space methods ({@link #ordToDoc}, {@link #iterator}) are not meaningful here and are not
 * supported; this view exists purely for scoring.
 */
final class HeapQuantizedByteVectorValues extends QuantizedByteVectorValues {

    private final int dimension;
    private final int size;
    private final int codeLength;
    private final float[] centroid;
    private final float centroidDp;
    private final OptimizedScalarQuantizer quantizer;
    private final ScalarEncoding encoding;

    // Per-record columns, all length == size.
    private final byte[][] codes;
    private final float[] lowerIntervals;
    private final float[] upperIntervals;
    private final float[] additionalCorrections;
    private final int[] quantizedComponentSums;

    private HeapQuantizedByteVectorValues(
        final int dimension,
        final int size,
        final int codeLength,
        final float[] centroid,
        final float centroidDp,
        final OptimizedScalarQuantizer quantizer,
        final ScalarEncoding encoding,
        final byte[][] codes,
        final float[] lowerIntervals,
        final float[] upperIntervals,
        final float[] additionalCorrections,
        final int[] quantizedComponentSums
    ) {
        this.dimension = dimension;
        this.size = size;
        this.codeLength = codeLength;
        this.centroid = centroid;
        this.centroidDp = centroidDp;
        this.quantizer = quantizer;
        this.encoding = encoding;
        this.codes = codes;
        this.lowerIntervals = lowerIntervals;
        this.upperIntervals = upperIntervals;
        this.additionalCorrections = additionalCorrections;
        this.quantizedComponentSums = quantizedComponentSums;
    }

    /**
     * Reads {@code size} records from the current position of {@code in} into heap, using the same
     * record layout the store writes (code bytes, then 3 correction floats, then the component sum int).
     * Reading via the {@link IndexInput} decoders keeps the byte/endianness handling identical to the
     * on-disk store — no manual parsing.
     */
    static HeapQuantizedByteVectorValues read(
        final IndexInput in,
        final int size,
        final int dimension,
        final int codeLength,
        final float[] centroid,
        final float centroidDp,
        final OptimizedScalarQuantizer quantizer,
        final ScalarEncoding encoding
    ) throws IOException {
        final byte[][] codes = new byte[size][];
        final float[] lowerIntervals = new float[size];
        final float[] upperIntervals = new float[size];
        final float[] additionalCorrections = new float[size];
        final int[] quantizedComponentSums = new int[size];
        final float[] correctionScratch = new float[3];

        for (int i = 0; i < size; i++) {
            final byte[] code = new byte[codeLength];
            in.readBytes(code, 0, codeLength);
            in.readFloats(correctionScratch, 0, 3);
            lowerIntervals[i] = correctionScratch[0];
            upperIntervals[i] = correctionScratch[1];
            additionalCorrections[i] = correctionScratch[2];
            quantizedComponentSums[i] = in.readInt();
            codes[i] = code;
        }

        return new HeapQuantizedByteVectorValues(
            dimension,
            size,
            codeLength,
            centroid,
            centroidDp,
            quantizer,
            encoding,
            codes,
            lowerIntervals,
            upperIntervals,
            additionalCorrections,
            quantizedComponentSums
        );
    }

    @Override
    public byte[] vectorValue(final int ord) {
        return codes[ord];
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(final int ord) {
        return new OptimizedScalarQuantizer.QuantizationResult(
            lowerIntervals[ord],
            upperIntervals[ord],
            additionalCorrections[ord],
            quantizedComponentSums[ord]
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
    public int getVectorByteLength() {
        return codeLength;
    }

    /** Heap-backed — no slice. Forces scorers to read through the values API (Java ADC path). */
    @Override
    public IndexInput getSlice() {
        return null;
    }

    @Override
    public int ordToDoc(final int ord) {
        throw new UnsupportedOperationException("HeapQuantizedByteVectorValues is scoring-only; ordToDoc is not supported");
    }

    @Override
    public DocIndexIterator iterator() {
        throw new UnsupportedOperationException("HeapQuantizedByteVectorValues is scoring-only; iterator is not supported");
    }

    @Override
    public VectorScorer scorer(final float[] query) {
        throw new UnsupportedOperationException("HeapQuantizedByteVectorValues does not build its own scorer");
    }

    @Override
    public QuantizedByteVectorValues copy() {
        // Immutable, columnar, and scratch-free (vectorValue returns the stored array) — safe to share.
        return this;
    }
}
