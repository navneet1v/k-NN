/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.block.scalar;

import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.read.block.scalar.Lucene104Backports;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer.QuantizationResult;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;

import java.io.IOException;

/**
 * The {@code OptimizedScalarQuantizer} block codec's write side: quantizes a cluster's prepared vectors
 * against its center and appends them as fixed-size blocks — the write counterpart of the scalar block
 * reader.
 *
 * <p>The centroid the vectors are quantized against is fixed for the sequence, so it is supplied at
 * construction. Each vector (already rotation-prepared by the source when the field is rotated) is quantized
 * against that center. The quantizer centers its input in place, so the source is required to hand out a
 * per-call copy the writer may mutate (see {@link BlockVectorFormat.Writer#writeBlocks}); the writer does not
 * defensively copy.
 *
 * <p>Only the 1-bit ({@code SINGLE_BIT_QUERY_NIBBLE}) and 2-bit ({@code DIBIT_QUERY_NIBBLE}) encodings are
 * supported; any other encoding is rejected.
 *
 * <p>Each block is written struct-of-arrays, corrective terms first, then the packed codes. Every block holds
 * {@link #blockSize()} vectors except the last, which may be partial. Blocks carry no header; the vector
 * count comes from the sequence length recorded in the field's metadata.
 *
 * <pre>
 *   one block of n vectors (n = blockSize, last block may be partial):
 *     lower  n x int   (floatToIntBits of lowerInterval)
 *     upper  n x int   (floatToIntBits of upperInterval)
 *     add    n x int   (floatToIntBits of additionalCorrection)
 *     sum    n x int   (quantizedComponentSum)
 *     codes  n x codeLen bytes (packed)
 * </pre>
 */
public final class OptimizedScalarQuantizedBlockWriter implements BlockVectorFormat.Writer {

    private static final String UNSUPPORTED_ENCODING = "unsupported scalar encoding: ";

    private final IndexOutput out;
    private final int blockSize;
    private final int dimension;
    private final OptimizedScalarQuantizer quantizer;
    private final ScalarEncoding encoding;

    /** The center every vector in the sequence is quantized against; fixed for this writer. */
    private final float[] centroid;

    private final float[] lower;
    private final float[] upper;
    private final float[] add;
    private final int[] sum;

    /** Unpacked per-dimension codes filled by {@code scalarQuantize}. */
    private final byte[] quantized;

    /** Packed codes for one vector. */
    private final byte[] packed;

    /** Contiguous copy of one block's codes, sized {@code blockSize * codeLength}. */
    private final byte[] codes;
    private final int codeLength;

    /** Vectors currently buffered in the columns — the next write position; reset per sequence. */
    private int buffered;

    public OptimizedScalarQuantizedBlockWriter(
        final IndexOutput out,
        final int blockSize,
        final int dimension,
        final OptimizedScalarQuantizer quantizer,
        final ScalarEncoding encoding,
        final float[] centroid
    ) {
        this.out = out;
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.quantizer = quantizer;
        this.encoding = encoding;
        this.centroid = centroid;
        this.lower = new float[blockSize];
        this.upper = new float[blockSize];
        this.add = new float[blockSize];
        this.sum = new int[blockSize];
        this.quantized = new byte[encoding.getDiscreteDimensions(dimension)];
        this.packed = switch (encoding) {
            case SINGLE_BIT_QUERY_NIBBLE, DIBIT_QUERY_NIBBLE -> new byte[encoding.getDocPackedLength(quantized.length)];
            default -> throw new UnsupportedOperationException(UNSUPPORTED_ENCODING + encoding);
        };
        this.codeLength = packed.length;
        this.codes = new byte[blockSize * codeLength];
    }

    @Override
    public int blockSize() {
        return blockSize;
    }

    @Override
    public void writeBlocks(final FloatVectorValues source) throws IOException {
        if (source.dimension() != dimension) {
            throw new IllegalArgumentException("source dimension " + source.dimension() + " != writer dimension " + dimension);
        }
        final int count = source.size();
        if (count == 0) {
            return;
        }
        resetBuffer();
        for (int ord = 0; ord < count; ord++) {
            buffer(source.vectorValue(ord));
            if (isBufferFull()) {
                writeBlock();
            }
        }
        writeBlock(); // partial last block; a no-op when the count was a multiple of blockSize
    }

    /** Whether the columns hold a full block. */
    private boolean isBufferFull() {
        return buffered == blockSize;
    }

    /** Clears the buffer cursor for the next block. */
    private void resetBuffer() {
        buffered = 0;
    }

    /**
     * Quantizes one vector against the center into the next buffer position: its corrective terms go to the
     * columns, its packed code into the codes buffer. The vector is the source's per-call copy, which
     * {@code scalarQuantize} centers in place.
     */
    private void buffer(final float[] vector) {
        final QuantizationResult terms = quantizer.scalarQuantize(vector, quantized, encoding.getBits(), centroid);
        lower[buffered] = terms.lowerInterval();
        upper[buffered] = terms.upperInterval();
        add[buffered] = terms.additionalCorrection();
        sum[buffered] = terms.quantizedComponentSum();
        pack();
        System.arraycopy(packed, 0, codes, buffered * codeLength, codeLength);
        buffered++;
    }

    /** Packs {@link #quantized} into {@link #packed} for the supported nibble encodings. */
    private void pack() {
        switch (encoding) {
            case SINGLE_BIT_QUERY_NIBBLE -> OptimizedScalarQuantizer.packAsBinary(quantized, packed);
            case DIBIT_QUERY_NIBBLE -> Lucene104Backports.transposeDibit(quantized, packed);
            default -> throw new UnsupportedOperationException(UNSUPPORTED_ENCODING + encoding);
        }
    }

    /**
     * Writes the buffered block struct-of-arrays (the four corrective-term columns, then the codes) and clears
     * the buffer. A no-op when nothing is buffered — the count was a multiple of blockSize.
     *
     * <p>The last block is written with exactly {@code buffered} vectors and is not padded to a full
     * {@code blockSize}; the reader recovers its length from the sequence count in {@code .clam}.
     */
    private void writeBlock() throws IOException {
        if (buffered < 1) {
            return;
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(Float.floatToIntBits(lower[j]));
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(Float.floatToIntBits(upper[j]));
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(Float.floatToIntBits(add[j]));
        }
        for (int j = 0; j < buffered; j++) {
            out.writeInt(sum[j]);
        }
        out.writeBytes(codes, 0, buffered * codeLength);
        resetBuffer();
    }
}
