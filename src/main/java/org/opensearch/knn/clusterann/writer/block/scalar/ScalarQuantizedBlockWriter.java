package org.opensearch.knn.clusterann.writer.block.scalar;

import org.opensearch.knn.clusterann.reader.block.scalar.ScalarEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;

import java.io.IOException;
import java.util.Arrays;

/**
 * {@link BlockVectorFormat.Writer} over scalar-quantized vectors — the exact inverse of
 * {@code ScalarQuantizedBlockReader}.
 *
 * <p>A block is columnar, corrections before codes:
 *
 * <pre>
 * lower[m] float | upper[m] float | add[m] float | sum[m] int | codes[m × packedBytes]
 * </pre>
 *
 * <p>with {@code m} the block's own vector count. That ordering is what lets a scan read the corrections, bound the
 * block, and walk away without touching the codes — the bulk of the bytes. It is also why the columns come first
 * rather than interleaved per vector.
 *
 * <p><b>Full blocks are exactly one stride.</b> The reader positions block {@code b} by arithmetic
 * ({@code b × blockSize × perVectorBytes}), never by walking, which holds because every block but the last is full.
 * The last block is written short, not padded: nothing seeks past it.
 *
 * <p><b>Codes are transposed into bit planes at write time</b>, once per vector, ever. That is the whole bet of the
 * format — the query is transposed once per posting, and every score is then {@code AND} plus {@code popcount}
 * instead of a multiply-add per dimension. It also means the bit order here and in the query's transposition must
 * agree; both use Lucene's own routines so they cannot drift apart.
 */
public final class ScalarQuantizedBlockWriter implements BlockVectorFormat.Writer {

    private static final int DOC_BITS_ONE = 1;
    private static final int DOC_BITS_TWO = 2;

    private final IndexOutput out;
    private final int blockSize;
    private final int dimension;
    private final int docBits;
    private final int packedBytes;
    private final OptimizedScalarQuantizer quantizer;

    // One block's columns, filled as vectors arrive and flushed together. Sized once; a block is the unit of I/O, so
    // it is also the unit of buffering.
    private final float[] lower;
    private final float[] upper;
    private final float[] add;
    private final int[] sum;
    private final byte[] codes;

    /** Scratch for one vector: the raw quantized values, then their bit-plane form. */
    private final byte[] rawCodes;
    private final byte[] packed;
    private final float[] centred;

    private int buffered;

    public ScalarQuantizedBlockWriter(
        IndexOutput out,
        int blockSize,
        int dimension,
        ScalarEncoding encoding,
        OptimizedScalarQuantizer quantizer
    ) {
        this.out = out;
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.docBits = encoding.getDocBitsPerDim();
        this.packedBytes = encoding.getDocPackedLength(dimension);
        this.quantizer = quantizer;

        if (docBits != DOC_BITS_ONE && docBits != DOC_BITS_TWO) {
            // Kept in step with the read side on purpose: its dot kernels cover one- and two-bit doc codes, so
            // writing a width it cannot score would produce a segment that opens and then fails per query.
            throw new IllegalArgumentException("Only 1- and 2-bit doc codes are supported, got: " + docBits);
        }

        this.lower = new float[blockSize];
        this.upper = new float[blockSize];
        this.add = new float[blockSize];
        this.sum = new int[blockSize];
        this.codes = new byte[blockSize * packedBytes];

        this.rawCodes = new byte[dimension];
        this.packed = new byte[packedBytes];
        this.centred = new float[dimension];
    }

    @Override
    public int blockSize() {
        return blockSize;
    }

    /**
     * Quantize {@code vector} against {@code reference} and buffer it, flushing the block when it fills.
     *
     * <p>The quantizer centres its input in place and picks the interval that minimises error for <em>this</em>
     * vector, which is why {@code lower} and {@code upper} are per-vector columns rather than per-block constants.
     */
    @Override
    public void addVector(float[] vector, float[] reference) throws IOException {
        if (vector.length != dimension) {
            throw new IllegalArgumentException("Expected " + dimension + " dimensions, got: " + vector.length);
        }

        System.arraycopy(vector, 0, centred, 0, dimension);
        OptimizedScalarQuantizer.QuantizationResult result = quantizer.multiScalarQuantize(
            centred,
            new byte[][] { rawCodes },
            new byte[] { (byte) docBits },
            reference
        )[0];

        lower[buffered] = result.lowerInterval();
        upper[buffered] = result.upperInterval();
        // EUCLIDEAN: ‖v−c‖². Inner product and cosine: ⟨v,c⟩. One column, two meanings, decided by the quantizer's
        // own similarity — which must be the field's, or the corrective term would be the wrong quantity.
        add[buffered] = result.additionalCorrection();
        sum[buffered] = result.quantizedComponentSum();

        transpose(rawCodes, packed);
        System.arraycopy(packed, 0, codes, buffered * packedBytes, packedBytes);

        buffered++;
        if (buffered == blockSize) {
            flush();
        }
    }

    /** Writes whatever is buffered, so the last, partial block is not lost. Idempotent. */
    @Override
    public void finish() throws IOException {
        if (buffered > 0) {
            flush();
        }
    }

    /** Bytes a full block occupies — the stride the reader's positioning arithmetic assumes. */
    public long fixedBlockBytes() {
        return (long) blockSize * packedBytes + (long) blockSize * (3 * Float.BYTES + Integer.BYTES);
    }

    private void flush() throws IOException {
        writeFloats(lower, buffered);
        writeFloats(upper, buffered);
        writeFloats(add, buffered);
        for (int i = 0; i < buffered; i++) {
            out.writeInt(sum[i]);
        }
        out.writeBytes(codes, 0, buffered * packedBytes);
        buffered = 0;
    }

    private void writeFloats(float[] values, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            out.writeInt(Float.floatToIntBits(values[i]));
        }
    }

    /**
     * Scatter one vector's quantized values into bit planes.
     *
     * <p>Lucene's own routines, so the bit order matches the query's transposition exactly. Getting that wrong would
     * pair dimension zero of the query with some other dimension of the document and yield plausible nonsense, which
     * is the one failure here that no read-side check would catch.
     */
    private void transpose(byte[] raw, byte[] destination) {
        if (docBits == DOC_BITS_ONE) {
            OptimizedScalarQuantizer.packAsBinary(raw, destination);
        } else {
            transposeDibit(raw, destination);
        }
    }

    /**
     * Scatter two-bit values into two bit planes: the low bit of every dimension, then the high bit.
     *
     * <p>Written here because the Lucene this builds against has no {@code transposeDibit}. The layout is fixed by
     * what {@code Int4DotProduct.dibit} reads — two stripes of {@code packedBytes / 2}, and within a stripe dimension
     * {@code i} at bit {@code 7 - (i mod 8)} of byte {@code i / 8}. Most significant bit first, matching
     * {@code packAsBinary}, so the one- and two-bit paths agree on where a dimension lives.
     *
     * <p>Getting the order wrong here is the failure no read-side check catches: the kernel would pair one dimension
     * of the query with another of the document and return scores that are individually plausible.
     */
    private void transposeDibit(byte[] raw, byte[] destination) {
        // The buffer is reused between vectors and the bits below are OR'd in, so a stale bit would survive.
        Arrays.fill(destination, (byte) 0);
        int stripeSize = destination.length / 2;
        for (int i = 0; i < dimension; i++) {
            int value = raw[i] & 0x3;
            int byteIndex = i >>> 3;
            int bit = 7 - (i & 7);
            if ((value & 1) != 0) {
                destination[byteIndex] |= (byte) (1 << bit);
            }
            if ((value & 2) != 0) {
                destination[stripeSize + byteIndex] |= (byte) (1 << bit);
            }
        }
    }
}
