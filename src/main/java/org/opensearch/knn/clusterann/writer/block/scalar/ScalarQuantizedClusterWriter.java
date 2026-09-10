package org.opensearch.knn.clusterann.writer.block.scalar;

import org.opensearch.knn.clusterann.reader.block.scalar.ScalarEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.reader.Centroid;
import org.opensearch.knn.clusterann.writer.ClusterWriter;
import org.opensearch.knn.clusterann.writer.Posting;
import org.opensearch.knn.clusterann.writer.VectorSource;

import java.io.IOException;

/**
 * Writes one scalar-quantized posting — the inverse of {@code ScalarQuantizedCluster}'s layout:
 *
 * <pre>
 * ordinals          clusterSize ints     // ascending ‖c−v‖, primary and SOAR entries mixed together
 * soarBitset        ceil(clusterSize/8)  // one bit per entry, set = this entry is a SOAR copy
 * sortedDistances   clusterSize floats   // ‖c−v‖ ascending, parallel to ordinals
 * block 0           lower[m] | upper[m] | add[m] | sum[m] | codes[m × packedBytes]
 * block 1           …
 * </pre>
 *
 * <p>The header's size follows from the entry count alone, which is what lets the reader compute where block zero
 * begins without reading anything — so the three columns are fixed width and in this order, and nothing
 * variable-length may be added between them.
 *
 * <p>Two invariants live here and cannot be checked from the read side:
 *
 * <ul>
 *   <li><b>{@code ordinals} and {@code sortedDistances} ascend together by {@code ‖c−v‖}</b>, which is what makes a
 *       scan's early termination sound. {@link Posting} enforces it before a byte is written.
 *   <li><b>The three columns are parallel</b>: entry {@code i} of each describes the same vector, and so does block
 *       position {@code i} of the block payload. A permutation applied to one and not the others would produce
 *       scores that are individually plausible and collectively wrong.
 * </ul>
 */
public final class ScalarQuantizedClusterWriter implements ClusterWriter {

    private final int blockSize;
    private final int dimension;
    private final ScalarEncoding encoding;
    private final OptimizedScalarQuantizer quantizer;
    private final VectorSimilarityFunction similarity;

    public ScalarQuantizedClusterWriter(
        int blockSize,
        int dimension,
        ScalarEncoding encoding,
        OptimizedScalarQuantizer quantizer,
        VectorSimilarityFunction similarity
    ) {
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.encoding = encoding;
        this.quantizer = quantizer;
        this.similarity = similarity;
    }

    @Override
    public void write(IndexOutput clap, Posting posting, Centroid reference, VectorSource vectors) throws IOException {
        writeHeader(clap, posting);

        ScalarQuantizedBlockWriter blocks = new ScalarQuantizedBlockWriter(clap, blockSize, dimension, encoding, quantizer);
        for (int ordinal : posting.ordinals()) {
            blocks.addVector(vectors.vector(ordinal), reference.vector());
        }
        blocks.finish();
    }

    /** The field's similarity, which decides what the {@code add} column means. Exposed for tests to assert against. */
    public VectorSimilarityFunction similarity() {
        return similarity;
    }

    /**
     * The three fixed-width columns, in the order the reader consumes them.
     *
     * <p>The SOAR bitset is written whole — {@code ceil(size/8)} bytes — even when no entry is a spill, because the
     * reader's arithmetic for where block zero begins assumes it is there. An absent bitset would shift every block.
     */
    private static void writeHeader(IndexOutput clap, Posting posting) throws IOException {
        for (int ordinal : posting.ordinals()) {
            clap.writeInt(ordinal);
        }

        writeSoarBitset(clap, posting.soar(), posting.size());

        for (float distance : posting.distances()) {
            clap.writeInt(Float.floatToIntBits(distance));
        }
    }

    /**
     * One bit per entry, least significant bit of each byte first, so byte {@code b} carries entries
     * {@code 8b .. 8b+7}.
     *
     * <p>Written a byte at a time rather than through the bitset's own longs, because the region is
     * {@code ceil(size/8)} bytes and a long-based write would round up to eight and shift block zero.
     */
    private static void writeSoarBitset(IndexOutput clap, FixedBitSet soar, int size) throws IOException {
        int bytes = (size + 7) / 8;
        for (int b = 0; b < bytes; b++) {
            int packed = 0;
            for (int bit = 0; bit < 8; bit++) {
                int entry = b * 8 + bit;
                if (entry < size && soar.get(entry)) {
                    packed |= 1 << bit;
                }
            }
            clap.writeByte((byte) packed);
        }
    }
}
