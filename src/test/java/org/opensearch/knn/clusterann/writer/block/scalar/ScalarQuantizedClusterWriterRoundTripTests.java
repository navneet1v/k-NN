package org.opensearch.knn.clusterann.writer.block.scalar;

import org.opensearch.knn.clusterann.reader.block.scalar.ScalarEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.reader.Centroid;
import org.opensearch.knn.clusterann.reader.block.scalar.ScalarQuantizedBlockReader;
import org.opensearch.knn.clusterann.writer.Posting;
import org.opensearch.knn.clusterann.writer.VectorSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Round-trip tests for the posting layout: write it, then read it back through the reader that has to consume it.
 *
 * <p>These exist because the write side owns invariants the read side cannot check. A posting sorted the wrong way,
 * a bitset that is a byte too long, columns permuted out of step — none of it makes a segment unreadable. It makes
 * scores plausible and wrong, or silently drops candidates. Asserting against the real reader is the only thing that
 * catches that.
 */
class ScalarQuantizedClusterWriterRoundTripTests {

    private static final String FILE = "posting";
    private static final int DIMENSION = 64;
    private static final int BLOCK_SIZE = 4;

    /**
     * The header columns survive the round trip: same ordinals, same distances, same SOAR bits, and block zero begins
     * exactly where the reader's arithmetic says it does.
     */
    @ParameterizedTest
    @ValueSource(ints = { 1, 2 })
    void headerRoundTrips(int docBits) throws IOException {
        ScalarEncoding encoding = ScalarEncoding.fromNumBits(docBits);
        int[] ordinals = { 7, 3, 11, 42, 5, 19, 2 };
        float[] distances = { 0.1f, 0.2f, 0.25f, 0.4f, 0.55f, 0.9f, 1.3f };
        FixedBitSet soar = new FixedBitSet(ordinals.length);
        soar.set(1);
        soar.set(4);
        Posting posting = new Posting(0, ordinals, distances, soar);

        try (Directory directory = new ByteBuffersDirectory()) {
            write(directory, posting, encoding);
            try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
                int[] readOrdinals = new int[ordinals.length];
                in.readInts(readOrdinals, 0, ordinals.length);
                assertEquals(ordinals.length, readOrdinals.length);
                for (int i = 0; i < ordinals.length; i++) {
                    assertEquals(ordinals[i], readOrdinals[i], "ordinal " + i);
                }

                int soarBytes = (ordinals.length + 7) / 8;
                byte[] readSoar = new byte[soarBytes];
                in.readBytes(readSoar, 0, soarBytes);
                for (int entry = 0; entry < ordinals.length; entry++) {
                    boolean set = (readSoar[entry / 8] & (1 << (entry % 8))) != 0;
                    assertEquals(soar.get(entry), set, "soar bit " + entry);
                }

                for (int i = 0; i < distances.length; i++) {
                    assertEquals(distances[i], Float.intBitsToFloat(in.readInt()), "distance " + i);
                }

                // The reader computes this rather than reading it, so the two have to agree exactly.
                assertEquals(posting.headerBytes(), in.getFilePointer());
            }
        }
    }

    /** Every block the writer produced is reachable and decodes, including the short last one. */
    @ParameterizedTest
    @ValueSource(ints = { 1, 2 })
    void blocksAreReadable(int docBits) throws IOException {
        ScalarEncoding encoding = ScalarEncoding.fromNumBits(docBits);
        int count = 7; // two full blocks of four, then a block of three
        Posting posting = sequentialPosting(count);

        try (Directory directory = new ByteBuffersDirectory()) {
            write(directory, posting, encoding);
            try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
                IndexInput blocks = in.slice("blocks", posting.headerBytes(), in.length() - posting.headerBytes());
                ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(blocks, BLOCK_SIZE, count, DIMENSION, encoding);

                assertEquals(2, reader.numBlocks());
                int seen = 0;
                for (int block = 0; block < reader.numBlocks(); block++) {
                    assertTrue(reader.advance(block), "block " + block);
                    reader.fetchBlock();
                    reader.readBlockVectors();
                    seen += reader.blockVectorCount();
                }
                assertEquals(count, seen, "every vector must land in exactly one block");
            }
        }
    }

    /** A posting whose distances descend is rejected before a byte is written — the invariant no reader can check. */
    @Test
    void unsortedPostingIsRejected() {
        assertThrows(
            IllegalArgumentException.class,
            () -> new Posting(0, new int[] { 1, 2 }, new float[] { 0.9f, 0.1f }, new FixedBitSet(2))
        );
    }

    private static Posting sequentialPosting(int count) {
        int[] ordinals = new int[count];
        float[] distances = new float[count];
        for (int i = 0; i < count; i++) {
            ordinals[i] = i;
            distances[i] = 0.1f * (i + 1);
        }
        return new Posting(0, ordinals, distances, new FixedBitSet(count));
    }

    private static void write(Directory directory, Posting posting, ScalarEncoding encoding) throws IOException {
        VectorSimilarityFunction similarity = VectorSimilarityFunction.EUCLIDEAN;
        ScalarQuantizedClusterWriter writer = new ScalarQuantizedClusterWriter(
            BLOCK_SIZE,
            DIMENSION,
            encoding,
            new OptimizedScalarQuantizer(similarity),
            similarity
        );
        assertEquals(similarity, writer.similarity());

        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            writer.write(out, posting, centroid(), vectors(posting));
        }
    }

    private static Centroid centroid() {
        float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = 0.01f * i;
        }
        float normSq = 0f;
        for (float value : vector) {
            normSq += value * value;
        }
        return new Centroid(vector, normSq);
    }

    /** Distinct, non-degenerate vectors, so the quantizer produces a real interval rather than a collapsed one. */
    private static VectorSource vectors(Posting posting) {
        int max = 0;
        for (int ordinal : posting.ordinals()) {
            max = Math.max(max, ordinal);
        }
        List<float[]> vectors = new ArrayList<>();
        for (int ord = 0; ord <= max; ord++) {
            float[] vector = new float[DIMENSION];
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = (float) Math.sin(0.37 * i + ord);
            }
            vectors.add(vector);
        }
        return VectorSource.fromList(vectors, null, DIMENSION);
    }
}
