/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.ScanParams;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link ScalarQuantizedBlockScorer} (JUnit 5).
 */
class ScalarQuantizedBlockScorerTests {

    /** 1-bit doc codes: the only width the kernel supports alongside dibit, and the cheapest to lay out. */
    private static final ScalarEncoding ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
    private static final int DIMENSION = 64;

    private static final int BLOCK_SIZE = 4;
    private static final int VECTOR_COUNT = 10;
    private static final String FILE = "quantized";

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    @Test
    void testReader_thenLendsBackTheSameInstance() throws IOException {
        // given
        ScalarQuantizedBlockReader reader = reader();

        // when / then
        assertSame(reader, scorer(reader, VectorSimilarityFunction.EUCLIDEAN).reader());
    }

    // ---------------------------------------------------------------- output contract

    /**
     * One candidate per accepted position, in ascending order. Getting this wrong is invisible to a caller that
     * only looks at the first hit, so it is asserted on the whole buffer.
     */
    @ParameterizedTest(name = "accepting {0} positions")
    @ValueSource(ints = { 1, 2, 3, 4 })
    void testScoreBlock_thenYieldsOneCandidatePerAcceptedPosition(int accepted) throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.EUCLIDEAN);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        for (int i = 0; i < accepted; i++) {
            validPos.set(i);
        }
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        scorer.scoreBlock(validPos, out);

        // then
        assertEquals(accepted, out.getSize(), "every accepted position must produce a candidate");
        int[] expectedPositions = new int[accepted];
        for (int i = 0; i < accepted; i++) {
            expectedPositions[i] = i;
        }
        assertArrayEquals(expectedPositions, Arrays.copyOf(out.getPositions(), accepted));
    }

    /** Non-contiguous acceptance: the candidates carry the positions asked for, not 0..n. */
    @Test
    void testScoreBlock_whenAcceptanceIsSparse_thenCarriesThosePositions() throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.EUCLIDEAN);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(1);
        validPos.set(3);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        scorer.scoreBlock(validPos, out);

        // then
        assertEquals(2, out.getSize());
        assertArrayEquals(new int[] { 1, 3 }, Arrays.copyOf(out.getPositions(), 2));
    }

    /**
     * The returned value is what the caller prunes on, so it has to be the maximum of the scores handed out —
     * not of some intermediate quantity behind them.
     */
    @ParameterizedTest(name = "{0}")
    @EnumSource(value = VectorSimilarityFunction.class, mode = EnumSource.Mode.EXCLUDE, names = "DOT_PRODUCT")
    void testScoreBlock_thenReturnsTheMaximumOfTheScoresItAppended(VectorSimilarityFunction sim) throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), sim);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(0, BLOCK_SIZE);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        float maxScore = scorer.scoreBlock(validPos, out);

        // then
        float expected = -Float.MAX_VALUE;
        for (int i = 0; i < out.getSize(); i++) {
            expected = Math.max(expected, out.getScores()[i]);
            assertTrue(out.getScores()[i] >= 0f, sim + " similarities are non-negative, got " + out.getScores()[i]);
        }
        assertEquals(BLOCK_SIZE, out.getSize());
        assertEquals(expected, maxScore, sim + " must return the best score it produced");
    }

    /** Nothing wanted means no candidates and a maximum that cannot beat any real threshold. */
    @Test
    void testScoreBlock_whenNothingIsAccepted_thenYieldsNoCandidates() throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.EUCLIDEAN);
        positionOn(scorer, 0);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        float maxScore = scorer.scoreBlock(new FixedBitSet(BLOCK_SIZE), out);

        // then
        assertEquals(0, out.getSize(), "a block with nothing accepted produces nothing");
        assertEquals(Float.NEGATIVE_INFINITY, maxScore, "an empty block must not look competitive");
    }

    /**
     * The bitset scan asks for the next set bit after the current one, and Lucene asserts that argument is
     * inside the set — so the highest position being accepted is its own case. Assertions are on in this JVM.
     */
    @ParameterizedTest(name = "block {0}")
    @ValueSource(ints = { 0, 2 })
    void testScoreBlock_whenTheLastPositionIsAccepted_thenStillTerminates(int block) throws IOException {
        // given — block 2 is the short last one, so its final position differs from a full block's
        ScalarQuantizedBlockReader reader = reader();
        BlockVectorScorer scorer = scorer(reader, VectorSimilarityFunction.EUCLIDEAN);
        positionOn(scorer, block);
        int count = reader.blockVectorCount();
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(count - 1);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        scorer.scoreBlock(validPos, out);

        // then
        assertEquals(1, out.getSize());
        assertEquals(count - 1, out.getPositions()[0]);
    }

    // ---------------------------------------------------------------- similarity branches

    /** Cosine scores through the same code dot product, and every score it hands out is usable. */
    @Test
    void testScoreBlock_whenSimilarityIsCosine_thenEveryScoreIsUsable() throws IOException {
        // given
        BlockVectorScorer cosine = scorer(reader(), VectorSimilarityFunction.COSINE);
        positionOn(cosine, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(0, BLOCK_SIZE);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        float maxScore = cosine.scoreBlock(validPos, out);

        // then
        assertEquals(BLOCK_SIZE, out.getSize());
        for (int i = 0; i < out.getSize(); i++) {
            float score = out.getScores()[i];
            assertFalse(Float.isNaN(score), "position " + i + " scored NaN");
            assertTrue(score >= 0f, "position " + i + " scored " + score);
            assertTrue(score <= maxScore, "position " + i + " exceeds the reported maximum");
        }
    }

    /** DOT_PRODUCT is not a similarity this format writes, so a segment claiming it cannot be scored. */
    @Test
    void testScoreBlock_whenSimilarityIsDotProduct_thenThrows() throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.DOT_PRODUCT);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(0, BLOCK_SIZE);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when / then
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> scorer.scoreBlock(validPos, out));
        assertTrue(e.getMessage().contains("MAXIMUM_INNER_PRODUCT"), e.getMessage());
    }

    /** Euclidean maps a distance through 1/(1+d), so its similarities are bounded — the others are not. */
    @Test
    void testScoreBlock_whenSimilarityIsEuclidean_thenScoresAreBoundedByOne() throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.EUCLIDEAN);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(0, BLOCK_SIZE);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        scorer.scoreBlock(validPos, out);

        // then
        for (int i = 0; i < out.getSize(); i++) {
            assertTrue(out.getScores()[i] >= 0f && out.getScores()[i] <= 1f, "score " + i + " was " + out.getScores()[i]);
        }
    }

    // ---------------------------------------------------------------- unsupported shapes

    @ParameterizedTest(name = "{0} is null")
    @ValueSource(strings = { "validPos", "out" })
    void testScoreBlock_whenAnArgumentIsNull_thenThrows(String nullArgument) throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.EUCLIDEAN);
        positionOn(scorer, 0);
        FixedBitSet validPos = "validPos".equals(nullArgument) ? null : new FixedBitSet(BLOCK_SIZE);
        BlockVectorScorer.BlockCandidates out = "out".equals(nullArgument) ? null : candidates();

        // when
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> scorer.scoreBlock(validPos, out));

        // then
        assertTrue(e.getMessage().contains("must not be null"), e.getMessage());
    }

    /**
     * The scorer takes the query width from the prepared context and does not check it — the kernels assume four
     * bits, so whoever prepares the context owns that constraint now. This pins the width the kernels were written
     * for, so a change to it cannot pass unnoticed.
     */
    @Test
    void testScoreBlock_whenQueryWidthIsTheDefault_thenScores() throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(), VectorSimilarityFunction.EUCLIDEAN, ENCODING, ScanParams.DEFAULT_QUERY_BITS);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(0);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        scorer.scoreBlock(validPos, out);

        // then
        assertEquals(1, out.getSize());
        assertEquals(4, ScanParams.DEFAULT_QUERY_BITS, "the kernels assume this width");
    }

    /**
     * A width with no kernel is refused when the encoding is handed over, not when scoring first reaches for one: the
     * reader stores any width happily, so the constructor is the only place that can catch it early. SEVEN_BIT is the
     * case at hand - 7-bit codes occupy a byte per dimension like 8-bit, but nothing has been verified to score them.
     */
    @Test
    void testConstructor_whenDocWidthHasNoKernel_thenThrows() {
        // given / when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> scorer(reader(ScalarEncoding.SEVEN_BIT), VectorSimilarityFunction.EUCLIDEAN, ScalarEncoding.SEVEN_BIT)
        );

        // then
        assertTrue(e.getMessage().contains("SEVEN_BIT"), e.getMessage());
    }

    /** Every width that does have a kernel is accepted: the two transposed ones and the packed nibble. */
    @ParameterizedTest(name = "{0}")
    @EnumSource(value = ScalarEncoding.class, names = { "SINGLE_BIT_QUERY_NIBBLE", "DIBIT_QUERY_NIBBLE", "PACKED_NIBBLE" })
    void testConstructor_whenDocWidthHasAKernel_thenScores(ScalarEncoding encoding) throws IOException {
        // given
        BlockVectorScorer scorer = scorer(reader(encoding), VectorSimilarityFunction.EUCLIDEAN, encoding);
        positionOn(scorer, 0);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        validPos.set(0);
        BlockVectorScorer.BlockCandidates out = candidates();

        // when
        scorer.scoreBlock(validPos, out);

        // then
        assertEquals(1, out.getSize());
    }

    // ---------------------------------------------------------------- helpers

    private static BlockVectorScorer.BlockCandidates candidates() {
        BlockVectorScorer.BlockCandidates out = new BlockVectorScorer.BlockCandidates();
        out.growNoCopy(BLOCK_SIZE);
        return out;
    }

    /** Loads {@code block} so the scorer has real corrections and codes to read. */
    private static void positionOn(BlockVectorScorer scorer, int block) throws IOException {
        assertTrue(scorer.reader().advance(block));
        scorer.reader().fetchBlock();
        scorer.reader().readBlockVectors();
    }

    private ScalarQuantizedBlockReader reader() throws IOException {
        return reader(ENCODING);
    }

    private ScalarQuantizedBlockReader reader(ScalarEncoding encoding) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        int packedBytes = encoding.getDocPackedLength(DIMENSION);
        write(directory, packedBytes);
        IndexInput in = directory.openInput(FILE, IOContext.DEFAULT);
        return new ScalarQuantizedBlockReader(in, BLOCK_SIZE, VECTOR_COUNT, DIMENSION, encoding);
    }

    private static ScalarQuantizedBlockScorer scorer(ScalarQuantizedBlockReader reader, VectorSimilarityFunction sim) {
        return scorer(reader, sim, ENCODING);
    }

    private static ScalarQuantizedBlockScorer scorer(
        ScalarQuantizedBlockReader reader,
        VectorSimilarityFunction sim,
        ScalarEncoding encoding
    ) {
        return scorer(reader, sim, encoding, ScanParams.DEFAULT_QUERY_BITS);
    }

    private static ScalarQuantizedBlockScorer scorer(
        ScalarQuantizedBlockReader reader,
        VectorSimilarityFunction sim,
        ScalarEncoding encoding,
        int queryBits
    ) {
        return new ScalarQuantizedBlockScorer(reader, scanContext(encoding, queryBits), encoding, sim);
    }

    /**
     * A prepared query, built here rather than by quantising one.
     *
     * <p>The scorer no longer quantises: it takes the already-prepared {@link SQScanContext}, so a test supplies the
     * quantised query directly. That is the point of the split — the arithmetic can be driven with chosen values
     * instead of whatever {@code OptimizedScalarQuantizer} happens to produce.
     */
    private static SQScanContext scanContext(ScalarEncoding encoding, int queryBits) {
        float[] query = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            query[i] = (i % 7) * 0.25f - 0.5f;
        }
        normalise(query);

        // Shape follows the encoding, as it does in ScalarQuantizedCluster: transposed query planes for an asymmetric
        // encoding, one code per dimension for a symmetric one, which is what Lucene's whole-code kernels require.
        byte[] transposed = new byte[encoding.isAsymmetric() ? encoding.getQueryPackedLength(DIMENSION) : DIMENSION];
        for (int i = 0; i < transposed.length; i++) {
            transposed[i] = (byte) ((0x33 + i) & (encoding.isAsymmetric() ? 0xFF : 0x0F));
        }

        return new SQScanContext(
            query,
            queryBits,
            1.0f,        // centroidNormSq
            transposed,
            -0.75f,      // lower
            0.05f,       // scale
            42f,         // componentSum
            2.5f         // correction
        );
    }

    private static void normalise(float[] vector) {
        double sumOfSquares = 0;
        for (float value : vector) {
            sumOfSquares += (double) value * value;
        }
        float norm = (float) Math.sqrt(sumOfSquares);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= norm;
        }
    }

    /**
     * The unpadded block layout the reader expects: per block, the four correction arrays for the vectors that
     * block holds, then their packed codes. Values are arbitrary but varied, so scores differ per position.
     */
    private static void write(Directory directory, int packedBytes) throws IOException {
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int first = 0; first < VECTOR_COUNT; first += BLOCK_SIZE) {
                int count = Math.min(BLOCK_SIZE, VECTOR_COUNT - first);
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(-1.0f - (first + i) * 0.1f));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(1.0f + (first + i) * 0.1f));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits((first + i) * 0.05f));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(first + i);
                }
                for (int i = 0; i < count; i++) {
                    for (int b = 0; b < packedBytes; b++) {
                        out.writeByte((byte) (0x5A + first + i + b));
                    }
                }
            }
        }
    }
}
