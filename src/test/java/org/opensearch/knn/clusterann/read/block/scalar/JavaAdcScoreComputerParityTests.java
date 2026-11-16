/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.quantization.Int4DotProduct;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link JavaAdcScoreComputer} behind {@link ScalarQuantizedBlockScorer} must reproduce the pre-change scorer bit for bit:
 * same candidates, same scores, same block maximum. The reference is {@link LegacyScalarQuantizedBlockScorer}, a
 * verbatim copy of that scorer kept in the test tree. Covers every code width the format writes, every similarity it
 * scores, an even and an odd dimension, full and short blocks, and dense through single-position acceptance.
 *
 * <p>Two fixture regimes: {@link Regime#WIDE} has intervals far from zero and small {@code ‖q−c‖²}, which drives the
 * EUCLIDEAN estimate negative and into the clamp; {@link Regime#NARROW} has intervals near zero and a large
 * {@code ‖q−c‖²}, which keeps it positive. A dedicated case asserts both branches were in fact taken.
 */
class JavaAdcScoreComputerParityTests {

    private static final int BLOCK_SIZE = 8;
    private static final int VECTOR_COUNT = 21;   // 2 full blocks and a short one
    private static final String FILE = "quantized";

    /** How the fixture's intervals and query corrections are drawn; see the class comment. */
    enum Regime {
        WIDE,
        NARROW
    }

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    static Stream<Arguments> cases() {
        List<Arguments> cases = new ArrayList<>();
        for (ScalarEncoding encoding : List.of(
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE,
            ScalarEncoding.DIBIT_QUERY_NIBBLE,
            ScalarEncoding.PACKED_NIBBLE
        )) {
            for (VectorSimilarityFunction sim : List.of(
                VectorSimilarityFunction.EUCLIDEAN,
                VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
                VectorSimilarityFunction.COSINE
            )) {
                for (int dimension : new int[] { 64, 37 }) {
                    for (Regime regime : Regime.values()) {
                        cases.add(Arguments.of(encoding, sim, dimension, regime));
                    }
                }
            }
        }
        return cases.stream();
    }

    @ParameterizedTest(name = "{0} {1} dim={2} {3}")
    @MethodSource("cases")
    void testScoreBlock_thenMatchesTheLegacyScorerBitForBit(
        ScalarEncoding encoding,
        VectorSimilarityFunction sim,
        int dimension,
        Regime regime
    ) throws IOException {
        assertParity(encoding, sim, dimension, regime);
    }

    /**
     * The EUCLIDEAN transform has two branches, the plain {@code 1/(1+d)} and the triangle-inequality clamp for a
     * negative estimate. Parity on one branch says nothing about the other, so both have to be seen to run.
     */
    @Test
    void testEuclidean_thenBothClampBranchesAreExercised() throws IOException {
        int clamped = 0;
        int plain = 0;
        for (ScalarEncoding encoding : List.of(
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE,
            ScalarEncoding.DIBIT_QUERY_NIBBLE,
            ScalarEncoding.PACKED_NIBBLE
        )) {
            for (int dimension : new int[] { 64, 37 }) {
                for (Regime regime : Regime.values()) {
                    int[] branches = assertParity(encoding, VectorSimilarityFunction.EUCLIDEAN, dimension, regime);
                    clamped += branches[0];
                    plain += branches[1];
                }
            }
        }
        assertTrue(clamped > 0, "no EUCLIDEAN estimate went negative, the clamp branch was never compared");
        assertTrue(plain > 0, "every EUCLIDEAN estimate went negative, the plain branch was never compared");
    }

    /**
     * Compares legacy and current over every block and acceptance pattern of one fixture. Returns how many scored
     * vectors took the EUCLIDEAN clamp branch and how many the plain one (both zero for other similarities).
     */
    private int[] assertParity(ScalarEncoding encoding, VectorSimilarityFunction sim, int dimension, Regime regime) throws IOException {
        Random random = new Random(31L * encoding.ordinal() + 7L * sim.ordinal() + dimension + 101L * regime.ordinal());
        ScalarQuantizedBlockReader reader = reader(encoding, dimension, random, regime);
        SQScanContext context = scanContext(encoding, dimension, sim, random, regime);
        int clamped = 0;
        int plain = 0;

        LegacyScalarQuantizedBlockScorer legacy = new LegacyScalarQuantizedBlockScorer(reader, context, encoding, sim);
        SQComputeContext computeContext = new SQComputeContext(
            reader.codes(),
            reader.lower(),
            reader.upper(),
            reader.addCor(),
            reader.sum(),
            encoding,
            encoding.getDocPackedLength(dimension),
            context
        );
        ScalarQuantizedBlockScorer current = new ScalarQuantizedBlockScorer(reader, computeContext, JavaAdcScoreComputer.INSTANCE);

        int blocks = reader.numBlocks();
        for (int block = 0; block < blocks; block++) {
            reader.advance(block);
            reader.fetchBlock();
            reader.readBlockVectors();
            int count = reader.blockVectorCount();

            for (FixedBitSet validPos : acceptancePatterns(count, random)) {
                BlockVectorScorer.BlockCandidates expected = candidates();
                BlockVectorScorer.BlockCandidates actual = candidates();

                float expectedMax = legacy.scoreBlock(validPos, expected);
                float actualMax = current.scoreBlock(validPos, actual);

                String where = encoding + " " + sim + " dim=" + dimension + " block=" + block + " accepted=" + validPos.cardinality();
                assertEquals(expected.getSize(), actual.getSize(), where + ": candidate count");
                assertEquals(Float.floatToIntBits(expectedMax), Float.floatToIntBits(actualMax), where + ": block maximum");
                for (int i = 0; i < expected.getSize(); i++) {
                    assertEquals(expected.getPositions()[i], actual.getPositions()[i], where + ": position " + i);
                    assertEquals(
                        Float.floatToIntBits(expected.getScores()[i]),
                        Float.floatToIntBits(actual.getScores()[i]),
                        where + ": score at position " + expected.getPositions()[i]
                    );
                    if (sim == VectorSimilarityFunction.EUCLIDEAN) {
                        if (preClampDistance(context, reader, encoding, expected.getPositions()[i]) < 0f) {
                            clamped++;
                        } else {
                            plain++;
                        }
                    }
                }
            }
        }
        return new int[] { clamped, plain };
    }

    /**
     * The EUCLIDEAN estimate before the clamp, {@code ‖q−c‖² + ‖v−c‖² − 2⟨q−c, v−c⟩}, computed here from the fixture
     * so a scored vector can be told to have taken one branch or the other. An oracle for branch classification only;
     * the parity assertions above do not depend on it.
     */
    private static float preClampDistance(SQScanContext q, ScalarQuantizedBlockReader r, ScalarEncoding encoding, int index) {
        int dimension = q.query().length;
        int packedBytes = encoding.getDocPackedLength(dimension);
        int offset = index * packedBytes;
        float rawDot = encoding.isAsymmetric()
            ? (encoding.getDocBitsPerDim() == 1
                ? Int4DotProduct.bit(q.transposed(), r.codes(), offset, packedBytes)
                : Int4DotProduct.dibit(q.transposed(), r.codes(), offset, packedBytes))
            : Int4DotProduct.nibble(q.transposed(), r.codes(), offset, packedBytes);
        float step = 1f / ((1 << encoding.getDocBitsPerDim()) - 1);
        float docScale = (r.upper()[index] - r.lower()[index]) * step;
        float score = r.lower()[index] * (q.lower() * dimension) + q.lower() * docScale * r.sum()[index] + r.lower()[index] * (q.scale() * q
            .componentSum()) + docScale * q.scale() * rawDot;
        return q.correction() + r.addCor()[index] - 2f * score;
    }

    // ---------------------------------------------------------------- fixtures

    /** Everything accepted, nothing, every other position, a random subset, only the last position. */
    private static List<FixedBitSet> acceptancePatterns(int count, Random random) {
        FixedBitSet all = new FixedBitSet(BLOCK_SIZE);
        all.set(0, count);
        FixedBitSet none = new FixedBitSet(BLOCK_SIZE);
        FixedBitSet odd = new FixedBitSet(BLOCK_SIZE);
        for (int i = 1; i < count; i += 2) {
            odd.set(i);
        }
        FixedBitSet subset = new FixedBitSet(BLOCK_SIZE);
        for (int i = 0; i < count; i++) {
            if (random.nextInt(3) == 0) {
                subset.set(i);
            }
        }
        FixedBitSet last = new FixedBitSet(BLOCK_SIZE);
        last.set(count - 1);
        return List.of(all, none, odd, subset, last);
    }

    private static BlockVectorScorer.BlockCandidates candidates() {
        BlockVectorScorer.BlockCandidates out = new BlockVectorScorer.BlockCandidates();
        out.growNoCopy(BLOCK_SIZE);
        Arrays.fill(out.getScores(), Float.NaN);
        return out;
    }

    private static SQScanContext scanContext(
        ScalarEncoding encoding,
        int dimension,
        VectorSimilarityFunction sim,
        Random random,
        Regime regime
    ) {
        float[] query = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            query[i] = (float) random.nextGaussian();
        }
        // Shape follows the encoding, as prepareScan builds it: query planes for a transposed document code, one
        // 0..15 code per (padded) dimension for packed nibbles.
        byte[] transposed;
        if (encoding.isAsymmetric()) {
            transposed = new byte[encoding.getQueryPackedLength(dimension)];
            random.nextBytes(transposed);
        } else {
            transposed = new byte[encoding.getDiscreteDimensions(dimension)];
            for (int i = 0; i < transposed.length; i++) {
                transposed[i] = (byte) random.nextInt(16);
            }
        }
        return new SQScanContext(
            query,
            ScanParams.DEFAULT_QUERY_BITS,
            0.5f + random.nextFloat(),               // centroidNormSq
            transposed,
            regime == Regime.WIDE ? -1f - random.nextFloat() : -0.01f * random.nextFloat(),       // lower
            regime == Regime.WIDE ? 0.01f + random.nextFloat() * 0.2f : 0.001f * random.nextFloat(), // scale
            random.nextInt(15 * dimension),          // componentSum
            // ‖q−c‖² for EUCLIDEAN, so kept positive: small enough to be overrun by the estimate in WIDE, large in NARROW
            regime == Regime.WIDE ? 0.5f + random.nextFloat() * 3f : 50f + random.nextFloat() * 10f,
            sim
        );
    }

    private ScalarQuantizedBlockReader reader(ScalarEncoding encoding, int dimension, Random random, Regime regime) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        int packedBytes = encoding.getDocPackedLength(dimension);
        write(directory, packedBytes, random, regime);
        ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(BLOCK_SIZE, dimension, encoding);
        reader.reset(directory.openInput(FILE, IOContext.DEFAULT), VECTOR_COUNT);
        return reader;
    }

    /** The unpadded block layout: per block, lower | upper | add | sum for the vectors it holds, then their codes. */
    private static void write(Directory directory, int packedBytes, Random random, Regime regime) throws IOException {
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int first = 0; first < VECTOR_COUNT; first += BLOCK_SIZE) {
                int count = Math.min(BLOCK_SIZE, VECTOR_COUNT - first);
                float[] lower = new float[count];
                for (int i = 0; i < count; i++) {
                    lower[i] = regime == Regime.WIDE ? -2f + random.nextFloat() : -0.01f * random.nextFloat();
                    out.writeInt(Float.floatToIntBits(lower[i]));
                }
                for (int i = 0; i < count; i++) {
                    float width = regime == Regime.WIDE ? 0.5f + random.nextFloat() * 2f : 0.001f + random.nextFloat() * 0.02f;
                    out.writeInt(Float.floatToIntBits(lower[i] + width));                            // upper > lower
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(random.nextFloat() * 4f));                    // add ≥ 0
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(random.nextInt(packedBytes * 8));                                   // Σᵢaᵢ
                }
                byte[] codes = new byte[count * packedBytes];
                random.nextBytes(codes);
                out.writeBytes(codes, codes.length);
            }
        }
    }
}
