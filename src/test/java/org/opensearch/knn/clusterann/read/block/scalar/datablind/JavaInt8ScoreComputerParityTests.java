/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.index.FloatVectorValues;
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
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8BlockWriter;
import org.opensearch.knn.quantization.DataBlindInt8Quantizer;
import org.opensearch.knn.quantization.Int8DotProduct;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link JavaInt8ScoreComputer} behind {@link DataBlindInt8BlockScorer} must reproduce the scorer it replaced bit for
 * bit: same candidates, same scores, same block maximum. The reference is {@link LegacyDataBlindInt8BlockScorer}, a
 * verbatim copy of the pre-seam scorer. Blocks are written with the production writer, so the reader parses a genuine
 * layout; members include the query itself and scaled copies so the Euclidean clamp is exercised on both sides.
 */
class JavaInt8ScoreComputerParityTests {

    private static final int BLOCK_SIZE = 8;
    private static final int VECTOR_COUNT = 21;   // 2 full blocks and a short one
    private static final String FILE = "blocks";

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    static Stream<Arguments> cases() {
        List<Arguments> cases = new ArrayList<>();
        for (VectorSimilarityFunction sim : List.of(
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            VectorSimilarityFunction.COSINE
        )) {
            for (int dimension : new int[] { 1, 7, 8, 9, 64, 100, 768, 1537 }) {
                cases.add(Arguments.of(sim, dimension));
            }
        }
        return cases.stream();
    }

    @ParameterizedTest(name = "{0} dim={1}")
    @MethodSource("cases")
    void testScoreBlock_thenMatchesTheLegacyScorerBitForBit(VectorSimilarityFunction sim, int dimension) throws IOException {
        Random random = new Random(31L * dimension + sim.ordinal());
        float[] query = randomVector(random, dimension);
        Fixture fixture = fixture(dimension, query, random);
        DataBlindInt8ScanContext context = scanContext(query, dimension, sim);

        LegacyDataBlindInt8BlockScorer legacy = new LegacyDataBlindInt8BlockScorer(fixture.legacyReader, context, sim);
        DataBlindInt8ComputeContext computeContext = new DataBlindInt8ComputeContext(
            fixture.reader.codes(),
            fixture.reader.scale(),
            fixture.reader.sum(),
            fixture.reader.sqNorm(),
            dimension,
            context
        );
        DataBlindInt8BlockScorer current = new DataBlindInt8BlockScorer(fixture.reader, computeContext, JavaInt8ScoreComputer.INSTANCE);

        int compared = 0;
        for (int block = 0; block < fixture.reader.numBlocks(); block++) {
            position(fixture.legacyReader, block);
            position(fixture.reader, block);
            int count = fixture.reader.blockVectorCount();
            for (FixedBitSet accepted : acceptancePatterns(count, random)) {
                BlockVectorScorer.BlockCandidates expected = candidates();
                BlockVectorScorer.BlockCandidates actual = candidates();
                float expectedMax = legacy.scoreBlock(accepted, expected);
                float actualMax = current.scoreBlock(accepted, actual);

                assertEquals(expected.getSize(), actual.getSize(), "candidate count");
                for (int i = 0; i < expected.getSize(); i++) {
                    assertEquals(expected.getPositions()[i], actual.getPositions()[i], "position " + i);
                    assertEquals(
                        Float.floatToIntBits(expected.getScores()[i]),
                        Float.floatToIntBits(actual.getScores()[i]),
                        "score at position " + expected.getPositions()[i]
                    );
                }
                assertEquals(Float.floatToIntBits(expectedMax), Float.floatToIntBits(actualMax), "block max");
                compared += expected.getSize();
            }
        }
        assertTrue(compared > 0, "the fixture must score something");
    }

    @Test
    void testEuclidean_thenBothClampBranchesAreExercised() throws IOException {
        int dimension = 64;
        Random random = new Random(7);
        float[] query = randomVector(random, dimension);
        Fixture fixture = fixture(dimension, query, random);
        DataBlindInt8ScanContext context = scanContext(query, dimension, VectorSimilarityFunction.EUCLIDEAN);

        int clamped = 0;
        int unclamped = 0;
        for (int block = 0; block < fixture.reader.numBlocks(); block++) {
            position(fixture.reader, block);
            for (int index = 0; index < fixture.reader.blockVectorCount(); index++) {
                long unsignedDot = Int8DotProduct.unsignedDot(context.queryCode(), fixture.reader.codes(), index * dimension, dimension);
                long signedDot = unsignedDot - context.terms().offsetResidual() - (long) DataBlindInt8Quantizer.OFFSET * fixture.reader
                    .sum()[index];
                double dot = (double) signedDot * context.terms().scale() * fixture.reader.scale()[index];
                double s = context.terms().sqNorm() + fixture.reader.sqNorm()[index] - 2.0 * dot;
                if (s < 0) {
                    clamped++;
                } else {
                    unclamped++;
                }
            }
        }
        assertTrue(clamped > 0, "a member equal to the query must drive the distance below zero at least once");
        assertTrue(unclamped > 0, "random members must stay on the unclamped side");
    }

    // ---------------------------------------------------------------- fixtures

    /** Two readers over the same written blocks, one per scorer, so neither scorer disturbs the other's cursor. */
    private record Fixture(DataBlindInt8BlockReader reader, DataBlindInt8BlockReader legacyReader) {
    }

    private Fixture fixture(int dimension, float[] query, Random random) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        float[][] members = members(dimension, query, random);
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            new DataBlindInt8BlockWriter(out, BLOCK_SIZE, dimension).writeBlocks(new ArrayFloatVectorValues(members, dimension));
        }
        IndexInput first = directory.openInput(FILE, IOContext.DEFAULT);
        IndexInput second = directory.openInput(FILE, IOContext.DEFAULT);
        DataBlindInt8BlockReader reader = new DataBlindInt8BlockReader(BLOCK_SIZE, dimension);
        DataBlindInt8BlockReader legacyReader = new DataBlindInt8BlockReader(BLOCK_SIZE, dimension);
        reader.reset(first, VECTOR_COUNT);
        legacyReader.reset(second, VECTOR_COUNT);
        return new Fixture(reader, legacyReader);
    }

    /**
     * Random members with edge cases mixed in: the query itself (drives the Euclidean clamp), a scaled copy, a zero
     * vector (the quantizer's {@code maxAbs == 0} path), and one vector of large magnitude.
     */
    private static float[][] members(int dimension, float[] query, Random random) {
        float[][] members = new float[VECTOR_COUNT][];
        for (int m = 0; m < VECTOR_COUNT; m++) {
            members[m] = randomVector(random, dimension);
        }
        members[0] = query.clone();
        members[3] = scaled(query, 2.5f);
        members[5] = new float[dimension];
        members[VECTOR_COUNT - 1] = scaled(randomVector(random, dimension), 1000f);
        return members;
    }

    private static DataBlindInt8ScanContext scanContext(float[] query, int dimension, VectorSimilarityFunction sim) {
        DataBlindInt8Quantizer quantizer = new DataBlindInt8Quantizer(dimension);
        byte[] queryCode = new byte[dimension];
        DataBlindInt8Quantizer.Int8Code corrections = quantizer.encode(query, queryCode);
        return new DataBlindInt8ScanContext(query, queryCode, quantizer.queryTerms(corrections), sim);
    }

    private static void position(DataBlindInt8BlockReader reader, int block) throws IOException {
        assertTrue(reader.advance(block));
        reader.fetchBlock();
        reader.readBlockVectors();
    }

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

    private static float[] randomVector(Random random, int dimension) {
        float[] v = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            v[i] = (float) random.nextGaussian();
        }
        return v;
    }

    private static float[] scaled(float[] v, float factor) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * factor;
        }
        return out;
    }

    /** Minimal in-memory {@link FloatVectorValues} over a fixed array, read by position. */
    private static final class ArrayFloatVectorValues extends FloatVectorValues {
        private final float[][] vectors;
        private final int dimension;

        private ArrayFloatVectorValues(float[][] vectors, int dimension) {
            this.vectors = vectors;
            this.dimension = dimension;
        }

        @Override
        public float[] vectorValue(int ord) {
            return vectors[ord];
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return vectors.length;
        }

        @Override
        public ArrayFloatVectorValues copy() {
            return this;
        }
    }
}
