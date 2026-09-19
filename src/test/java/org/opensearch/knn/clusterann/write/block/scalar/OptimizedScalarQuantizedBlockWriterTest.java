/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.block.scalar;

import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer.QuantizationResult;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Unit test for the {@code .clap} block codec: quantizes a bounded source and checks the SoA block bytes. */
class OptimizedScalarQuantizedBlockWriterTest {

    private static final int DIMENSION = 2;
    private static final int BLOCK_SIZE = 2;
    private static final int DOC_BITS = 1; // SINGLE_BIT_QUERY_NIBBLE
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;
    private static final float[] CENTROID = { 0.5f, 0.5f };
    private static final ScalarEncoding ENCODING = ScalarEncoding.fromNumBits(DOC_BITS);
    private static final int DISCRETE_DIMS = ENCODING.getDiscreteDimensions(DIMENSION);
    private static final int CODE_LENGTH = ENCODING.getDocPackedLength(DISCRETE_DIMS);

    @Test
    void writesPartialLastBlockWhenCountNotMultipleOfBlockSize() throws IOException {
        // 3 vectors, blockSize 2 -> a full block of 2 then a partial block of 1.
        final float[][] vectors = { { 0f, 0f }, { 1f, 0f }, { 0f, 1f } };
        final Expected[] expected = reference(vectors);

        try (Directory dir = new ByteBuffersDirectory()) {
            writeToFile(dir, vectors);
            try (IndexInput in = dir.openInput("blk", IOContext.DEFAULT)) {
                assertBlock(in, expected, 0, BLOCK_SIZE); // full block: v0, v1
                assertBlock(in, expected, BLOCK_SIZE, 1); // partial block: v2
                assertEquals(in.length(), in.getFilePointer(), "no trailing bytes");
            }
        }
    }

    @Test
    void writesOnlyFullBlocksWhenCountIsMultipleOfBlockSize() throws IOException {
        // 4 vectors, blockSize 2 -> two full blocks; the trailing writeBlock() must be a no-op.
        final float[][] vectors = { { 0f, 0f }, { 1f, 0f }, { 0f, 1f }, { 1f, 1f } };
        final Expected[] expected = reference(vectors);

        try (Directory dir = new ByteBuffersDirectory()) {
            writeToFile(dir, vectors);
            try (IndexInput in = dir.openInput("blk", IOContext.DEFAULT)) {
                assertBlock(in, expected, 0, BLOCK_SIZE);
                assertBlock(in, expected, BLOCK_SIZE, BLOCK_SIZE);
                assertEquals(in.length(), in.getFilePointer(), "no trailing bytes; final writeBlock() is a no-op");
            }
        }
    }

    @Test
    void writesNothingWhenSourceIsEmpty() throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            writeToFile(dir, new float[0][]);
            assertEquals(0L, dir.fileLength("blk"), "an empty source writes no bytes");
        }
    }

    /** Writes {@code vectors} as blocks to {@code blk} in {@code dir}. */
    private static void writeToFile(final Directory dir, final float[][] vectors) throws IOException {
        try (IndexOutput out = dir.createOutput("blk", IOContext.DEFAULT)) {
            // clones: the writer centers each vector in place per the writeBlocks contract.
            new OptimizedScalarQuantizedBlockWriter(out, BLOCK_SIZE, DIMENSION, new OptimizedScalarQuantizer(METRIC), ENCODING, CENTROID)
                .writeBlocks(FloatVectorValues.fromFloats(Arrays.stream(vectors).map(float[]::clone).toList(), DIMENSION));
        }
    }

    /** Quantizes each vector independently (then packs) to compare the emitted columns and codes against. */
    private static Expected[] reference(final float[][] vectors) {
        final Expected[] expected = new Expected[vectors.length];
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(METRIC);
        for (int i = 0; i < vectors.length; i++) {
            final byte[] quantized = new byte[DISCRETE_DIMS];
            final QuantizationResult terms = quantizer.scalarQuantize(vectors[i].clone(), quantized, ENCODING.getBits(), CENTROID);
            final byte[] code = new byte[CODE_LENGTH];
            OptimizedScalarQuantizer.packAsBinary(quantized, code); // SINGLE_BIT_QUERY_NIBBLE
            expected[i] = new Expected(
                terms.lowerInterval(),
                terms.upperInterval(),
                terms.additionalCorrection(),
                terms.quantizedComponentSum(),
                code
            );
        }
        return expected;
    }

    /** Reads one block and asserts its columns, then its packed codes, match the reference. */
    private static void assertBlock(final IndexInput in, final Expected[] expected, final int base, final int count) throws IOException {
        for (int j = 0; j < count; j++) {
            assertEquals(expected[base + j].lower, Float.intBitsToFloat(in.readInt()), "lower @ " + (base + j));
        }
        for (int j = 0; j < count; j++) {
            assertEquals(expected[base + j].upper, Float.intBitsToFloat(in.readInt()), "upper @ " + (base + j));
        }
        for (int j = 0; j < count; j++) {
            assertEquals(expected[base + j].add, Float.intBitsToFloat(in.readInt()), "add @ " + (base + j));
        }
        for (int j = 0; j < count; j++) {
            assertEquals(expected[base + j].sum, in.readInt(), "sum @ " + (base + j));
        }
        for (int j = 0; j < count; j++) {
            final byte[] code = new byte[CODE_LENGTH];
            in.readBytes(code, 0, CODE_LENGTH);
            assertArrayEquals(expected[base + j].code, code, "code @ " + (base + j));
        }
    }

    private record Expected(float lower, float upper, float add, int sum, byte[] code) {
    }
}
