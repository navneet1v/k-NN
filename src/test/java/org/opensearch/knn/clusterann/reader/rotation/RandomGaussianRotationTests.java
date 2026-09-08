/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.rotation;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link RandomGaussianRotation} (JUnit 5).
 *
 * <p>Bytes come from {@link #write}, an independent encoder of the same layout, so the reader is driven over a real
 * {@link IndexInput}. The fixtures are chosen so the three reversible conventions are each observable: the blocks are
 * asymmetric, so applying a transpose differs; the permutation is not the identity, so gathering differs from
 * scattering; and the two blocks differ in width, so a destination offset that ignored the first block's width lands
 * somewhere visible.
 */
class RandomGaussianRotationTests {

    private static final String FILE = "clar";

    /**
     * Four dimensions in two blocks of two. Block 0 is asymmetric — row 0 is {@code (1, 2)}, row 1 is {@code (3, 4)};
     * block 1 is the swap, which makes its output trivially checkable.
     */
    private static final float[][][] BLOCKS = { { { 1f, 2f }, { 3f, 4f } }, { { 0f, 1f }, { 1f, 0f } } };

    /**
     * Block 0 gathers dimensions 0 and 3, block 1 gathers 1 and 2 — scattered rather than contiguous, which is the
     * whole point of storing a permutation, and enough to tell a gather from a scatter.
     */
    private static final int[][] PERMUTATION = { { 0, 3 }, { 1, 2 } };

    private static final int DIMENSION = 4;

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- applying

    /**
     * The whole transform, worked by hand. {@code src = (10, 1, 2, 0.5)}:
     *
     * <ul>
     *   <li>block 0 gathers dimensions (0, 3) = {@code (10, 0.5)}; row 0 gives {@code 1·10 + 2·0.5 = 11}, row 1 gives
     *       {@code 3·10 + 4·0.5 = 32}
     *   <li>block 1 gathers dimensions (1, 2) = {@code (1, 2)}; the swap gives {@code (2, 1)}
     * </ul>
     *
     * <p>Block 0's output occupies {@code dest[0..2)} and block 1's {@code dest[2..4)}, so the result is
     * {@code (11, 32, 2, 1)}. Every convention is in play at once: gather with global indices, row-major rows, and
     * contiguous per-block output.
     */
    @Test
    void testRotate_thenGathersPerBlockAndWritesContiguously() throws IOException {
        // given
        Rotation rotation = rotation();
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 10f, 1f, 2f, 0.5f }, dest);

        // then
        assertArrayEquals(new float[] { 11f, 32f, 2f, 1f }, dest, 1e-6f);
    }

    /**
     * The orientation, isolated. Feeding block 0 a source that is 1 in dimension 0 and 0 in dimension 3 selects the
     * block's first <em>column</em>, {@code (1, 3)}. The transpose would select the first row, {@code (1, 2)} — so this
     * is the case that fails if the matrix is applied the wrong way round.
     */
    @Test
    void testRotate_thenDotsEachRowWithTheBlocksInput() throws IOException {
        // given
        Rotation rotation = rotation();
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 1f, 0f, 0f, 0f }, dest);

        // then
        assertArrayEquals(new float[] { 1f, 3f, 0f, 0f }, dest, 1e-6f, "row-major, dest[j] = row j · gathered");
    }

    /**
     * The permutation direction, isolated. Dimension 3 is the second slot of block 0, so a source that is 1 there and
     * 0 elsewhere selects that block's second column, {@code (2, 4)}. Read as a scatter instead — {@code src[3]} going
     * to slot 3 — the value would come out in block 1 and the first two outputs would be zero.
     */
    @Test
    void testRotate_thenTreatsTheIndicesAsAGather() throws IOException {
        // given
        Rotation rotation = rotation();
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 0f, 0f, 0f, 1f }, dest);

        // then
        assertArrayEquals(new float[] { 2f, 4f, 0f, 0f }, dest, 1e-6f, "indices[slot] names the source dimension");
    }

    /** Blocks need not be the same width, and a wider first block must push the second one's output along. */
    @Test
    void testRotate_whenTheBlocksDifferInWidth_thenTheOffsetsFollowThem() throws IOException {
        // given — a block of 3 then a block of 1, so the second writes dest[3] alone
        float[][][] blocks = { { { 1f, 0f, 0f }, { 0f, 1f, 0f }, { 0f, 0f, 1f } }, { { 2f } } };
        int[][] permutation = { { 3, 1, 0 }, { 2 } };
        Rotation rotation = new RandomGaussianRotation(write(blocks, permutation), DIMENSION);
        float[] dest = new float[DIMENSION];

        // when — the identity block echoes its gathered input, so dest[0..3) is (src[3], src[1], src[0])
        rotation.rotate(new float[] { 7f, 8f, 9f, 10f }, dest);

        // then
        assertArrayEquals(new float[] { 10f, 8f, 7f, 18f }, dest, 1e-6f);
    }

    /** The plan is made in the unrotated space while the scan runs in the rotated one, so the query has to survive. */
    @Test
    void testRotate_thenLeavesTheSourceUntouched() throws IOException {
        // given
        Rotation rotation = rotation();
        float[] src = { 10f, 1f, 2f, 0.5f };

        // when
        rotation.rotate(src, new float[DIMENSION]);

        // then
        assertArrayEquals(new float[] { 10f, 1f, 2f, 0.5f }, src);
    }

    /** Orthogonal blocks make the whole transform orthogonal, which is what leaves the ranking alone. */
    @Test
    void testRotate_whenEveryBlockIsOrthogonal_thenLengthIsPreserved() throws IOException {
        // given — a rotation by 30° and a swap, both orthogonal
        float cos = (float) Math.cos(Math.PI / 6);
        float sin = (float) Math.sin(Math.PI / 6);
        float[][][] blocks = { { { cos, -sin }, { sin, cos } }, { { 0f, 1f }, { 1f, 0f } } };
        Rotation rotation = new RandomGaussianRotation(write(blocks, PERMUTATION), DIMENSION);
        float[] src = { 3f, 1f, 4f, 1.5f };
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(src, dest);

        // then
        assertEquals(normSq(src), normSq(dest), 1e-4f, "an orthogonal transform cannot change a vector's length");
    }

    /** Read once and kept, so a second query pays nothing — and gets the same answer. */
    @Test
    void testRotate_whenCalledTwice_thenReadsTheRotationOnce() throws IOException {
        // given
        Rotation rotation = rotation();
        float[] first = new float[DIMENSION];
        float[] second = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 2f, 5f, 1f, 3f }, first);
        long afterFirst = rotation.ramBytesUsed();
        rotation.rotate(new float[] { 2f, 5f, 1f, 3f }, second);

        // then
        assertArrayEquals(first, second, 1e-6f);
        assertEquals(afterFirst, rotation.ramBytesUsed(), "a second call loads nothing further");
    }

    /**
     * One {@code .clar} holds several fields' rotations, so a field's own begins partway into the file — and it is the
     * caller's slice, not an offset carried here, that reaches it. Reading from zero is only correct because the region
     * handed over starts at the header, so this pins that the two halves of the convention meet.
     */
    @Test
    void testRotate_whenTheFieldsRotationIsPartWayIntoTheFile_thenTheSliceReachesIt() throws IOException {
        // given — 64 bytes belonging to some earlier field, then this field's rotation
        long offset = 64L;
        IndexInput clar = writeAt(BLOCKS, PERMUTATION, offset);
        Rotation rotation = new RandomGaussianRotation(clar.slice("field", offset, clar.length() - offset), DIMENSION);
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 1f, 0f, 0f, 0f }, dest);

        // then
        assertArrayEquals(new float[] { 1f, 3f, 0f, 0f }, dest, 1e-6f);
    }

    // ---------------------------------------------------------------- shape

    @Test
    void testDimension_thenReportsWhatItWasBuiltWith() throws IOException {
        assertEquals(DIMENSION, rotation().dimension());
    }

    /** A vector of the wrong length would be rotated into a different space, so it is rejected rather than truncated. */
    @ParameterizedTest(name = "length {0}")
    @ValueSource(ints = { 3, 5 })
    void testRotate_whenTheVectorIsTheWrongLength_thenThrows(int length) throws IOException {
        // given
        Rotation rotation = rotation();

        // when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> rotation.rotate(new float[length], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains(DIMENSION + "-dimensional"), e.getMessage());
    }

    @Test
    void testRotate_whenTheDestinationIsTheWrongLength_thenThrows() throws IOException {
        // given
        Rotation rotation = rotation();

        // when / then
        assertThrows(IllegalArgumentException.class, () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION + 1]));
    }

    // ---------------------------------------------------------------- corruption

    /**
     * The blocks have to account for every dimension exactly. A short set would leave a tail of {@code dest} untouched
     * — whatever the caller's array happened to hold — which is a wrong query rather than a failed one.
     */
    @Test
    void testRotate_whenTheBlocksDoNotCoverTheVector_thenThrows() throws IOException {
        // given — one block of 2 for a 4-dimensional field
        float[][][] blocks = { { { 1f, 2f }, { 3f, 4f } } };
        Rotation rotation = new RandomGaussianRotation(write(blocks, new int[][] { { 0, 1 } }), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("blocks cover 2 dimensions"), e.getMessage());
    }

    /** A block wider than what is left cannot be part of a permutation of the vector. */
    @Test
    void testRotate_whenABlockIsWiderThanTheVector_thenThrows() throws IOException {
        // given — a block of 5 for a 4-dimensional field
        float[][][] blocks = { new float[5][5] };
        Rotation rotation = new RandomGaussianRotation(write(blocks, new int[][] { { 0, 1, 2, 3, 0 } }), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("has dimension 5"), e.getMessage());
    }

    /**
     * A repeated index is the dangerous corruption: it silently drops whichever dimension went unclaimed, and every
     * score that follows is of a vector missing a coordinate. Caught rather than trusted.
     */
    @Test
    void testRotate_whenADimensionIsGatheredTwice_thenThrows() throws IOException {
        // given — dimension 1 twice, dimension 3 never
        Rotation rotation = new RandomGaussianRotation(write(BLOCKS, new int[][] { { 0, 1 }, { 1, 2 } }), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("dimension 1 is gathered by more than one block"), e.getMessage());
    }

    @Test
    void testRotate_whenAnIndexIsOutsideTheVector_thenThrows() throws IOException {
        // given
        Rotation rotation = new RandomGaussianRotation(write(BLOCKS, new int[][] { { 0, 9 }, { 1, 2 } }), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("gathers dimension 9"), e.getMessage());
    }

    /** A permutation naming a different number of dimensions than its block is wide cannot feed it. */
    @Test
    void testRotate_whenAPermutationDisagreesWithItsBlock_thenThrows() throws IOException {
        // given — block 0 is 2 wide, its permutation names 1 dimension
        Rotation rotation = new RandomGaussianRotation(writeMismatched(), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("its permutation names 1 dimensions"), e.getMessage());
    }

    /**
     * A width the region cannot satisfy is refused before the allocation, so it says the entry and the file disagree
     * rather than surfacing as an EOF partway through a matrix.
     */
    @Test
    void testRotate_whenTheRegionIsTooShort_thenThrows() throws IOException {
        // given — a header claiming a 2x2 block, then nothing
        Rotation rotation = new RandomGaussianRotation(writeHeaderOnly(), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("needs 16 bytes"), e.getMessage());
    }

    /** A block holds at least one dimension, so a count above the dimension cannot be describing this field. */
    @Test
    void testRotate_whenThereAreMoreBlocksThanDimensions_thenThrows() throws IOException {
        // given
        Rotation rotation = new RandomGaussianRotation(writeBlockCount(DIMENSION + 1), DIMENSION);

        // when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> rotation.rotate(new float[DIMENSION], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains("numBlocks must be in [0, " + DIMENSION + "]"), e.getMessage());
    }

    // ---------------------------------------------------------------- accounting

    /** Constructing one reads nothing, so an unqueried field carries no blocks — and says so. */
    @Test
    void testRamBytesUsed_whenNotYetApplied_thenCountsNoBlocks() throws IOException {
        // given
        Rotation cold = rotation();
        long beforeUse = cold.ramBytesUsed();

        // when
        cold.rotate(new float[DIMENSION], new float[DIMENSION]);

        // then
        assertTrue(cold.ramBytesUsed() > beforeUse, beforeUse + " -> " + cold.ramBytesUsed());
    }

    // ---------------------------------------------------------------- helpers

    private static float normSq(float[] vector) {
        float sum = 0f;
        for (float value : vector) {
            sum += value * value;
        }
        return sum;
    }

    private Rotation rotation() throws IOException {
        return new RandomGaussianRotation(write(BLOCKS, PERMUTATION), DIMENSION);
    }

    private IndexInput write(float[][][] blocks, int[][] permutation) throws IOException {
        return output(out -> {
            out.writeVInt(blocks.length);
            for (float[][] block : blocks) {
                out.writeVInt(block.length);
                for (float[] row : block) {
                    for (float value : row) {
                        out.writeInt(Float.floatToIntBits(value));
                    }
                }
            }
            for (int[] indices : permutation) {
                out.writeVInt(indices.length);
                for (int index : indices) {
                    out.writeInt(index);
                }
            }
        });
    }

    /** The rotation preceded by another field's bytes, for the case that slices it back out. */
    private IndexInput writeAt(float[][][] blocks, int[][] permutation, long offset) throws IOException {
        return output(out -> {
            out.writeBytes(new byte[(int) offset], 0, (int) offset);
            out.writeVInt(blocks.length);
            for (float[][] block : blocks) {
                out.writeVInt(block.length);
                for (float[] row : block) {
                    for (float value : row) {
                        out.writeInt(Float.floatToIntBits(value));
                    }
                }
            }
            for (int[] indices : permutation) {
                out.writeVInt(indices.length);
                for (int index : indices) {
                    out.writeInt(index);
                }
            }
        });
    }

    /** Two 2-wide blocks, but block 0's permutation names one dimension — not expressible through {@link #write}. */
    private IndexInput writeMismatched() throws IOException {
        return output(out -> {
            out.writeVInt(2);
            for (int block = 0; block < 2; block++) {
                out.writeVInt(2);
                for (int value = 0; value < 4; value++) {
                    out.writeInt(Float.floatToIntBits(1f));
                }
            }
            out.writeVInt(1);      // block 0's permutation, one short
            out.writeInt(0);
            out.writeVInt(2);
            out.writeInt(1);
            out.writeInt(2);
        });
    }

    /** A count and a width, and then the file ends — the region cannot hold the block it promises. */
    private IndexInput writeHeaderOnly() throws IOException {
        return output(out -> {
            out.writeVInt(1);
            out.writeVInt(2);
        });
    }

    private IndexInput writeBlockCount(int numBlocks) throws IOException {
        return output(out -> out.writeVInt(numBlocks));
    }

    private IndexInput output(Encoder encoder) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            encoder.encode(out);
        }
        return directory.openInput(FILE, IOContext.DEFAULT);
    }

    private interface Encoder {
        void encode(IndexOutput out) throws IOException;
    }
}
