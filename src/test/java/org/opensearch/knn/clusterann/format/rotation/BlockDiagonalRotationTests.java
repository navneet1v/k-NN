/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

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
 * Fast, isolated unit tests for {@link BlockDiagonalRotation} (JUnit 5).
 *
 * <p>The fixtures are chosen so the three reversible conventions are each observable: the blocks are asymmetric, so
 * applying a transpose differs; the permutation is not the identity, so gathering differs from scattering; and the two
 * blocks differ in width in one case, so a destination offset that ignored the first block's width lands somewhere
 * visible.
 */
class BlockDiagonalRotationTests {

    private static final String FILE = "clar";

    /**
     * Four dimensions in two blocks of two. Block 0 is asymmetric — row 0 is {@code (1, 2)}, row 1 is {@code (3, 4)};
     * block 1 is the swap, which makes its output trivially checkable.
     */
    private static final float[][][] BLOCKS = { { { 1f, 2f }, { 3f, 4f } }, { { 0f, 1f }, { 1f, 0f } } };

    /**
     * Block 0 gathers dimensions 0 and 3, block 1 gathers 1 and 2 — scattered rather than contiguous, which is the whole
     * point of storing a permutation, and enough to tell a gather from a scatter.
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

    // ---------------------------------------------------------------- round trip

    /**
     * The whole point of the implementation owning both directions: what {@code create} produces, {@code writeTo} lays down and
     * {@code read} takes back must rotate identically. Every convention is in play at once — a transposed matrix, an
     * inverted permutation or a block written at the wrong offset all show up as a disagreement here.
     *
     * <p>The block layout is reached through the dimension, since the width is {@code min(DEFAULT_BLOCK_DIMENSION,
     * dimension)}: at or below 64 there is a <em>single</em> block, above it several, and a dimension that is not a
     * multiple of 64 leaves a narrower last one. The single-block cases matter most — an implementation tempted to skip
     * the permutation there would rotate into a different space from the one its own file describes, and nothing else
     * would say so.
     */
    @ParameterizedTest(name = "dimension {0}")
    @ValueSource(ints = { 1, 4, 64, 65, 100, 128 })
    void testRoundTrip_thenReadAgreesWithGenerated(int dimension) throws IOException {
        // given
        BlockDiagonalRotation generated = BlockDiagonalRotationFormat.INSTANCE.create(dimension);
        float[] src = ramp(dimension);

        float[] beforeWrite = new float[dimension];
        generated.rotate(src, beforeWrite);

        // when
        Rotation reread = writeAndRead(generated, dimension);
        float[] afterRead = new float[dimension];
        reread.rotate(src, afterRead);

        // then
        assertArrayEquals(beforeWrite, afterRead, 1e-6f, "a rotation must survive its own round trip");
    }

    /** Orthogonal blocks make the whole transform orthogonal, which is what leaves the ranking alone. */
    @ParameterizedTest(name = "dimension {0}")
    @ValueSource(ints = { 1, 4, 7, 16, 65, 128 })
    void testCreate_thenPreservesLength(int dimension) throws IOException {
        // given — 65 and 128 cross into several blocks, where each block must be orthogonal on its own
        BlockDiagonalRotation rotation = BlockDiagonalRotationFormat.INSTANCE.create(dimension);
        float[] src = ramp(dimension);
        float[] dest = new float[dimension];

        // when
        rotation.rotate(src, dest);

        // then
        // Relative, not absolute: the ramp's normSq grows like d³/3 — ~700k at d = 128 — so a fixed epsilon would sit
        // below float precision. Each output accumulates bDim products, on blocks orthogonalised in float themselves.
        assertEquals(normSq(src), normSq(dest), normSq(src) * 1e-5f, "an orthogonal transform cannot change a vector's length");
    }

    /** The bytes written are the field's {@code clarLength}, so the caller can record the region from one number. */
    @Test
    void testWriteTo_thenReturnsTheBytesItWrote() throws IOException {
        // given
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, BLOCKS, PERMUTATION);

        // when
        Directory directory = directory();
        long written;
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            written = BlockDiagonalRotationFormat.INSTANCE.write(out, rotation);
        }

        // then
        try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
            assertEquals(in.length(), written, "the length reported must be the region actually laid down");
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
     * {@code (11, 32, 2, 1)}: gather with global indices, row-major rows, contiguous per-block output.
     */
    @Test
    void testRotate_thenGathersPerBlockAndWritesContiguously() throws IOException {
        // given
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, BLOCKS, PERMUTATION);
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 10f, 1f, 2f, 0.5f }, dest);

        // then
        assertArrayEquals(new float[] { 11f, 32f, 2f, 1f }, dest, 1e-6f);
    }

    /**
     * The orientation, isolated. Feeding block 0 a source that is 1 in dimension 0 selects the block's first
     * <em>column</em>, {@code (1, 3)}. The transpose would select the first row, {@code (1, 2)}.
     */
    @Test
    void testRotate_thenDotsEachRowWithTheBlocksInput() throws IOException {
        // given
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, BLOCKS, PERMUTATION);
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 1f, 0f, 0f, 0f }, dest);

        // then
        assertArrayEquals(new float[] { 1f, 3f, 0f, 0f }, dest, 1e-6f, "row-major, dest[j] = row j · gathered");
    }

    /**
     * The permutation direction, isolated. Dimension 3 is the second slot of block 0, so a source that is 1 there selects
     * that block's second column, {@code (2, 4)}. Read as a scatter, the value would come out in block 1.
     */
    @Test
    void testRotate_thenTreatsTheIndicesAsAGather() throws IOException {
        // given
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, BLOCKS, PERMUTATION);
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
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, blocks, permutation);
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
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, BLOCKS, PERMUTATION);
        float[] src = { 10f, 1f, 2f, 0.5f };

        // when
        rotation.rotate(src, new float[DIMENSION]);

        // then
        assertArrayEquals(new float[] { 10f, 1f, 2f, 0.5f }, src);
    }

    /** A vector of the wrong length would be rotated into a different space, so it is rejected rather than truncated. */
    @ParameterizedTest(name = "length {0}")
    @ValueSource(ints = { 3, 5 })
    void testRotate_whenTheVectorIsTheWrongLength_thenThrows(int length) {
        // given
        BlockDiagonalRotation rotation = rotationOver(DIMENSION, BLOCKS, PERMUTATION);

        // when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> rotation.rotate(new float[length], new float[DIMENSION])
        );

        // then
        assertTrue(e.getMessage().contains(DIMENSION + "-dimensional"), e.getMessage());
    }

    // ---------------------------------------------------------------- laziness and accounting

    /**
     * A rotation read from a region holds only the region until it is applied, so a field that is never queried never
     * pays for its blocks — 288 kilobytes at {@code d = 768} in eight blocks of 96.
     */
    @Test
    void testRead_thenReadsNothingUntilApplied() throws IOException {
        // given
        Rotation rotation = writeAndRead(rotationOver(DIMENSION, BLOCKS, PERMUTATION), DIMENSION);
        long cold = rotation.ramBytesUsed();

        // when
        rotation.rotate(new float[DIMENSION], new float[DIMENSION]);

        // then
        assertTrue(rotation.ramBytesUsed() > cold, cold + " -> " + rotation.ramBytesUsed());
    }

    /** Read once and kept, so a second query pays nothing — and gets the same answer. */
    @Test
    void testRotate_whenCalledTwice_thenReadsTheRotationOnce() throws IOException {
        // given
        Rotation rotation = writeAndRead(rotationOver(DIMENSION, BLOCKS, PERMUTATION), DIMENSION);
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
     * caller's slice, not an offset carried here, that reaches it.
     */
    @Test
    void testRead_whenTheFieldsRotationIsPartWayIntoTheFile_thenTheSliceReachesIt() throws IOException {
        // given — 64 bytes belonging to some earlier field, then this field's rotation
        Directory directory = directory();
        long offset = 64L;
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            out.writeBytes(new byte[(int) offset], 0, (int) offset);
            BlockDiagonalRotationFormat.INSTANCE.write(out, rotationOver(DIMENSION, BLOCKS, PERMUTATION));
        }

        // when
        try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
            Rotation rotation = BlockDiagonalRotationFormat.INSTANCE.read(in.slice("field", offset, in.length() - offset), DIMENSION);
            float[] dest = new float[DIMENSION];
            rotation.rotate(new float[] { 1f, 0f, 0f, 0f }, dest);

            // then
            assertArrayEquals(new float[] { 1f, 3f, 0f, 0f }, dest, 1e-6f);
        }
    }

    // ---------------------------------------------------------------- corruption

    /**
     * The blocks have to account for every dimension exactly. A short set would leave a tail of {@code dest} untouched —
     * whatever the caller's array happened to hold — which is a wrong query rather than a failed one.
     */
    @Test
    void testRead_whenTheBlocksDoNotCoverTheVector_thenThrows() throws IOException {
        // given — one block of 2 written for a 4-dimensional field
        BlockDiagonalRotation partial = rotationOver(2, new float[][][] { { { 1f, 2f }, { 3f, 4f } } }, new int[][] { { 0, 1 } });

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> {
            Rotation reread = writeAndRead(partial, DIMENSION);
            reread.rotate(new float[DIMENSION], new float[DIMENSION]);
        });

        // then
        assertTrue(e.getMessage().contains("blocks cover 2 dimensions"), e.getMessage());
    }

    /**
     * A repeated index is the dangerous corruption: it silently drops whichever dimension went unclaimed, and every score
     * that follows is of a vector missing a coordinate.
     */
    @Test
    void testRead_whenADimensionIsGatheredTwice_thenThrows() throws IOException {
        // given — dimension 1 twice, dimension 3 never
        BlockDiagonalRotation duplicated = rotationOver(DIMENSION, BLOCKS, new int[][] { { 0, 1 }, { 1, 2 } });

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> {
            Rotation reread = writeAndRead(duplicated, DIMENSION);
            reread.rotate(new float[DIMENSION], new float[DIMENSION]);
        });

        // then
        assertTrue(e.getMessage().contains("dimension 1 is gathered by more than one block"), e.getMessage());
    }

    @Test
    void testRead_whenAnIndexIsOutsideTheVector_thenThrows() throws IOException {
        // given
        BlockDiagonalRotation outOfRange = rotationOver(DIMENSION, BLOCKS, new int[][] { { 0, 9 }, { 1, 2 } });

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> {
            Rotation reread = writeAndRead(outOfRange, DIMENSION);
            reread.rotate(new float[DIMENSION], new float[DIMENSION]);
        });

        // then
        assertTrue(e.getMessage().contains("gathers dimension 9"), e.getMessage());
    }

    /**
     * A width the region cannot satisfy is refused before the allocation, so it says the entry and the file disagree
     * rather than surfacing as an EOF partway through a matrix.
     */
    @Test
    void testRead_whenTheRegionIsTooShort_thenThrows() throws IOException {
        // given — a header claiming a 2x2 block, then nothing
        Directory directory = directory();
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            out.writeVInt(1);
            out.writeVInt(2);
        }

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> {
            try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
                BlockDiagonalRotationFormat.INSTANCE.read(in, DIMENSION).rotate(new float[DIMENSION], new float[DIMENSION]);
            }
        });

        // then
        assertTrue(e.getMessage().contains("needs 16 bytes"), e.getMessage());
    }

    /** A block holds at least one dimension, so a count above the dimension cannot be describing this field. */
    @Test
    void testRead_whenThereAreMoreBlocksThanDimensions_thenThrows() throws IOException {
        // given
        Directory directory = directory();
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            out.writeVInt(DIMENSION + 1);
        }

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> {
            try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
                BlockDiagonalRotationFormat.INSTANCE.read(in, DIMENSION).rotate(new float[DIMENSION], new float[DIMENSION]);
            }
        });

        // then
        assertTrue(e.getMessage().contains("numBlocks must be in [0, " + DIMENSION + "]"), e.getMessage());
    }

    // ---------------------------------------------------------------- generation

    @Test
    void testCreate_whenTheDimensionIsNotPositive_thenThrows() {
        assertThrows(IllegalArgumentException.class, () -> BlockDiagonalRotationFormat.INSTANCE.create(0));
    }

    /** The seed is fixed, so a segment rebuilt from the same vectors rotates the same way. */
    @Test
    void testCreate_thenIsDeterministic() throws IOException {
        // given
        float[] src = ramp(8);
        float[] first = new float[8];
        float[] second = new float[8];

        // when
        BlockDiagonalRotationFormat.INSTANCE.create(8).rotate(src, first);
        BlockDiagonalRotationFormat.INSTANCE.create(8).rotate(src, second);

        // then
        assertArrayEquals(first, second, 0f, "the same dimension must produce the same rotation");
    }

    @Test
    void testRead_whenTheRegionIsMissing_thenThrows() {
        assertThrows(IllegalArgumentException.class, () -> BlockDiagonalRotationFormat.INSTANCE.read(null, DIMENSION));
    }

    // ---------------------------------------------------------------- helpers

    /** A rotation straight over in-memory blocks — the constructor and {@code Blocks} are package-private. */
    private static BlockDiagonalRotation rotationOver(int dimension, float[][][] matrices, int[][] indices) {
        return new BlockDiagonalRotation(dimension, new BlockDiagonalRotation.Blocks(matrices, indices));
    }

    private static float[] ramp(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = 0.5f + i;
        }
        return vector;
    }

    private static float normSq(float[] vector) {
        float sum = 0f;
        for (float value : vector) {
            sum += value * value;
        }
        return sum;
    }

    private Rotation writeAndRead(BlockDiagonalRotation rotation, int dimension) throws IOException {
        Directory directory = directory();
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            BlockDiagonalRotationFormat.INSTANCE.write(out, rotation);
        }
        IndexInput in = directory.openInput(FILE, IOContext.DEFAULT);
        return BlockDiagonalRotationFormat.INSTANCE.read(in, dimension);
    }

    private Directory directory() {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        return directory;
    }
}
