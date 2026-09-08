/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link ClusterANNFieldMeta} (JUnit 5).
 *
 * <p>This is a parser, so the tests come in two halves: a valid entry must survive the round trip
 * value for value and leave the input positioned exactly at its end, and an entry that cannot describe
 * a real field must be rejected as {@link CorruptIndexException} rather than trusted.
 *
 * <p>Bytes come from {@link ClusterANNFieldMetaEncoder}, which writes the layout independently of the
 * record — so a change to one side and not the other shows up here.
 */
class ClusterANNFieldMetaTests {

    private static final int BLOCK_SIZE = 32;

    // ---------------------------------------------------------------- round trip

    @Test
    void testRead_whenEntryIsValid_thenRoundTripsEveryValue() throws IOException {
        // given
        ClusterANNFieldMetaEncoder encoder = new ClusterANNFieldMetaEncoder().dimension(16)
            .vectorCount(30)
            .centroidCount(3)
            .similarityFunction(ClusterANNFieldMetaEncoder.SIMILARITY_COSINE)
            .docBits((byte) 20)
            .quantizerId((byte) 1)
            .quantizerParams(new byte[] { 7, 8, 9 })
            .clacOffset(0L)
            .clacLength(512L)
            .clacCentroidsOffset(64L)
            .clacRotatedCentroidsOffset(128L)
            .clapOffset(4L)
            .clapLength(900L)
            .clapCentroidOffsets(0L, 100L, 300L)
            .centroidLengths(100, 200, 600)
            .clusterSizes(10, 12, 8);

        // when
        ClusterANNFieldMeta meta = read(encoder);

        // then
        assertEquals(BLOCK_SIZE, meta.blockSize(), "blockSize comes from the file header, not the entry");
        assertEquals(16, meta.dimension());
        assertEquals(30, meta.vectorCount());
        assertEquals(3, meta.centroidCount());
        assertEquals(VectorSimilarityFunction.COSINE, meta.similarityFunction());
        assertEquals(20, meta.docBits());
        assertEquals(1, meta.quantizerId());
        assertArrayEquals(new byte[] { 7, 8, 9 }, meta.quantizerParams());
        assertEquals(0L, meta.clacOffset());
        assertEquals(512L, meta.clacLength());
        assertEquals(64L, meta.clacCentroidsOffset());
        assertEquals(NO_ROTATION, meta.clacRotatedCentroidsOffset(), "this entry is unrotated, so it has no rotated centroids");
        assertEquals(4L, meta.clapOffset());
        assertEquals(900L, meta.clapLength());
        assertArrayEquals(new long[] { 0L, 100L, 300L }, meta.clapCentroidOffsets());
        assertArrayEquals(new int[] { 100, 200, 600 }, meta.centroidLengths());
        assertArrayEquals(new int[] { 10, 12, 8 }, meta.clusterSizes());
    }

    @ParameterizedTest(name = "encoded {0}")
    @CsvSource({ "0, EUCLIDEAN", "1, DOT_PRODUCT", "2, COSINE" })
    void testRead_whenSimilarityIsKnown_thenDecodesIt(byte encoded, VectorSimilarityFunction expected) throws IOException {
        // given / when
        ClusterANNFieldMeta meta = read(new ClusterANNFieldMetaEncoder().similarityFunction(encoded));

        // then
        assertEquals(expected, meta.similarityFunction());
    }

    @Test
    void testRead_whenQuantizerHasNoParams_thenParamsAreEmptyRatherThanNull() throws IOException {
        // given / when
        ClusterANNFieldMeta meta = read(new ClusterANNFieldMetaEncoder().quantizerParams(new byte[0]));

        // then
        assertArrayEquals(new byte[0], meta.quantizerParams());
    }

    @Test
    void testRead_whenFieldHasNoVectors_thenEntryIsStillReadable() throws IOException {
        // given / when — an empty field: no vectors, so no clusters, so none of the sized arrays
        ClusterANNFieldMeta meta = read(new ClusterANNFieldMetaEncoder().vectorCount(0).centroidCount(0));

        // then
        assertEquals(0, meta.vectorCount());
        assertEquals(0, meta.centroidCount());
        assertEquals(0, meta.clapCentroidOffsets().length);
        assertTrue(meta.isEmpty());
    }

    // ---------------------------------------------------------------- emptiness

    /**
     * Emptiness is about vectors, not clusters: a field with vectors is not empty even if the entry claims no
     * clusters for them, since the offsets it carries still locate real data.
     */
    @ParameterizedTest(name = "{0} vectors in {1} clusters")
    @CsvSource({ "0, 0, true", "1, 1, false", "30, 3, false", "30, 0, false" })
    void testIsEmpty_thenFollowsVectorCountAlone(int vectorCount, int centroidCount, boolean expectedEmpty) throws IOException {
        // given / when
        ClusterANNFieldMeta meta = read(new ClusterANNFieldMetaEncoder().vectorCount(vectorCount).centroidCount(centroidCount));

        // then
        assertEquals(expectedEmpty, meta.isEmpty());
    }

    // ---------------------------------------------------------------- rotation

    /** A rotated field locates two extra things: the matrix, and the centroids in the rotated space. */
    @Test
    void testRead_whenRotationIsPresent_thenReadsBothOfItsOffsets() throws IOException {
        // given / when
        ClusterANNFieldMeta meta = read(
            new ClusterANNFieldMetaEncoder().rotationId((byte) ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN)
                .clacRotatedCentroidsOffset(128L)
                .clarOffset(256L)
                .clarLength(64L)
        );

        // then
        assertEquals(ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, meta.rotationId());
        assertEquals(128L, meta.clacRotatedCentroidsOffset());
        assertEquals(256L, meta.clarOffset());
        assertEquals(64L, meta.clarLength());
        assertTrue(meta.hasRotation());
    }

    /**
     * Rotation is not inferred from the similarity function: an L2 field may be stored unrotated, and an
     * inner-product field may be stored rotated. Both used to be impossible to express.
     */
    @ParameterizedTest(name = "similarity {0}, rotationId {1}")
    @CsvSource({ "0, 0, false", "0, 1, true", "1, 1, true", "2, 0, false" })
    void testRead_thenRotationIsIndependentOfSimilarity(byte similarity, byte rotationId, boolean expectedHasRotation) throws IOException {
        // given / when
        ClusterANNFieldMeta meta = read(new ClusterANNFieldMetaEncoder().similarityFunction(similarity).rotationId(rotationId));

        // then
        assertEquals(expectedHasRotation, meta.hasRotation());
    }

    @Test
    void testRead_whenThereIsNoRotation_thenNeitherOffsetIsConsumed() throws IOException {
        // given — a sentinel straight after the entry stands in for whatever follows it in the file
        int sentinel = 0x5EED;
        try (Directory directory = new ByteBuffersDirectory()) {
            write(directory, new ClusterANNFieldMetaEncoder().rotationId((byte) ClusterANNFieldMeta.ROTATION_NONE), sentinel);

            // when
            try (ChecksumIndexInput in = directory.openChecksumInput(ENTRY)) {
                ClusterANNFieldMeta meta = ClusterANNFieldMeta.read(in, BLOCK_SIZE);

                // then
                assertFalse(meta.hasRotation());
                assertEquals(NO_ROTATION, meta.clarOffset());
                assertEquals(NO_ROTATION, meta.clarLength());
                assertEquals(NO_ROTATION, meta.clacRotatedCentroidsOffset());
                assertEquals(sentinel, in.readInt(), "an unrotated field must spend the id byte and no rotated value");
            }
        }
    }

    @ParameterizedTest(name = "rotationId {0}")
    @ValueSource(ints = { 2, 7, 127, -1 })
    void testRead_whenRotationIsUnknown_thenThrows(int rotationId) {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().rotationId((byte) rotationId))
        );

        // then
        assertTrue(e.getMessage().contains("Unknown rotation: " + (byte) rotationId), e.getMessage());
    }

    @Test
    void testRead_whenRotationOffsetIsNegative_thenThrows() {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().rotationId((byte) ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN).clarOffset(-8L))
        );

        // then
        assertTrue(e.getMessage().contains("Negative clarOffset"), e.getMessage());
    }

    @Test
    void testRead_whenRotationLengthIsNegative_thenThrows() {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().rotationId((byte) ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN).clarLength(-8L))
        );

        // then
        assertTrue(e.getMessage().contains("Negative clarLength"), e.getMessage());
    }

    // ---------------------------------------------------------------- corrupt entries

    @ParameterizedTest(name = "encoded {0}")
    @ValueSource(ints = { 3, 42, -1 })
    void testRead_whenSimilarityIsUnknown_thenThrows(int encoded) {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().similarityFunction((byte) encoded))
        );

        // then
        assertTrue(e.getMessage().contains("Unknown similarity function: " + (byte) encoded), e.getMessage());
    }

    @ParameterizedTest(name = "dimension {0}")
    @ValueSource(ints = { 0, -1, -4096 })
    void testRead_whenDimensionIsNotPositive_thenThrows(int dimension) {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().dimension(dimension))
        );

        // then
        assertTrue(e.getMessage().contains("Dimension must be positive, got: " + dimension), e.getMessage());
    }

    @Test
    void testRead_whenVectorCountIsNegative_thenThrows() {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().vectorCount(-1).centroidCount(0))
        );

        // then
        assertTrue(e.getMessage().contains("Negative vectorCount: -1"), e.getMessage());
    }

    /**
     * A cluster holds at least one vector, so more clusters than vectors cannot be true — and rejecting it
     * before the sized arrays are read is what stops a corrupt count from driving a huge allocation.
     */
    @ParameterizedTest(name = "{0} clusters over {1} vectors")
    @CsvSource({ "4, 3", "1, 0", "-1, 10" })
    void testRead_whenCentroidCountCannotBeTrue_thenThrows(int centroidCount, int vectorCount) {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().vectorCount(vectorCount).centroidCount(centroidCount))
        );

        // then
        assertTrue(e.getMessage().contains("centroidCount must be in [0, " + vectorCount + "], got: " + centroidCount), e.getMessage());
    }

    /**
     * A count is only ever as big as the file can back. Bounding it against the bytes that remain, rather than
     * against a guessed maximum, is what stops a corrupt entry from asking for gigabytes of heap — the array is
     * allocated from the count, so an unbounded count OOMs the reader before the footer is ever verified.
     */
    @Test
    void testRead_whenCentroidCountIsMoreThanTheFileCouldHold_thenThrowsWithoutAllocating() {
        // given — the entry claims a couple of billion clusters; the bytes for them are not there
        ClusterANNFieldMetaEncoder encoder = new ClusterANNFieldMetaEncoder().vectorCount(Integer.MAX_VALUE)
            .centroidCount(3)
            .declaredCentroidCount(Integer.MAX_VALUE);

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> read(encoder));

        // then
        assertTrue(e.getMessage().contains("centroidCount=" + Integer.MAX_VALUE), e.getMessage());
        assertTrue(e.getMessage().contains("but only"), e.getMessage());
    }

    @Test
    void testRead_whenQuantizerParamsLengthIsMoreThanTheFileCouldHold_thenThrowsWithoutAllocating() {
        // given — an empty payload behind a length claiming 2GB of it
        ClusterANNFieldMetaEncoder encoder = new ClusterANNFieldMetaEncoder().quantizerParams(new byte[0])
            .quantizerParamsLength(Integer.MAX_VALUE);

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> read(encoder));

        // then
        assertTrue(e.getMessage().contains("quantizerParamsLength=" + Integer.MAX_VALUE), e.getMessage());
    }

    @Test
    void testRead_whenQuantizerParamsLengthIsNegative_thenThrows() {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(new ClusterANNFieldMetaEncoder().quantizerParamsLength(-3))
        );

        // then
        assertTrue(e.getMessage().contains("Negative quantizerParamsLength: -3"), e.getMessage());
    }

    /** Each offset locates data in another file, so a negative one is unusable and must be caught by name. */
    private static Stream<Arguments> negativeOffsets() {
        return Stream.of(
            Arguments.of("clacOffset", (Consumer<ClusterANNFieldMetaEncoder>) e -> e.clacOffset(-1L)),
            Arguments.of("clacLength", (Consumer<ClusterANNFieldMetaEncoder>) e -> e.clacLength(-1L)),
            Arguments.of("clacCentroidsOffset", (Consumer<ClusterANNFieldMetaEncoder>) e -> e.clacCentroidsOffset(-1L)),
            Arguments.of("clapOffset", (Consumer<ClusterANNFieldMetaEncoder>) e -> e.clapOffset(-1L)),
            Arguments.of("clapLength", (Consumer<ClusterANNFieldMetaEncoder>) e -> e.clapLength(-1L))
        );
    }

    @ParameterizedTest(name = "negative {0}")
    @MethodSource("negativeOffsets")
    void testRead_whenAnOffsetIsNegative_thenThrowsNamingIt(String name, Consumer<ClusterANNFieldMetaEncoder> corrupt) {
        // given
        ClusterANNFieldMetaEncoder encoder = new ClusterANNFieldMetaEncoder();
        corrupt.accept(encoder);

        // when
        CorruptIndexException e = assertThrows(CorruptIndexException.class, () -> read(encoder));

        // then
        assertTrue(e.getMessage().contains("Negative " + name + ": -1"), e.getMessage());
    }

    /** Only reachable on a rotated field, since an unrotated one carries no such offset to be negative. */
    @Test
    void testRead_whenTransformedCentroidOffsetIsNegative_thenThrows() {
        // given / when
        CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> read(
                new ClusterANNFieldMetaEncoder().rotationId((byte) ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN)
                    .clacRotatedCentroidsOffset(-1L)
            )
        );

        // then
        assertTrue(e.getMessage().contains("Negative clacRotatedCentroidsOffset: -1"), e.getMessage());
    }

    // ---------------------------------------------------------------- constructor invariants

    /**
     * The three arrays are indexed by cluster, so one of the wrong length means the entry and the counts it
     * was written with disagree. Checked in the record rather than only in {@link ClusterANNFieldMeta#read}
     * so the write side cannot build an entry the read side would reject.
     */
    @ParameterizedTest(name = "{0} of the wrong length")
    @ValueSource(strings = { "clapCentroidOffsets", "centroidLengths", "clusterSizes" })
    void testConstructor_whenASizedArrayDoesNotMatchCentroidCount_thenThrows(String name) {
        // given — three clusters, but the named array holds two
        long[] offsets = "clapCentroidOffsets".equals(name) ? new long[] { 0L, 100L } : new long[] { 0L, 100L, 300L };
        int[] lengths = "centroidLengths".equals(name) ? new int[] { 100, 200 } : new int[] { 100, 200, 600 };
        int[] sizes = "clusterSizes".equals(name) ? new int[] { 10, 12 } : new int[] { 10, 12, 8 };

        // when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> meta(3, offsets, lengths, sizes, ClusterANNFieldMeta.ROTATION_NONE, NO_ROTATION, NO_ROTATION, NO_ROTATION)
        );

        // then
        assertTrue(e.getMessage().contains(name + " must hold centroidCount=3 entries, got: 2"), e.getMessage());
    }

    @Test
    void testConstructor_whenRotationIsClaimedWithoutAMatrixOffset_thenThrows() {
        // given / when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> meta(0, EMPTY_OFFSETS, EMPTY_COUNTS, EMPTY_COUNTS, ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, 128L, NO_ROTATION, 64L)
        );

        // then
        assertTrue(e.getMessage().contains("disagrees with clarOffset"), e.getMessage());
    }

    @Test
    void testConstructor_whenRotationIsClaimedWithoutAMatrixLength_thenThrows() {
        // given / when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> meta(0, EMPTY_OFFSETS, EMPTY_COUNTS, EMPTY_COUNTS, ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, 128L, 256L, NO_ROTATION)
        );

        // then
        assertTrue(e.getMessage().contains("disagrees with clarLength"), e.getMessage());
    }

    /** The rotated centroids are on the same terms as the matrix: a rotated field has to locate both. */
    @Test
    void testConstructor_whenRotationIsClaimedWithoutTransformedCentroids_thenThrows() {
        // given / when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> meta(0, EMPTY_OFFSETS, EMPTY_COUNTS, EMPTY_COUNTS, ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, NO_ROTATION, 256L, 64L)
        );

        // then
        assertTrue(e.getMessage().contains("disagrees with clacRotatedCentroidsOffset"), e.getMessage());
    }

    @ParameterizedTest(name = "{0} given without a rotation")
    @ValueSource(strings = { "clarOffset", "clarLength", "clacRotatedCentroidsOffset" })
    void testConstructor_whenARotationOnlyOffsetIsGivenWithoutARotation_thenThrows(String name) {
        // given
        long clacTransformed = "clacRotatedCentroidsOffset".equals(name) ? 128L : NO_ROTATION;
        long clarOffset = "clarOffset".equals(name) ? 256L : NO_ROTATION;
        long clarLength = "clarLength".equals(name) ? 64L : NO_ROTATION;

        // when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> meta(
                0,
                EMPTY_OFFSETS,
                EMPTY_COUNTS,
                EMPTY_COUNTS,
                ClusterANNFieldMeta.ROTATION_NONE,
                clacTransformed,
                clarOffset,
                clarLength
            )
        );

        // then
        assertTrue(e.getMessage().contains("disagrees with " + name), e.getMessage());
    }

    // ---------------------------------------------------------------- helpers

    private static final String ENTRY = "entry";
    private static final long NO_ROTATION = ClusterANNFieldMeta.NO_ROTATION;
    private static final long[] EMPTY_OFFSETS = new long[0];
    private static final int[] EMPTY_COUNTS = new int[0];

    /** Encodes the entry, then reads it back the way the reader does. */
    private static ClusterANNFieldMeta read(ClusterANNFieldMetaEncoder encoder) throws IOException {
        try (Directory directory = new ByteBuffersDirectory()) {
            write(directory, encoder, null);
            try (ChecksumIndexInput in = directory.openChecksumInput(ENTRY)) {
                return ClusterANNFieldMeta.read(in, BLOCK_SIZE);
            }
        }
    }

    private static void write(Directory directory, ClusterANNFieldMetaEncoder encoder, Integer trailer) throws IOException {
        try (IndexOutput out = directory.createOutput(ENTRY, IOContext.DEFAULT)) {
            encoder.write(out);
            if (trailer != null) {
                out.writeInt(trailer);
            }
        }
    }

    /** A record built directly, for the invariants that {@link ClusterANNFieldMeta#read} cannot reach. */
    private static ClusterANNFieldMeta meta(
        int centroidCount,
        long[] clapCentroidOffsets,
        int[] centroidLengths,
        int[] clusterSizes,
        int rotationId,
        long clacRotatedCentroidsOffset,
        long clarOffset,
        long clarLength
    ) {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            8,
            30,
            centroidCount,
            VectorSimilarityFunction.EUCLIDEAN,
            20,          // docBits
            rotationId,
            0,           // quantizerId
            new byte[0],
            0L,          // clacOffset
            512L,        // clacLength
            64L,         // clacCentroidsOffset
            clacRotatedCentroidsOffset,
            0L,          // clapOffset
            900L,        // clapLength
            clapCentroidOffsets,
            centroidLengths,
            clusterSizes,
            clarOffset,
            clarLength,
            null
        );
    }
}
