/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;

import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link RotationFormats} and {@link IdentityRotation} (JUnit 5).
 *
 * <p>The registry is the one place that maps an id onto an implementation, so what these cover is that mapping: which ids it
 * serves, what it refuses, and that an implementation cannot be half-registered — its {@code id()} and the id it answers to must
 * be the same number, or a segment would be written under one and read under another.
 */
class RotationFormatsTests {

    private static final int DIMENSION = 4;

    private static final String FILE = "clar";

    /**
     * Each id resolves to the implementation that {@code .clam} means by it. Asserted by type because the id lives only
     * in the registry now — an implementation cannot state its own number, so this switch is the whole of the mapping
     * and the only thing that could get it wrong.
     */
    @Test
    void testForId_thenMapsEachIdToItsImplementation() {
        assertInstanceOf(IdentityRotationFormat.class, RotationFormats.forId(ROTATION_NONE));
        assertInstanceOf(BlockDiagonalRotationFormat.class, RotationFormats.forId(ROTATION_RANDOM_GAUSSIAN));
    }

    /**
     * What an implementation generates is what it reads back — the same type on both sides. Generics make that a compile
     * time guarantee rather than a runtime check, so this only has to show the pair meets at all.
     */
    @ParameterizedTest(name = "rotationId {0}")
    @ValueSource(ints = { ROTATION_NONE, ROTATION_RANDOM_GAUSSIAN })
    void testCreate_thenProducesARotationOfTheDimensionAsked(int rotationId) {
        assertEquals(DIMENSION, RotationFormats.create(rotationId, DIMENSION).dimension());
    }

    /**
     * An id this reader cannot apply is refused outright rather than defaulted: a rotation applied wrongly is still a
     * rotation, so a segment written by an implementation this format does not have would be scored against a space its vectors
     * were never stored in, silently.
     */
    @ParameterizedTest(name = "rotationId {0}")
    @ValueSource(ints = { 2, 7, 42, -1 })
    void testForId_whenTheFamilyIsUnknown_thenThrows(int rotationId) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> RotationFormats.forId(rotationId));
        assertTrue(e.getMessage().contains("Unsupported rotationId: " + rotationId), e.getMessage());
    }

    // ---------------------------------------------------------------- write

    /**
     * The write side's counterpart to {@link RotationFormats#read}: what an id writes, the same id reads back, and the
     * rotation that comes out agrees with the one that went in. That round trip is what makes the pair of by-id entry
     * points usable at all — a caller with only an id never sees the implementation's own type.
     */
    @ParameterizedTest(name = "rotationId {0}")
    @ValueSource(ints = { ROTATION_NONE, ROTATION_RANDOM_GAUSSIAN })
    void testWrite_thenTheSameIdReadsItBack(int rotationId) throws IOException {
        // given
        Rotation generated = RotationFormats.create(rotationId, DIMENSION);
        float[] src = { 2f, 5f, 1f, 3f };
        float[] before = new float[DIMENSION];
        generated.rotate(src, before);

        // when
        try (Directory directory = new ByteBuffersDirectory()) {
            long written;
            try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
                written = RotationFormats.write(rotationId, out, generated);
            }

            // then — an implementation that stores nothing still round-trips; it just has no region to read
            try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
                Rotation reread = RotationFormats.read(rotationId, written == 0L ? null : in, DIMENSION);
                float[] after = new float[DIMENSION];
                reread.rotate(src, after);
                assertArrayEquals(before, after, 1e-6f, "a rotation must survive the by-id round trip");
            }
        }
    }

    /**
     * The by-id write cannot be checked by the compiler — an id carries no type — so a mismatched pair has to fail
     * loudly rather than lay down bytes nothing can read back. Named on both sides so the caller can see which half
     * was wrong.
     */
    @Test
    void testWrite_whenTheRotationIsFromAnotherImplementation_thenThrows() throws IOException {
        // given — an identity rotation offered to the block-diagonal id
        Rotation foreign = RotationFormats.create(ROTATION_NONE, DIMENSION);

        // when
        try (Directory directory = new ByteBuffersDirectory(); IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> RotationFormats.write(ROTATION_RANDOM_GAUSSIAN, out, foreign)
            );

            // then
            assertTrue(e.getMessage().contains("rotationId " + ROTATION_RANDOM_GAUSSIAN), e.getMessage());
            assertTrue(e.getMessage().contains("IdentityRotation"), e.getMessage());
        }
    }

    @Test
    void testWrite_whenTheImplementationIsUnknown_thenThrows() throws IOException {
        try (Directory directory = new ByteBuffersDirectory(); IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            assertThrows(
                IllegalArgumentException.class,
                () -> RotationFormats.write(42, out, RotationFormats.create(ROTATION_NONE, DIMENSION))
            );
        }
    }

    // ---------------------------------------------------------------- identity

    /**
     * An unrotated field gets the identity, not {@code null} — so a caller rotates unconditionally instead of carrying
     * the branch. Asserted by applying it: the vector comes back unchanged, in a different array.
     */
    @Test
    void testRead_whenTheFieldIsNotRotated_thenGivesTheIdentity() throws IOException {
        // given — no region, which is what an unrotated field has
        Rotation rotation = RotationFormats.read(ROTATION_NONE, null, DIMENSION);
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 2f, 5f, 1f, 3f }, dest);

        // then
        assertInstanceOf(IdentityRotation.class, rotation);
        assertArrayEquals(new float[] { 2f, 5f, 1f, 3f }, dest);
        assertEquals(DIMENSION, rotation.dimension());
        assertTrue(rotation.ramBytesUsed() > 0, "a shallow size is still a size");
    }

    /**
     * The identity stores nothing, and says so by writing nothing — the same statement {@code clarOffset = NO_ROTATION}
     * in {@code .clam} makes from the other side. It touches {@code out} not at all, so {@code null} is safe here.
     */
    @Test
    void testWrite_whenTheFieldIsNotRotated_thenWritesNothing() throws IOException {
        assertEquals(
            0L,
            RotationFormats.write(ROTATION_NONE, null, new IdentityRotation(DIMENSION)),
            "no region means no bytes and no output to touch"
        );
    }

    /** A vector of the wrong length is rejected rather than copied short, even by the implementation that only copies. */
    @Test
    void testRotate_whenTheVectorIsTheWrongLength_thenThrows() {
        Rotation rotation = new IdentityRotation(DIMENSION);
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> rotation.rotate(new float[DIMENSION - 1], new float[DIMENSION])
        );
        assertTrue(e.getMessage().contains(DIMENSION + "-dimensional"), e.getMessage());
    }
}
