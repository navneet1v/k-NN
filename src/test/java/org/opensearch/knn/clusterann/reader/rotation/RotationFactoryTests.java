/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.rotation;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link RotationFactory} and {@link IdentityRotation} (JUnit 5).
 *
 * <p>The factory is the only place that maps a field's {@code rotationId} onto a family, so what these cover is that
 * mapping: which ids it serves, what it refuses, and that a field carrying no rotation still gets something a caller
 * can apply without asking whether it needs to.
 */
class RotationFactoryTests {

    private static final int DIMENSION = 2;

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- dispatch

    /**
     * An unrotated field gets the identity, not {@code null} — so a caller rotates unconditionally instead of carrying
     * the branch. Asserted by applying it: the vector comes back unchanged, in a different array.
     */
    @Test
    void testCreate_whenTheFieldIsNotRotated_thenGivesTheIdentity() throws IOException {
        // given
        Rotation rotation = RotationFactory.create(fieldMeta(ClusterANNFieldMeta.ROTATION_NONE), null);
        float[] dest = new float[DIMENSION];

        // when
        rotation.rotate(new float[] { 2f, 5f }, dest);

        // then
        assertArrayEquals(new float[] { 2f, 5f }, dest);
        assertEquals(DIMENSION, rotation.dimension());
        assertTrue(rotation.ramBytesUsed() > 0, "a shallow size is still a size");
    }

    /** The identity needs no {@code .clar}, which is what lets the reader pass {@code null} for an unrotated field. */
    @Test
    void testCreate_whenTheFieldIsNotRotated_thenNeedsNoRotationFile() {
        assertInstanceOf(IdentityRotation.class, RotationFactory.create(fieldMeta(ClusterANNFieldMeta.ROTATION_NONE), null));
    }

    /** A random Gaussian field gets the block-diagonal family, applied over the region the caller hands it. */
    @Test
    void testCreate_whenTheFieldIsRandomGaussian_thenGivesTheBlockDiagonalRotation() throws IOException {
        // given
        ClusterANNFieldMeta fieldMeta = fieldMeta(ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, 0L);

        // when
        Rotation rotation = RotationFactory.create(fieldMeta, clar());
        float[] dest = new float[DIMENSION];
        rotation.rotate(new float[] { 1f, 0f }, dest);

        // then
        assertInstanceOf(RandomGaussianRotation.class, rotation);
        assertArrayEquals(new float[] { 1f, 3f }, dest, 1e-6f);
    }

    /** Building it reads nothing, so the file it was handed can be empty until a query actually needs the matrix. */
    @Test
    void testCreate_thenReadsNothing() throws IOException {
        // given / when
        Rotation rotation = RotationFactory.create(fieldMeta(ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, 0L), open(0));

        // then
        assertEquals(DIMENSION, rotation.dimension());
    }

    /** An id this reader cannot apply is refused outright: a rotation applied wrongly is silently wrong scoring. */
    @ParameterizedTest(name = "rotationId {0}")
    @ValueSource(ints = { 2, 7, 42 })
    void testCreate_whenTheRotationIsUnsupported_thenThrows(int rotationId) {
        // given / when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> RotationFactory.create(fieldMeta(rotationId, 0L), null)
        );

        // then
        assertTrue(e.getMessage().contains("Unsupported rotationId: " + rotationId), e.getMessage());
    }

    // ---------------------------------------------------------------- helpers

    private static ClusterANNFieldMeta fieldMeta(int rotationId) {
        return fieldMeta(rotationId, ClusterANNFieldMeta.NO_ROTATION);
    }

    private static ClusterANNFieldMeta fieldMeta(int rotationId, long clarOffset) {
        boolean rotated = rotationId != ClusterANNFieldMeta.ROTATION_NONE;
        return new ClusterANNFieldMeta(
            32,                                     // blockSize
            DIMENSION,
            1,                                      // vectorCount
            1,                                      // centroidCount
            VectorSimilarityFunction.EUCLIDEAN,
            1,                                      // docBits
            rotationId,
            0,                                      // quantizerId
            new byte[0],                            // quantizerParams
            0L,                                     // clacOffset
            0L,                                     // clacLength
            0L,                                     // clacCentroidsOffset
            rotated ? 0L : ClusterANNFieldMeta.NO_ROTATION,
            0L,                                     // clapOffset
            0L,                                     // clapLength
            new long[1],                            // clapCentroidOffsets
            new int[1],                             // centroidLengths
            new int[] { 1 },                        // clusterSizes
            clarOffset,
            // The entry rejects a length that disagrees with the offset, so it follows whichever the caller chose.
            clarOffset == ClusterANNFieldMeta.NO_ROTATION ? ClusterANNFieldMeta.NO_ROTATION : 64L,
            null                                    // ordToDoc, which no rotation consults
        );
    }

    /**
     * A single 2×2 block over the identity permutation. Row 0 is {@code (1, 2)}, row 1 is {@code (3, 4)} — asymmetric,
     * so the orientation is observable even from here. The layout is exercised properly in
     * {@code RandomGaussianRotationTests}; this only needs enough for the factory's choice to be visible.
     */
    private IndexInput clar() throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput("clar", IOContext.DEFAULT)) {
            out.writeVInt(1);                       // numBlocks
            out.writeVInt(DIMENSION);               // the one block's width
            for (float value : new float[] { 1f, 2f, 3f, 4f }) {
                out.writeInt(Float.floatToIntBits(value));
            }
            out.writeVInt(DIMENSION);               // its permutation
            for (int index = 0; index < DIMENSION; index++) {
                out.writeInt(index);
            }
        }
        return directory.openInput("clar", IOContext.DEFAULT);
    }

    private IndexInput open(int bytes) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput("empty", IOContext.DEFAULT)) {
            out.writeBytes(new byte[bytes], 0, bytes);
        }
        return directory.openInput("empty", IOContext.DEFAULT);
    }
}
