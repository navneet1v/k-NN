package org.opensearch.knn.clusterann.writer.rotation;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.reader.rotation.RandomGaussianRotation;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Round-trip tests for the rotation: rotate a vector with the writer, then with the reader over the bytes the writer
 * produced, and require the same answer.
 *
 * <p>This is the check that matters most for a rotation, because every plausible mistake is silent. A transposed
 * matrix is still orthogonal; a scattering permutation instead of a gathering one is still a permutation. Either
 * would score against a different space and return neighbours that look reasonable and are wrong.
 */
class RandomGaussianRotationWriterRoundTripTests {

    private static final String FILE = "rotation";
    private static final int DIMENSION = 40;
    private static final int BLOCK_DIMENSION = 16; // two full blocks then a remainder, so the tail case is covered
    private static final long SEED = 42L;

    @Test
    void writerAndReaderAgree() throws IOException {
        RandomGaussianRotationWriter writer = new RandomGaussianRotationWriter(DIMENSION, BLOCK_DIMENSION, SEED);
        assertEquals(ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, writer.rotationId());

        float[] source = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            source[i] = (float) Math.cos(0.19 * i) * (i % 5 + 1);
        }

        float[] written = new float[DIMENSION];
        writer.rotate(source, written);

        try (Directory directory = new ByteBuffersDirectory()) {
            long length;
            try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
                length = writer.write(out);
                assertEquals(length, out.getFilePointer());
            }
            try (IndexInput in = directory.openInput(FILE, IOContext.DEFAULT)) {
                RandomGaussianRotation read = new RandomGaussianRotation(in.slice("clar", 0, length), DIMENSION);
                float[] readBack = new float[DIMENSION];
                read.rotate(source, readBack);

                for (int i = 0; i < DIMENSION; i++) {
                    assertEquals(written[i], readBack[i], 1e-5f, "dimension " + i);
                }
            }
        }
    }

    /** Orthogonal, so it preserves length — the property that lets distances and rankings survive the rotation. */
    @Test
    void rotationPreservesNorm() {
        RandomGaussianRotationWriter writer = new RandomGaussianRotationWriter(DIMENSION, BLOCK_DIMENSION, SEED);

        float[] source = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            source[i] = (float) ((i % 7) - 3) * 0.5f;
        }
        float[] rotated = new float[DIMENSION];
        writer.rotate(source, rotated);

        assertEquals(norm(source), norm(rotated), 1e-3f);
        assertTrue(norm(source) > 0f, "a degenerate input would make this vacuous");
    }

    /** The identity writer stores nothing, so a field using it records no .clar region at all. */
    @Test
    void identityWritesNothing() throws IOException {
        IdentityRotationWriter identity = new IdentityRotationWriter(DIMENSION);
        assertEquals(ClusterANNFieldMeta.ROTATION_NONE, identity.rotationId());
        assertEquals(0L, identity.write(null));

        float[] source = { 1f, 2f, 3f };
        float[] destination = new float[3];
        new IdentityRotationWriter(3).rotate(source, destination);
        assertEquals(1f, destination[0]);
        assertEquals(3f, destination[2]);
    }

    @Test
    void factoryPicksTheFamilyTheIdNames() {
        assertEquals(
            ClusterANNFieldMeta.ROTATION_NONE,
            RotationWriterFactory.create(ClusterANNFieldMeta.ROTATION_NONE, DIMENSION, SEED).rotationId()
        );
        assertEquals(
            ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN,
            RotationWriterFactory.create(ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN, DIMENSION, SEED).rotationId()
        );
    }

    private static float norm(float[] vector) {
        float sum = 0f;
        for (float value : vector) {
            sum += value * value;
        }
        return (float) Math.sqrt(sum);
    }
}
