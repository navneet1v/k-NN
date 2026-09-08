/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
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
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Fast, isolated unit tests for {@link CentroidVectorValues} (JUnit 5).
 *
 * <p>Records are fixed width, so a centroid is reached by arithmetic rather than by scanning — which means the width
 * calculation is the whole of the behaviour. Bytes come from {@link #write}, an independent encoder, and the values
 * are derived from the ordinal so a mis-seek shows up as a wrong number rather than as a pass.
 */
class CentroidVectorValuesTests {

    private static final int DIMENSION = 4;
    private static final int NUM_CENTROIDS = 5;
    private static final String FILE = "clac";

    /** Bytes the norm-carrying raw region occupies, and so where the region written after it begins: 5 × 5 floats. */
    private static final long RAW_REGION_BYTES = (long) NUM_CENTROIDS * (DIMENSION + 1) * Float.BYTES;

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- geometry

    @Test
    void testGeometry_thenReportsWhatItWasBuiltWith() throws IOException {
        // given / when
        CentroidVectorValues centroids = centroids(0L, true);

        // then
        assertEquals(DIMENSION, centroids.dimension());
        assertEquals(NUM_CENTROIDS, centroids.size());
        assertEquals(VectorEncoding.FLOAT32, centroids.getEncoding(), "inherited, but the codes are floats");
    }

    // ---------------------------------------------------------------- reading

    /** Random access: every centroid is one seek away, so they read correctly in any order. */
    @ParameterizedTest(name = "centroid {0}")
    @ValueSource(ints = { 0, 1, 4 })
    void testVectorValue_thenReadsThatCentroid(int ordinal) throws IOException {
        // given
        CentroidVectorValues centroids = centroids(0L, true);

        // when
        float[] vector = centroids.vectorValue(ordinal);

        // then
        assertArrayEquals(expectedVector(ordinal), vector, "centroid " + ordinal);
        assertEquals(expectedNorm(ordinal), centroids.norm(), "norm of centroid " + ordinal);
    }

    /** Backwards and repeated reads work too: the cursor holds no state beyond where it last was. */
    @Test
    void testVectorValue_whenReadOutOfOrder_thenStillReadsEachCentroid() throws IOException {
        // given
        CentroidVectorValues centroids = centroids(0L, true);

        // when / then
        for (int ordinal : new int[] { 3, 0, 4, 0, 2 }) {
            assertArrayEquals(expectedVector(ordinal), centroids.vectorValue(ordinal), "centroid " + ordinal);
            assertEquals(expectedNorm(ordinal), centroids.norm(), "norm of centroid " + ordinal);
        }
    }

    /**
     * A region need not begin at the start of the file: a field's rotated centroids follow its raw ones, so the
     * cursor reads whichever region it was sliced over.
     */
    @Test
    void testVectorValue_whenTheRegionIsOffset_thenReadsFromThere() throws IOException {
        // given — the raw region occupies the first RAW_REGION_BYTES, the rotated region follows
        CentroidVectorValues centroids = centroids(RAW_REGION_BYTES, true);

        // when / then — the encoder writes the transformed region with negated values, so a read at the wrong
        // offset would come back positive
        for (int ordinal = 0; ordinal < NUM_CENTROIDS; ordinal++) {
            float[] expected = expectedVector(ordinal);
            for (int i = 0; i < expected.length; i++) {
                expected[i] = -expected[i];
            }
            assertArrayEquals(expected, centroids.vectorValue(ordinal), "transformed centroid " + ordinal);
        }
    }

    /** The buffer is reused, so two reads hand back the same array — a caller must consume before asking again. */
    @Test
    void testVectorValue_thenReusesItsBuffer() throws IOException {
        // given
        CentroidVectorValues centroids = centroids(0L, true);

        // when
        float[] first = centroids.vectorValue(0);
        float[] second = centroids.vectorValue(1);

        // then
        assertSame(first, second, "the buffer is documented as reused; a copy per call would be a silent allocation");
        assertArrayEquals(expectedVector(1), first, "and it holds the most recent centroid");
    }

    @ParameterizedTest(name = "ordinal {0}")
    @ValueSource(ints = { -1, 5, 99 })
    void testVectorValue_whenOrdinalIsOutOfRange_thenThrows(int ordinal) throws IOException {
        // given
        CentroidVectorValues centroids = centroids(0L, true);

        // when
        IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class, () -> centroids.vectorValue(ordinal));

        // then
        assertEquals("Centroid ordinal must be in [0, " + NUM_CENTROIDS + "), got: " + ordinal, e.getMessage());
    }

    // ---------------------------------------------------------------- norms

    /**
     * A region without norms is narrower per record, so reading one is a different stride — and asking for a norm it
     * does not carry is a mistake rather than something to compute quietly.
     */
    @Test
    void testNorm_whenTheRegionCarriesNone_thenThrowsAndTheStrideIsNarrower() throws IOException {
        // given
        CentroidVectorValues centroids = centroids(0L, false);

        // when / then — 4 floats per record rather than 5, so centroid 1 starts 16 bytes in
        assertArrayEquals(expectedVector(1), centroids.vectorValue(1), "a normless region reads on a narrower stride");
        IllegalStateException e = assertThrows(IllegalStateException.class, centroids::norm);
        assertEquals("This centroid region carries no norm", e.getMessage());
    }

    // ---------------------------------------------------------------- copy

    /**
     * {@code copy()} exists so concurrent readers do not share a cursor. Each has its own file pointer and its own
     * buffer, so one reading centroid 4 cannot move or overwrite another mid-read.
     */
    @Test
    void testCopy_thenIsIndependentOfTheOriginal() throws IOException {
        // given
        CentroidVectorValues original = centroids(0L, true);
        CentroidVectorValues copy = (CentroidVectorValues) original.copy();

        // when
        float[] fromOriginal = original.vectorValue(0);
        float[] fromCopy = copy.vectorValue(4);

        // then
        assertNotSame(fromOriginal, fromCopy, "each cursor owns its buffer");
        assertArrayEquals(expectedVector(0), fromOriginal, "the copy's read must not disturb the original's");
        assertArrayEquals(expectedVector(4), fromCopy);
        assertEquals(expectedNorm(0), original.norm());
        assertEquals(expectedNorm(4), copy.norm());
    }

    @Test
    void testCopy_thenKeepsTheSameGeometry() throws IOException {
        // given
        CentroidVectorValues original = centroids(0L, true);

        // when
        FloatVectorValues copy = original.copy();

        // then
        assertEquals(original.dimension(), copy.dimension());
        assertEquals(original.size(), copy.size());
    }

    // ---------------------------------------------------------------- helpers

    /** Values follow the ordinal, so a read from the wrong record is a wrong number rather than a plausible one. */
    private static float[] expectedVector(int ordinal) {
        float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = ordinal * 10f + i;
        }
        return vector;
    }

    private static float expectedNorm(int ordinal) {
        return 100f + ordinal;
    }

    /**
     * A cursor over one region of the two the file holds, sliced out at {@code offset} — which is how a caller hands
     * this class a region rather than a file and an offset to keep straight.
     */
    private CentroidVectorValues centroids(long offset, boolean hasNorm) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        write(directory, hasNorm);
        IndexInput input = directory.openInput(FILE, IOContext.DEFAULT);
        long regionBytes = (long) NUM_CENTROIDS * (DIMENSION + (hasNorm ? 1 : 0)) * Float.BYTES;
        return new CentroidVectorValues(input.slice("region", offset, regionBytes), NUM_CENTROIDS, DIMENSION, hasNorm);
    }

    /**
     * A raw region followed by a transformed one, the latter negated so a read at the wrong offset is obvious. Both
     * regions carry norms when {@code hasNorm}, which is what makes the two offsets differ.
     */
    private static void write(Directory directory, boolean hasNorm) throws IOException {
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int sign : new int[] { 1, -1 }) {
                for (int ordinal = 0; ordinal < NUM_CENTROIDS; ordinal++) {
                    for (float value : expectedVector(ordinal)) {
                        out.writeInt(Float.floatToIntBits(sign * value));
                    }
                    if (hasNorm) {
                        out.writeInt(Float.floatToIntBits(expectedNorm(ordinal)));
                    }
                }
            }
        }
    }
}
