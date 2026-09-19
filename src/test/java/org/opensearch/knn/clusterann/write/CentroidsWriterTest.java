/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidData;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidOffsets;

import java.io.IOException;
import java.util.stream.Stream;

import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Byte-layout tests for {@link CentroidsWriter}, one case per {@code (centroidCount, rotation)} shape. Each
 * case is written and then read back region by region against the spec layout: region 1 ({@code ordToCentroid}
 * as ints), region 2 ({@code vector | normSq}), and — only when the field is rotated — region 3
 * ({@code vector | normSq}). Region offsets and the total length are derived from the dimensions/counts rather
 * than hard-coded, and {@code normSq} is recomputed independently to confirm the writer derives it correctly.
 *
 * <p>Those cases all write at position 0, so they say nothing about whether the inner offsets are relative to the
 * field's region or absolute in the file — at 0 the two coincide.
 * {@link #write_innerOffsetsAreRelativeToTheFieldsRegion} covers the difference.
 */
class CentroidsWriterTest {

    private static final String CLAC_FILE = "clusterann.clac";

    /** One centroid layout to serialize; {@code rotate} marks a rotated field (region 3 present). */
    private record Case(String name, int[] ordToCentroid, float[][] centroids, boolean rotate) {
        @Override
        public String toString() {
            return name;
        }
    }

    private static Stream<Case> cases() {
        return Stream.of(
            new Case("1 centroid, unrotated", new int[] { 0, 0, 0 }, new float[][] { { 1f, 2f } }, false),
            new Case("1 centroid, rotated", new int[] { 0, 0, 0 }, new float[][] { { 1f, 2f } }, true),
            new Case("2 centroids, unrotated", new int[] { 0, 1, 0, 1, 0 }, new float[][] { { 1f, 2f }, { 3f, 4f } }, false),
            new Case("2 centroids, rotated", new int[] { 0, 1, 0, 1, 0 }, new float[][] { { 1f, 2f }, { 3f, 4f } }, true),
            new Case("empty field", new int[0], new float[0][], false)
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void write_laysOutRegionsPerSpec(Case testCase) throws IOException {
        final CentroidData data = new CentroidData(testCase.ordToCentroid(), testCase.centroids());
        final int numCentroids = testCase.centroids().length;
        final int dim = numCentroids == 0 ? 0 : testCase.centroids()[0].length;
        final Rotation rotation = RotationFormats.create(testCase.rotate() ? ROTATION_RANDOM_GAUSSIAN : ROTATION_NONE, dim);
        final boolean rotated = testCase.rotate();

        try (Directory dir = new ByteBuffersDirectory()) {
            final CentroidOffsets offsets;
            final long length;
            try (IndexOutput out = dir.createOutput(CLAC_FILE, IOContext.DEFAULT)) {
                offsets = CentroidsWriter.write(out, data, rotation);
                length = out.getFilePointer();
            }

            // Region offsets follow from the sizes: region 1 is int-per-ord, region 2 is (vector + normSq) per
            // centroid, region 3 (rotated fields only) is (vector + normSq) per centroid.
            final long region1Bytes = (long) testCase.ordToCentroid().length * Integer.BYTES;
            final long region2Bytes = (long) numCentroids * (dim + 1) * Float.BYTES;
            assertEquals(0L, offsets.clacOffset(), "region 1 at start");
            assertEquals(region1Bytes, offsets.clacCentroidsOffset(), "region 2 after region 1");
            if (rotated) {
                assertEquals(region1Bytes + region2Bytes, offsets.clacRotatedCentroidsOffset(), "region 3 after region 2");
            } else {
                assertEquals(-1L, offsets.clacRotatedCentroidsOffset(), "no region 3 (sentinel offset)");
            }

            try (IndexInput in = dir.openInput(CLAC_FILE, IOContext.DEFAULT)) {
                // Region 1: ordToCentroid.
                in.seek(offsets.clacOffset());
                assertArrayEquals(testCase.ordToCentroid(), readInts(in, testCase.ordToCentroid().length), "region 1 ordToCentroid");

                // Region 2: vector | normSq.
                in.seek(offsets.clacCentroidsOffset());
                for (int c = 0; c < numCentroids; c++) {
                    assertArrayEquals(testCase.centroids()[c], readFloats(in, dim), 0f, "centroid " + c);
                    assertEquals(normSq(testCase.centroids()[c]), Float.intBitsToFloat(in.readInt()), 0f, "normSq " + c);
                }

                // Region 3 (rotated fields only): vector | normSq. The writer rotates each centroid with the
                // field's rotation, so the expected bytes are that same rotation applied here.
                if (rotated) {
                    for (int c = 0; c < numCentroids; c++) {
                        final float[] expected = new float[dim];
                        rotation.rotate(testCase.centroids()[c], expected);
                        assertArrayEquals(expected, readFloats(in, dim), 0f, "rotated centroid " + c);
                        assertEquals(normSq(expected), Float.intBitsToFloat(in.readInt()), 0f, "rotated normSq " + c);
                    }
                }
                assertEquals(length, in.length(), "reported length matches file");
                assertEquals(in.length(), in.getFilePointer(), "read every byte");
            }
        }
    }

    /**
     * The same layout written at a non-zero position, which is where every field but the first one lives.
     *
     * <p>{@link #write_laysOutRegionsPerSpec} cannot catch a confusion between absolute and relative offsets: it writes
     * at position 0, where the two are equal. Here a preamble stands in for an earlier field, and the regions are read
     * back the way {@code KNN1030ClusterANNVectorsReader} reads them — slice the field's region at {@code clacOffset},
     * then slice regions 2 and 3 out of <em>that</em>. An absolute inner offset is applied twice under that scheme and
     * lands at {@code clacOffset + offset}, so it fails here rather than at query time.
     */
    @Test
    void write_innerOffsetsAreRelativeToTheFieldsRegion() throws IOException {
        final int[] ordToCentroid = { 0, 1, 0, 1, 0 };
        final float[][] centroids = { { 1f, 2f }, { 3f, 4f } };
        final int dim = 2;
        final CentroidData data = new CentroidData(ordToCentroid, centroids);
        final Rotation rotation = RotationFormats.create(ROTATION_RANDOM_GAUSSIAN, dim);

        // An earlier field's bytes: anything, as long as this field does not begin at 0.
        final long preambleBytes = 37L;

        try (Directory dir = new ByteBuffersDirectory()) {
            final CentroidOffsets offsets;
            final long fieldBytes;
            try (IndexOutput out = dir.createOutput(CLAC_FILE, IOContext.DEFAULT)) {
                for (long b = 0; b < preambleBytes; b++) {
                    out.writeByte((byte) 0xAB);
                }
                offsets = CentroidsWriter.write(out, data, rotation);
                fieldBytes = out.getFilePointer() - offsets.clacOffset();
            }

            final long region1Bytes = (long) ordToCentroid.length * Integer.BYTES;
            final long region2Bytes = (long) centroids.length * (dim + 1) * Float.BYTES;
            assertEquals(preambleBytes, offsets.clacOffset(), "the field's region is absolute: it starts past the preamble");
            assertEquals(region1Bytes, offsets.clacCentroidsOffset(), "region 2 is relative to the field's region, not to the file");
            assertEquals(
                region1Bytes + region2Bytes,
                offsets.clacRotatedCentroidsOffset(),
                "region 3 is relative to the field's region, not to the file"
            );

            try (IndexInput file = dir.openInput(CLAC_FILE, IOContext.DEFAULT)) {
                // The reader's two-step slicing, which is what makes the inner offsets relative.
                final IndexInput field = file.slice("field", offsets.clacOffset(), fieldBytes);

                final IndexInput region1 = field.slice("ordToCentroid", 0L, region1Bytes);
                assertArrayEquals(ordToCentroid, readInts(region1, ordToCentroid.length), "region 1 ordToCentroid");

                final IndexInput region2 = field.slice("centroids", offsets.clacCentroidsOffset(), region2Bytes);
                for (int c = 0; c < centroids.length; c++) {
                    assertArrayEquals(centroids[c], readFloats(region2, dim), 0f, "centroid " + c);
                    assertEquals(normSq(centroids[c]), Float.intBitsToFloat(region2.readInt()), 0f, "normSq " + c);
                }

                final IndexInput region3 = field.slice("centroids-rotated", offsets.clacRotatedCentroidsOffset(), region2Bytes);
                for (int c = 0; c < centroids.length; c++) {
                    final float[] expected = new float[dim];
                    rotation.rotate(centroids[c], expected);
                    assertArrayEquals(expected, readFloats(region3, dim), 0f, "rotated centroid " + c);
                    assertEquals(normSq(expected), Float.intBitsToFloat(region3.readInt()), 0f, "rotated normSq " + c);
                }
            }
        }
    }

    /** Squared L2 norm, recomputed here to check the writer's stored value independently. */
    private static float normSq(float[] vector) {
        float sum = 0f;
        for (final float value : vector) {
            sum += value * value;
        }
        return sum;
    }

    private static int[] readInts(IndexInput in, int n) throws IOException {
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            values[i] = in.readInt();
        }
        return values;
    }

    private static float[] readFloats(IndexInput in, int n) throws IOException {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = Float.intBitsToFloat(in.readInt());
        }
        return values;
    }
}
