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
            assertEquals(region1Bytes, offsets.clacVectorOffset(), "region 2 after region 1");
            if (rotated) {
                assertEquals(region1Bytes + region2Bytes, offsets.clacTransformedOffset(), "region 3 after region 2");
            } else {
                assertEquals(-1L, offsets.clacTransformedOffset(), "no region 3 (sentinel offset)");
            }

            try (IndexInput in = dir.openInput(CLAC_FILE, IOContext.DEFAULT)) {
                // Region 1: ordToCentroid.
                in.seek(offsets.clacOffset());
                assertArrayEquals(testCase.ordToCentroid(), readInts(in, testCase.ordToCentroid().length), "region 1 ordToCentroid");

                // Region 2: vector | normSq.
                in.seek(offsets.clacVectorOffset());
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
