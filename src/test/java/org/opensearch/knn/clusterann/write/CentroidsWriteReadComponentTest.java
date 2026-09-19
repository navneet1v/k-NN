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
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.read.CentroidVectorValues;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidData;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidOffsets;

import java.io.IOException;

import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Component test spanning the {@code .clac} write and read sides: centroids written by {@link CentroidsWriter}
 * into region 3 (the rotated/quantization-space centroids the reader scans) are read back through the reader
 * primitive {@link CentroidVectorValues}. Run for both rotation families — a block-diagonal rotation and the
 * identity one (an unrotated field, whose region 3 is a copy of its input-space centroids) — confirming the
 * two sides agree on the {@code vector | normSq} layout and stride.
 */
class CentroidsWriteReadComponentTest {

    private static final String CLAC_FILE = "clusterann.clac";
    private static final int DIMENSION = 6;

    @ParameterizedTest(name = "rotated={0}")
    @ValueSource(booleans = { false, true })
    void rotatedCentroids_writtenThenReadBack(final boolean rotated) throws IOException {
        final float[][] centroids = { { 1f, 2f, 3f, 4f, 5f, 6f }, { -2f, 0.5f, 7f, -1f, 3f, 2f }, { 9f, 8f, 7f, 6f, 5f, 4f } };
        final CentroidData data = new CentroidData(new int[] { 0, 1, 2, 1, 0 }, centroids);
        final Rotation rotation = RotationFormats.create(rotated ? ROTATION_RANDOM_GAUSSIAN : ROTATION_NONE, DIMENSION);

        try (Directory dir = new ByteBuffersDirectory()) {
            final CentroidOffsets offsets;
            try (IndexOutput out = dir.createOutput(CLAC_FILE, IOContext.DEFAULT)) {
                offsets = CentroidsWriter.write(out, data, rotation);
            }

            if (!rotated) {
                // Unrotated field: no region 3 — the reader would scan region 2 (input space) directly.
                assertEquals(-1L, offsets.clacRotatedCentroidsOffset(), "unrotated field has no region 3 (sentinel offset)");
                return;
            }

            try (IndexInput in = dir.openInput(CLAC_FILE, IOContext.DEFAULT)) {
                // Region 3 is what the reader scans: (dimension + 1) floats per centroid — vector, then normSq.
                final long region3Bytes = (long) centroids.length * (DIMENSION + 1) * Float.BYTES;
                final IndexInput region3 = in.slice("region3", offsets.clacRotatedCentroidsOffset(), region3Bytes);
                final CentroidVectorValues reader = new CentroidVectorValues(region3, centroids.length, DIMENSION);

                for (int c = 0; c < centroids.length; c++) {
                    final float[] expected = new float[DIMENSION];
                    rotation.rotate(centroids[c], expected);
                    assertArrayEquals(expected, reader.vectorValue(c).clone(), 0f, "centroid " + c);
                    assertEquals(dotSelf(expected), reader.norm(), 0f, "normSq " + c);
                }
            }
        }
    }

    private static float dotSelf(final float[] vector) {
        float sum = 0f;
        for (final float value : vector) {
            sum += value * value;
        }
        return sum;
    }
}
