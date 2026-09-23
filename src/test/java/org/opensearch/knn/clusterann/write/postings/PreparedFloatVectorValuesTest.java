/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.FloatVectorValues;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Unit tests for {@link PreparedFloatVectorValues}: it re-indexes a source's vectors to a subset of ordinals
 * (in the given order), rotates each into a single scratch buffer, is addressed positionally via
 * {@code vectorValue(int)}, and rejects {@code iterator()} since a cluster has no document space.
 */
class PreparedFloatVectorValuesTest {

    private static final int DIMENSION = 2;

    private static FloatVectorValues source() {
        return FloatVectorValues.fromFloats(
            List.of(new float[] { 0f, 0f }, new float[] { 1f, 1f }, new float[] { 2f, 2f }, new float[] { 3f, 3f }),
            DIMENSION
        );
    }

    private static Rotation identity() {
        return RotationFormats.create(ROTATION_NONE, DIMENSION);
    }

    @Test
    void reindexesToOrdinalsIntoScratch() throws IOException {
        final int[] ordinals = { 3, 1 }; // a reordered subset
        final PreparedFloatVectorValues view = new PreparedFloatVectorValues(source(), ordinals, identity());

        assertEquals(2, view.size(), "size is the number of ordinals");
        assertEquals(DIMENSION, view.dimension(), "dimension is the source's");
        assertArrayEquals(new float[] { 3f, 3f }, view.vectorValue(0), 0f, "position 0 -> source[3]");
        assertArrayEquals(new float[] { 1f, 1f }, view.vectorValue(1), 0f, "position 1 -> source[1]");
        assertThrows(IndexOutOfBoundsException.class, () -> view.vectorValue(2), "position out of range");
    }

    @Test
    void copyProducesAnEquivalentView() throws IOException {
        final int[] ordinals = { 2, 0 };
        final PreparedFloatVectorValues view = new PreparedFloatVectorValues(source(), ordinals, identity());

        final FloatVectorValues copy = view.copy();
        assertEquals(view.size(), copy.size());
        assertEquals(view.dimension(), copy.dimension());
        for (int i = 0; i < view.size(); i++) {
            assertArrayEquals(view.vectorValue(i).clone(), copy.vectorValue(i).clone(), 0f, "copy matches at " + i);
        }
    }

    @Test
    void iteratorIsUnsupported() {
        final PreparedFloatVectorValues view = new PreparedFloatVectorValues(source(), new int[] { 0, 1, 2 }, identity());
        // Members are ordinals in nearest-first order, which a monotonic DocIdSetIterator can't represent;
        // the view is addressed positionally via vectorValue(int) instead.
        assertThrows(UnsupportedOperationException.class, view::iterator);
    }

    @Test
    void rotatesEachVectorIntoScratch() throws IOException {
        // A non-identity rotation: the view must return each source vector rotated by it, not a raw copy.
        final Rotation rotation = RotationFormats.create(ROTATION_RANDOM_GAUSSIAN, DIMENSION);
        final PreparedFloatVectorValues view = new PreparedFloatVectorValues(source(), new int[] { 3, 1 }, rotation);

        final float[] expected3 = new float[DIMENSION];
        rotation.rotate(new float[] { 3f, 3f }, expected3);
        final float[] expected1 = new float[DIMENSION];
        rotation.rotate(new float[] { 1f, 1f }, expected1);
        assertArrayEquals(expected3, view.vectorValue(0), 0f, "position 0 -> rotated source[3]");
        assertArrayEquals(expected1, view.vectorValue(1), 0f, "position 1 -> rotated source[1]");
    }
}
