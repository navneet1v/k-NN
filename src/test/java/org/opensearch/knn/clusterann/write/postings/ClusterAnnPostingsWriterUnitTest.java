/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.write.QuantizationParams;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter.ClusterMembers;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter.ClusterRegion;
import org.opensearch.knn.clusterann.ClusteringResult;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;

/**
 * Unit test for {@link ClusterAnnPostingsWriter}'s orchestration, isolated from real quantization: the cluster
 * writer factory {@code ClusterWriterFactory#newWriter} is stubbed with a static mock that returns a recording
 * writer, so the assertions cover only what the orchestrator owns — the partition (primary + secondary assignment per
 * centroid, unassigned ordinals skipped), the nearest-first ordering handed to each cluster, and the {@link
 * PostingsRegions} bookkeeping (contiguous regions summing to the whole {@code .clap}). How a cluster region is
 * laid out is out of scope here and covered end-to-end in {@link ClusterAnnPostingsWriterTest}.
 */
class ClusterAnnPostingsWriterUnitTest {

    private static final int DIMENSION = 2;
    private static final int BLOCK_SIZE = 2;
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;

    // vectorValue[i] = {i*10, 0}, so a recorded first component maps back to its ordinal (component / 10).
    private static final float[][] VECTORS = { { 0f, 0f }, { 10f, 0f }, { 20f, 0f }, { 30f, 0f } };
    private static final float[][] CENTROIDS = { { 0f, 0f }, { 20f, 0f }, { 99f, 0f } };

    /**
     * c0 primary: v0,v1. c1 primary: v2, secondary: v0. c2: empty. v3: unassigned (skipped).
     * Distances are set so the sort reorders c1 to put the near primary before the far secondary.
     */
    private static ClusteringResult clustering() {
        return new ClusteringResult(
            CENTROIDS,
            new int[] { 0, 0, 1, -1 },
            new float[] { 0.5f, 1.5f, 0.2f, Float.NaN },
            new int[] { 1, -1, -1, -1 },
            new float[] { 3.0f, Float.NaN, Float.NaN, Float.NaN },
            3
        );
    }

    @Test
    void write_partitionsSortsAndReportsRegions() throws IOException {
        final List<RecordingClusterWriter> created = new ArrayList<>();
        final PostingsRegions layout;

        try (MockedStatic<ClusterWriterFactory> mocked = Mockito.mockStatic(ClusterWriterFactory.class)) {
            mocked.when(() -> ClusterWriterFactory.newWriter(any(), anyInt(), anyInt(), any(), any())).thenAnswer(inv -> {
                final RecordingClusterWriter w = new RecordingClusterWriter();
                created.add(w);
                return w;
            });

            try (Directory dir = new ByteBuffersDirectory()) {
                try (IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
                    layout = new ClusterAnnPostingsWriter(
                        BLOCK_SIZE,
                        new QuantizationParams(0, (byte) 1),
                        RotationFormats.create(ROTATION_NONE, DIMENSION)
                    ).write(out, clustering(), source(), METRIC);
                }
            }
        }

        // A cluster writer is built once per centroid, empty ones included.
        assertEquals(3, created.size(), "one cluster writer per centroid");

        // Partition: primary + secondary assignment per centroid; v3 (unassigned) skipped; c2 empty.
        assertArrayEquals(new int[] { 2, 2, 0 }, layout.clusterSizes());

        // Ordering handed to each non-empty cluster, nearest-first.
        assertArrayEquals(new int[] { 0, 1 }, created.get(0).members(), "c0 members nearest-first");
        assertArrayEquals(new int[] { 2, 0 }, created.get(1).members(), "c1 primary before far secondary assignment");
        assertTrue(created.get(2).wroteNothing(), "empty cluster writes no members");

        // Regions: contiguous, in centroid order, together spanning the whole .clap; empty cluster adds nothing.
        assertEquals(0L, layout.clapOffset());
        assertEquals(3, layout.centroidOffsets().length);
        assertEquals(0L, layout.centroidOffsets()[0]);
        assertEquals(layout.centroidOffsets()[0] + layout.centroidLengths()[0], layout.centroidOffsets()[1]);
        assertEquals(layout.centroidOffsets()[1] + layout.centroidLengths()[1], layout.centroidOffsets()[2]);
        assertEquals(0, layout.centroidLengths()[2], "empty cluster region has zero length");
        assertEquals(
            layout.centroidLengths()[0] + layout.centroidLengths()[1] + layout.centroidLengths()[2],
            layout.clapLength(),
            "cluster regions span the whole .clap"
        );
    }

    /**
     * For inner product and cosine the closest members carry the largest squared-L2 distance, so nearest-first
     * is descending: each cluster's members are handed to the writer farthest-distance-first — the reverse of
     * the L2 ordering in {@link #write_partitionsSortsAndReportsRegions}.
     */
    @Test
    void write_ordersFarthestFirstForInnerProduct() throws IOException {
        final List<RecordingClusterWriter> created = new ArrayList<>();

        try (MockedStatic<ClusterWriterFactory> mocked = Mockito.mockStatic(ClusterWriterFactory.class)) {
            mocked.when(() -> ClusterWriterFactory.newWriter(any(), anyInt(), anyInt(), any(), any())).thenAnswer(inv -> {
                final RecordingClusterWriter w = new RecordingClusterWriter();
                created.add(w);
                return w;
            });

            try (Directory dir = new ByteBuffersDirectory()) {
                try (IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
                    new ClusterAnnPostingsWriter(
                        BLOCK_SIZE,
                        new QuantizationParams(0, (byte) 1),
                        RotationFormats.create(ROTATION_NONE, DIMENSION)
                    ).write(out, clustering(), source(), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
                }
            }
        }

        // Reverse of the L2 case: c0 by descending distance (v1 d=1.5 before v0 d=0.5); c1 far secondary before near primary.
        assertArrayEquals(new int[] { 1, 0 }, created.get(0).members(), "c0 members farthest-first");
        assertArrayEquals(new int[] { 0, 2 }, created.get(1).members(), "c1 far secondary before near primary");
        assertTrue(created.get(2).wroteNothing(), "empty cluster writes no members");
    }

    /**
     * When the field's region does not start at 0 (a prior field already wrote into {@code .clap}), the
     * per-centroid offsets are relative to the region start (clapOffset), not absolute file positions — the
     * reader slices the region at clapOffset and seeks within that slice. Absolute offsets would be wrong here.
     */
    @Test
    void write_centroidOffsetsAreRelativeToRegionStart() throws IOException {
        final int prefix = 64; // bytes a previous field already wrote, so the region starts past 0
        final PostingsRegions layout;

        try (MockedStatic<ClusterWriterFactory> mocked = Mockito.mockStatic(ClusterWriterFactory.class)) {
            mocked.when(() -> ClusterWriterFactory.newWriter(any(), anyInt(), anyInt(), any(), any()))
                .thenAnswer(inv -> new RecordingClusterWriter());

            try (Directory dir = new ByteBuffersDirectory()) {
                try (IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
                    out.writeBytes(new byte[prefix], prefix); // advance past a non-zero region start
                    layout = new ClusterAnnPostingsWriter(
                        BLOCK_SIZE,
                        new QuantizationParams(0, (byte) 1),
                        RotationFormats.create(ROTATION_NONE, DIMENSION)
                    ).write(out, clustering(), source(), METRIC);
                }
            }
        }

        // clapOffset is the absolute region start; the per-centroid offsets are relative to it.
        assertEquals(prefix, layout.clapOffset(), "clapOffset is the absolute region start");
        assertEquals(0L, layout.centroidOffsets()[0], "first centroid offset is relative to the region, not the absolute file position");

        // Offsets are relative and contiguous from the region start, and together span the region.
        long expected = 0L;
        for (int c = 0; c < layout.centroidOffsets().length; c++) {
            assertEquals(expected, layout.centroidOffsets()[c], "centroid " + c + " offset is relative and contiguous");
            expected += layout.centroidLengths()[c];
        }
        assertEquals(expected, layout.clapLength(), "relative offsets and lengths span the field's region");
    }

    private static FloatVectorValues source() {
        return FloatVectorValues.fromFloats(List.of(VECTORS[0], VECTORS[1], VECTORS[2], VECTORS[3]), DIMENSION);
    }

    /**
     * Stands in for a real cluster writer: records the member ordinals handed to its one {@code write} call
     * (so partition and ordering can be checked) and writes a marker per member so the cluster region has a
     * real, non-empty length. An empty cluster records nothing and occupies no bytes.
     */
    private static final class RecordingClusterWriter implements ClusterWriter {

        private int[] members;

        int[] members() {
            return members;
        }

        boolean wroteNothing() {
            return members == null;
        }

        @Override
        public ClusterRegion write(final IndexOutput out, final FloatVectorValues vectors, final ClusterMembers members)
            throws IOException {
            final long start = out.getFilePointer();
            if (members.ordinals().length > 0) {
                this.members = members.ordinals().clone();
                for (final int ordinal : members.ordinals()) {
                    out.writeInt(ordinal);
                }
            }
            return new ClusterRegion(start, out.getFilePointer() - start);
        }
    }
}
