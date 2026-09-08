/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.orchestration;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtil;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.reader.CentroidVectorValues;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.reader.Clusters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Fast, isolated unit tests for {@link CentroidPlanner} (JUnit 5 + Mockito).
 *
 * <p>The planner is geometry and ordering, so the centroids are written by hand and chosen so that every expected order
 * can be worked out on paper. {@link #CENTROIDS} in particular is arranged so the three metrics disagree about it —
 * a planner that ranked by the wrong quantity would produce a different order rather than a plausible one.
 *
 * <p>Only {@link Clusters#centroids()}, {@link Clusters#clusterSize} and the field's similarity are stubbed. Mockito
 * because {@link Clusters} is final and owns file inputs; the centroids themselves are real bytes read by a real
 * {@link CentroidVectorValues}, since the seek arithmetic is part of what is being exercised.
 */
class CentroidPlannerTests {

    private static final int DIMENSION = 2;
    private static final String FILE = "clac";

    /** Ranked against {@link #QUERY}, these give a different order under each metric. See the tests for the sums. */
    private static final float[][] CENTROIDS = { { 10f, 0f }, { 0f, 1f }, { 3f, 0f }, { 0.5f, 5f } };

    private static final float[] QUERY = { 1f, 0f };

    /** Every cluster occupied, so nothing is dropped for being empty. */
    private static final int[] ALL_OCCUPIED = { 4, 4, 4, 4 };

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- ranking

    /**
     * EUCLIDEAN ranks by squared distance: {@code d² = ‖q‖² − 2⟨q,c⟩ + ‖c‖²}, giving 81, 2, 4 and 25.25 for the four
     * centroids. Note that ordinal 1 wins despite pointing the other way from the query — which is the point of using
     * a distance rather than an inner product.
     */
    @Test
    void testPlan_whenEuclidean_thenRanksByDistanceClosestFirst() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, ALL_OCCUPIED);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 4));

        // then
        assertArrayEquals(new int[] { 1, 2, 3, 0 }, probes);
    }

    /**
     * DOT_PRODUCT ranks by inner product — 10, 0, 3 and 0.5 — which reverses EUCLIDEAN's verdict on the same centroids:
     * the far one at (10, 0) is the best inner product and the worst distance.
     */
    @Test
    void testPlan_whenDotProduct_thenRanksByInnerProduct() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.DOT_PRODUCT, CENTROIDS, ALL_OCCUPIED);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 4));

        // then
        assertArrayEquals(new int[] { 0, 2, 3, 1 }, probes);
    }

    /** Maximum inner product orders centroids the same way as the plain inner product; only the score scaling differs. */
    @Test
    void testPlan_whenMaximumInnerProduct_thenRanksAsDotProductDoes() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, CENTROIDS, ALL_OCCUPIED);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 4));

        // then
        assertArrayEquals(new int[] { 0, 2, 3, 1 }, probes);
    }

    /**
     * COSINE has to divide by ‖c‖, and this pair is built so that skipping the division changes the answer: (5, 5) has
     * the larger inner product with the query — 5 against 1 — while (1, 0) is perfectly aligned with it. Cosine ranks
     * the aligned one first, a raw inner product ranks the long one first.
     */
    @Test
    void testPlan_whenCosine_thenRanksByAngleRatherThanByRawInnerProduct() throws IOException {
        // given
        float[][] centroids = { { 5f, 5f }, { 1f, 0f } };
        int[] occupied = { 4, 4 };

        // when
        int[] byCosine = CentroidPlanner.plan(clusters(VectorSimilarityFunction.COSINE, centroids, occupied), QUERY, new PlanParams(1, 2));
        int[] byDot = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.DOT_PRODUCT, centroids, occupied),
            QUERY,
            new PlanParams(1, 2)
        );

        // then
        assertArrayEquals(new int[] { 1, 0 }, byCosine, "the aligned centroid wins on angle");
        assertArrayEquals(new int[] { 0, 1 }, byDot, "and loses on inner product, so the division is doing the work");
    }

    /** The query is shared across every cluster of a scan, so ranking must leave it as it was. */
    @Test
    void testPlan_thenLeavesTheQueryUntouched() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, ALL_OCCUPIED);
        float[] query = QUERY.clone();

        // when
        CentroidPlanner.plan(clusters, query, new PlanParams(1, 4));

        // then
        assertArrayEquals(QUERY, query);
    }

    // ---------------------------------------------------------------- bounds

    /** The ceiling is the heap's capacity, so the clusters beyond it are never ordered — and never named. */
    @Test
    void testPlan_thenNamesAtMostMaxProbes() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, ALL_OCCUPIED);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 2));

        // then
        assertArrayEquals(new int[] { 1, 2 }, probes, "the two closest, still closest-first");
    }

    /** A ceiling above the field is not an error; the plan is simply everything there is to probe. */
    @Test
    void testPlan_whenMaxProbesExceedsTheField_thenNamesEveryOccupiedCluster() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, ALL_OCCUPIED);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 100));

        // then
        assertArrayEquals(new int[] { 1, 2, 3, 0 }, probes);
    }

    /**
     * The floor holds when the field can supply it. Nothing cuts below it yet, so this asserts the invariant rather
     * than a behaviour — the point being that a pruner added later cannot quietly violate it.
     */
    @Test
    void testPlan_thenNamesAtLeastMinProbes() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, ALL_OCCUPIED);
        PlanParams params = new PlanParams(4, 10);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, params);

        // then
        assertTrue(probes.length >= params.minProbes(), "got " + probes.length);
    }

    /** A floor cannot conjure clusters: a field with fewer occupied clusters than the floor yields what it has. */
    @Test
    void testPlan_whenTheFieldHasFewerClustersThanTheFloor_thenNamesWhatItHas() throws IOException {
        // given — ordinal 1, the closest, is empty
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, new int[] { 4, 0, 4, 4 });

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(4, 10));

        // then
        assertEquals(3, probes.length, "three occupied clusters, and a floor of four does not invent a fourth");
    }

    // ---------------------------------------------------------------- empty clusters

    /**
     * An empty cluster is dropped during the sweep, not left for the walk to skip — otherwise it would occupy a probe
     * slot and the ceiling would buy fewer scanned clusters than it says. Ordinal 1 is the closest of the four, so a
     * planner that kept it would push a real candidate out of a two-probe plan.
     */
    @Test
    void testPlan_whenAClusterIsEmpty_thenItTakesNoProbeSlot() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, new int[] { 4, 0, 4, 4 });

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 2));

        // then
        assertArrayEquals(new int[] { 2, 3 }, probes, "the next two closest, with the empty one passed over");
    }

    @Test
    void testPlan_whenEveryClusterIsEmpty_thenPlansNothing() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, CENTROIDS, new int[] { 0, 0, 0, 0 });

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 4));

        // then
        assertEquals(0, probes.length);
    }

    /** A field with no clusters is answered without reading a centroid — there is no region to read one from. */
    @Test
    void testPlan_whenTheFieldHasNoClusters_thenPlansNothingWithoutReading() throws IOException {
        // given
        Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(0);

        // when
        int[] probes = CentroidPlanner.plan(clusters, QUERY, new PlanParams(1, 4));

        // then
        assertEquals(0, probes.length);
    }

    // ---------------------------------------------------------------- helpers

    /**
     * Clusters over real centroid bytes. Only a EUCLIDEAN region carries ‖c‖², matching what the writer stores, so the
     * other metrics exercise the planner's fallback of measuring the norm from the vector.
     */
    private Clusters clusters(VectorSimilarityFunction similarity, float[][] centroids, int[] clusterSizes) throws IOException {
        CentroidVectorValues base = centroidValues(centroids, similarity == VectorSimilarityFunction.EUCLIDEAN);

        Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(centroids.length);
        when(clusters.clusterMeta()).thenReturn(fieldMeta(similarity, centroids.length, clusterSizes));
        // A fresh cursor per call, as the real one hands out — so a test may plan twice over the same centroids.
        when(clusters.centroids()).thenAnswer(invocation -> base.copy());
        for (int ordinal = 0; ordinal < clusterSizes.length; ordinal++) {
            when(clusters.clusterSize(ordinal)).thenReturn(clusterSizes[ordinal]);
        }
        return clusters;
    }

    /** Writes the centroid region the way the format lays it out: the vector, then ‖c‖² when the region carries one. */
    private CentroidVectorValues centroidValues(float[][] centroids, boolean withNorm) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (float[] centroid : centroids) {
                for (float value : centroid) {
                    out.writeInt(Float.floatToIntBits(value));
                }
                if (withNorm) {
                    out.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
                }
            }
        }
        IndexInput input = directory.openInput(FILE, IOContext.DEFAULT);
        return new CentroidVectorValues(input, centroids.length, centroids[0].length, withNorm);
    }

    /** The planner reads only the similarity from the entry; the rest is filled in to make a valid one. */
    private static ClusterANNFieldMeta fieldMeta(VectorSimilarityFunction similarity, int centroidCount, int[] clusterSizes) {
        int vectorCount = 0;
        for (int size : clusterSizes) {
            vectorCount += size;
        }
        return new ClusterANNFieldMeta(
            32,                                     // blockSize
            DIMENSION,
            vectorCount,
            centroidCount,
            similarity,
            1,                                      // docBits
            ClusterANNFieldMeta.ROTATION_NONE,
            0,                                      // quantizerId
            new byte[0],                            // quantizerParams
            0L,                                     // clacOffset
            0L,                                     // clacLength
            0L,                                     // clacCentroidsOffset
            ClusterANNFieldMeta.NO_ROTATION,        // clacRotatedCentroidsOffset
            0L,                                     // clapOffset
            0L,                                     // clapLength
            new long[centroidCount],                // clapCentroidOffsets
            new int[centroidCount],                 // centroidLengths
            clusterSizes,
            ClusterANNFieldMeta.NO_ROTATION,        // clarOffset
            ClusterANNFieldMeta.NO_ROTATION,        // clarLength
            null                                    // ordToDoc, never consulted by a plan
        );
    }
}
