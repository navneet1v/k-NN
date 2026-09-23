/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.ClusteringResult;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ClusterBuilderTests {

    /** Shared executor for exercising the parallel clustering/SOAR path. */
    private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);
    private static final TaskExecutor TASK_EXECUTOR = new TaskExecutor(EXECUTOR);

    @AfterAll
    static void shutdownExecutor() {
        EXECUTOR.shutdownNow();
    }

    @Test
    void testBuild_spillCapOne_matchesDefaultAndNoExtras() throws IOException {
        // Explicit cap of 1 must equal the default public build: single secondary, no extra-spill rows.
        FloatVectorValues source = createRandomVectors(4000, 16, 99L);

        ClusteringResult viaDefault = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);
        ClusteringResult viaCapOne = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null, 1, 0.0f);

        assertArrayEquals(viaDefault.soarAssignments(), viaCapOne.soarAssignments());
        assertArrayEquals(viaDefault.soarDistances(), viaCapOne.soarDistances(), 0f);
        // At cap 1 the extra-spill fields are absent entirely.
        assertNull(viaDefault.extraSoarAssignments());
        assertNull(viaDefault.extraSoarDistances());
        assertNull(viaCapOne.extraSoarAssignments());
    }

    @Test
    void testBuild_spillCapTwo_allowsSecondSpill() throws IOException {
        // Cap 2 with a generous margin: boundary vectors may carry one extra spill. Verify the
        // extra-spill rows are well-formed and every extra secondary is distinct from the first.
        FloatVectorValues source = createRandomVectors(4000, 16, 99L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, TASK_EXECUTOR, 2, 0.5f);

        assertNotNull(result.extraSoarAssignments());
        assertNotNull(result.extraSoarDistances());
        assertEquals(4000, result.extraSoarAssignments().length);

        int totalExtras = 0;
        for (int i = 0; i < 4000; i++) {
            int[] ex = result.extraSoarAssignments()[i];
            float[] exd = result.extraSoarDistances()[i];
            if (ex == null) {
                // Only vectors with no primary residual/secondary at all leave a null row.
                assertEquals(-1, result.soarAssignments()[i]);
                continue;
            }
            assertEquals(ex.length, exd.length);
            // Cap 2 => at most one spill beyond the first secondary.
            assertTrue(ex.length <= 1);
            for (int e = 0; e < ex.length; e++) {
                assertTrue(ex[e] >= 0 && ex[e] < result.numCentroids());
                assertTrue(ex[e] != result.soarAssignments()[i], "extra spill must differ from first secondary");
                assertFalse(Float.isNaN(exd[e]));
                assertTrue(exd[e] >= 0f);
                totalExtras++;
            }
        }
        assertTrue(totalExtras > 0, "expected cap 2 with a wide margin to produce some second spills");
    }

    @Test
    void testBuild_zeroMargin_collapsesInteriorVectorsToPrimaryOnly() throws IOException {
        // Cap 2 but a zero margin: a second spill is kept only on an exact score tie, which random
        // Gaussian data never produces — so no vector should carry an extra spill.
        FloatVectorValues source = createRandomVectors(4000, 16, 99L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null, 2, 0.0f);

        assertNotNull(result.extraSoarAssignments());
        for (int i = 0; i < 4000; i++) {
            int[] ex = result.extraSoarAssignments()[i];
            assertTrue(ex == null || ex.length == 0, "zero margin must not spill beyond the first secondary");
        }
    }

    @Test
    void testBuild_producesValidResult() throws IOException {
        FloatVectorValues source = createRandomVectors(1000, 32, 42L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);

        assertNotNull(result);
        assertTrue(result.numCentroids() > 0, "Should have > 0 centroids");
        assertEquals(1000, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.numCentroids());
        }
    }

    @Test
    void testBuild_largeInput_exercisesReservoirReplacement() throws IOException {
        // n > RESERVOIR_SIZE (4096) so the reservoir fills and then randomly replaces entries,
        // covering the Algorithm-R replacement branch and centroid seeding on large merges.
        FloatVectorValues source = createRandomVectors(5000, 16, 7L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, TASK_EXECUTOR);

        assertNotNull(result);
        assertTrue(result.numCentroids() > 0, "Should have > 0 centroids");
        assertEquals(5000, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.numCentroids());
        }
    }

    @Test
    void testBuild_smallInput_singleClusterSkipsReservoir() throws IOException {
        // n <= TARGET_CLUSTER_SIZE (512): HierarchicalKMeans collapses to a single centroid and
        // ClusterBuilder skips reservoir sampling entirely. Verify the degenerate result is valid:
        // one centroid, every vector assigned to it, no SOAR secondary.
        FloatVectorValues source = createRandomVectors(100, 16, 3L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);

        assertNotNull(result);
        assertEquals(1, result.numCentroids());
        assertEquals(100, result.assignments().length);
        assertEquals(100, result.distances().length);
        for (int i = 0; i < 100; i++) {
            assertEquals(0, result.assignments()[i]);
            assertTrue(result.distances()[i] >= 0f);
            // Single cluster => SOAR disabled (numCentroids == 1).
            assertEquals(-1, result.soarAssignments()[i]);
            assertTrue(Float.isNaN(result.soarDistances()[i]));
        }
    }

    @Test
    void testBuild_soarSecondariesWellFormed() throws IOException {
        // Enough vectors relative to the default target cluster size to produce multiple
        // centroids, so SOAR (always enabled via ClusterBuilderConstants) assigns secondaries.
        FloatVectorValues source = createRandomVectors(4000, 16, 99L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);

        assertNotNull(result);
        assertNotNull(result.soarAssignments());
        assertEquals(4000, result.soarAssignments().length);
        assertEquals(4000, result.soarDistances().length);

        int assignedSecondaries = 0;
        for (int i = 0; i < 4000; i++) {
            int a = result.soarAssignments()[i];
            if (a >= 0) {
                assertTrue(a < result.numCentroids());
                // A real secondary carries a real (finite, non-negative) member distance.
                assertFalse(Float.isNaN(result.soarDistances()[i]));
                assertTrue(result.soarDistances()[i] >= 0f);
                assignedSecondaries++;
            }
        }
        assertTrue(assignedSecondaries > 0, "expected SOAR to assign some secondaries");
    }

    @Test
    void testBuild_primaryDistancesMatchAssignedCentroid() throws IOException {
        FloatVectorValues source = createRandomVectors(3000, 8, 42L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);

        assertEquals(3000, result.distances().length);
        // Every primary distance is finite and non-negative (squared L2).
        for (int i = 0; i < 3000; i++) {
            float d = result.distances()[i];
            assertFalse(Float.isNaN(d));
            assertTrue(d >= 0f);
        }
    }

    @Test
    void testBuild_parallel_noCorruption() throws IOException {
        FloatVectorValues source = createRandomVectors(4000, 32, 42L);

        for (int trial = 0; trial < 3; trial++) {
            ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, TASK_EXECUTOR);
            assertNotNull(result);
            assertEquals(4000, result.assignments().length);
            for (int a : result.assignments()) {
                assertTrue(a >= 0 && a < result.numCentroids());
            }
        }
    }

    @Test
    void testBuild_assignmentsCoverAllVectors() throws IOException {
        FloatVectorValues source = createRandomVectors(3000, 8, 42L);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);

        // Every vector has a valid primary assignment (the writer builds postings from these).
        int[] counts = new int[result.numCentroids()];
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.numCentroids());
            counts[a]++;
        }
        int total = 0;
        for (int c : counts) {
            total += c;
        }
        assertEquals(3000, total);
    }

    @Test
    void testBuild_emptyInput() throws IOException {
        FloatVectorValues source = FloatVectorValues.fromFloats(new ArrayList<>(), 8);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.EUCLIDEAN, null);

        assertNotNull(result);
        assertEquals(0, result.numCentroids());
        assertEquals(0, result.assignments().length);
        assertEquals(0, result.distances().length);
        assertEquals(0, result.soarDistances().length);
    }

    @Test
    void testBuild_innerProduct() throws IOException {
        List<float[]> vecs = new ArrayList<>();
        Random rng = new Random(42L);
        for (int i = 0; i < 2000; i++) {
            float[] v = new float[16];
            for (int d = 0; d < 16; d++)
                v[d] = (float) rng.nextGaussian();
            // Normalize for IP
            float norm = 0;
            for (float f : v)
                norm += f * f;
            norm = (float) Math.sqrt(norm);
            for (int d = 0; d < 16; d++)
                v[d] /= norm;
            vecs.add(v);
        }
        FloatVectorValues source = FloatVectorValues.fromFloats(vecs, 16);

        ClusteringResult result = ClusterBuilder.build(source, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, TASK_EXECUTOR);

        assertNotNull(result);
        assertTrue(result.numCentroids() > 0);
    }

    private FloatVectorValues createRandomVectors(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vecs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++)
                v[d] = (float) rng.nextGaussian();
            vecs.add(v);
        }
        return FloatVectorValues.fromFloats(vecs, dim);
    }
}
