/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.TaskExecutor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HierarchicalKMeansTests {

    /** Shared executor for exercising the parallel assignment path. */
    private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);
    private static final TaskExecutor TASK_EXECUTOR = new TaskExecutor(EXECUTOR);

    @AfterAll
    static void shutdownExecutor() {
        EXECUTOR.shutdownNow();
    }

    @Test
    void dropEmptyClusters_noEmpties_returnsInputsUnchanged() {
        float[][] centroids = { { 0f, 0f }, { 1f, 1f } };
        int[] assignments = { 0, 1, 0, 1 };
        int[] counts = { 2, 2 };

        HierarchicalKMeans.Compacted out = HierarchicalKMeans.dropEmptyClusters(centroids, assignments, counts);

        // No empties -> same arrays handed back (no copy, no remap).
        assertSame(centroids, out.centroids());
        assertSame(assignments, out.assignments());
    }

    @Test
    void dropEmptyClusters_removesEmptyAndRemapsAssignments() {
        // Cluster 1 is empty; 0 and 2 survive and must be renumbered to 0 and 1.
        float[][] centroids = { { 0f, 0f }, { 9f, 9f }, { 2f, 2f } };
        int[] assignments = { 0, 2, 2, 0 };
        int[] counts = { 2, 0, 2 };

        HierarchicalKMeans.Compacted out = HierarchicalKMeans.dropEmptyClusters(centroids, assignments, counts);

        assertEquals(2, out.centroids().length);
        assertArrayEquals(new float[] { 0f, 0f }, out.centroids()[0], 0f);
        assertArrayEquals(new float[] { 2f, 2f }, out.centroids()[1], 0f);
        // old 0 -> new 0, old 2 -> new 1
        assertArrayEquals(new int[] { 0, 1, 1, 0 }, out.assignments());
    }

    @Test
    void testCluster_producesValidResult() throws IOException {
        FloatVectorValues source = createRandomVectors(1000, 16, 42L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(64).seed(42L).maxIterations(10).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertNotNull(result);
        assertTrue(result.numCentroids() > 0, "Should have > 0 centroids: " + result.numCentroids());
        assertEquals(1000, result.assignments().length);
    }

    @Test
    void testCluster_allAssignmentsValid() throws IOException {
        FloatVectorValues source = createRandomVectors(500, 8, 77L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(32).seed(42L).maxIterations(10).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        for (int i = 0; i < result.assignments().length; i++) {
            int a = result.assignments()[i];
            assertTrue(
                a >= 0 && a < result.numCentroids(),
                "Assignment " + a + " at index " + i + " should be in [0, " + result.numCentroids() + ")"
            );
        }
    }

    @Test
    void testCluster_parallel_noCorruption() throws IOException {
        FloatVectorValues source = createRandomVectors(2000, 32, 42L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(32)
            .seed(42L)
            .executor(TASK_EXECUTOR)
            .maxIterations(10)
            .build();

        // Run multiple times to catch intermittent threading issues
        for (int trial = 0; trial < 3; trial++) {
            HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);
            assertNotNull(result);
            assertEquals(2000, result.assignments().length);
            for (int a : result.assignments()) {
                assertTrue(a >= 0 && a < result.numCentroids());
            }
        }
    }

    @Test
    void testCluster_smallInput() throws IOException {
        FloatVectorValues source = createRandomVectors(10, 4, 42L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(100).seed(42L).maxIterations(5).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertNotNull(result);
        assertTrue(result.numCentroids() <= 10);
        assertEquals(10, result.assignments().length);
    }

    @Test
    void testCluster_deepRecursiveSplit() throws IOException {
        // Large input with a small targetSize forces multi-level recursive splitting
        // (top-level k-means then recursive splitRecursive on oversized clusters).
        FloatVectorValues source = createRandomVectors(20000, 16, 2024L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(64)
            .seed(42L)
            .executor(TASK_EXECUTOR)
            .maxIterations(8)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertNotNull(result);
        assertEquals(20000, result.assignments().length);
        // Many clusters expected: 20000 / 64 is well above one flat level.
        assertTrue(result.numCentroids() > 16, "deep split should produce many centroids");
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.numCentroids());
        }
    }

    @Test
    void testCluster_coarseToFine_matchesNearestLeaf() throws IOException {
        // Well-separated blobs: each vector's globally nearest leaf lies under its own top-level
        // centroid, so the coarse-to-fine probe (nearest few top-level centroids) always contains
        // the true parent and the assignment must equal an exact brute-force nearest-leaf scan.
        int dim = 8;
        int blobs = 12;
        int perBlob = 60; // > targetSize below to force splitting into multiple leaves
        Random rng = new Random(123L);
        List<float[]> vectors = new ArrayList<>();
        for (int b = 0; b < blobs; b++) {
            float center = b * 100f; // large gaps => unambiguous nearest blob
            for (int i = 0; i < perBlob; i++) {
                float[] v = new float[dim];
                for (int d = 0; d < dim; d++) {
                    v[d] = center + (float) rng.nextGaussian();
                }
                vectors.add(v);
            }
        }
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, dim);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(20).seed(42L).maxIterations(10).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);
        float[][] centroids = result.centroids();
        int[] assignments = result.assignments();

        for (int i = 0; i < vectors.size(); i++) {
            float[] vec = vectors.get(i);
            float assignedDist = squareDist(vec, centroids[assignments[i]]);
            float bestDist = Float.MAX_VALUE;
            for (float[] centroid : centroids) {
                bestDist = Math.min(bestDist, squareDist(vec, centroid));
            }
            assertEquals(bestDist, assignedDist, 1e-4f, "coarse-to-fine assignment for vector " + i + " must match the exact nearest leaf");
        }
    }

    private static float squareDist(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    @Test
    void testCluster_withInitialCentroids_mergePath() throws IOException {
        // Pre-seeded centroids exercise the merge-path branch that skips k-means++ init.
        FloatVectorValues source = createRandomVectors(1000, 8, 55L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(64).seed(42L).maxIterations(8).build();

        int dim = 8;
        int k = 16;
        float[][] initialCentroids = new float[k][dim];
        Random rng = new Random(99L);
        for (int c = 0; c < k; c++) {
            for (int d = 0; d < dim; d++)
                initialCentroids[c][d] = (float) rng.nextGaussian();
        }

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config, initialCentroids);

        assertNotNull(result);
        assertEquals(1000, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.numCentroids());
        }
    }

    @Test
    void testCluster_singleCentroid_tinyDataset() throws IOException {
        // n <= targetSize returns a single centroid (mean) without splitting.
        FloatVectorValues source = createRandomVectors(20, 4, 3L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(64).seed(42L).maxIterations(5).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertNotNull(result);
        assertEquals(1, result.numCentroids());
        assertEquals(20, result.assignments().length);
        for (int a : result.assignments()) {
            assertEquals(0, a);
        }
    }

    @Test
    void testCluster_emptyInput() throws IOException {
        FloatVectorValues source = FloatVectorValues.fromFloats(new ArrayList<>(), 8);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(64).seed(42L).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertNotNull(result);
        assertEquals(0, result.numCentroids());
        assertEquals(0, result.assignments().length);
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
