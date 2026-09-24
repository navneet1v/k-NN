/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.VectorUtil;
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

    /**
     * Every emitted centroid must own at least one vector. The posting writer depends on it -
     * OptimizedScalarQuantizedClusterWriter rejects a zero-member cluster outright - and the split branch of
     * {@link HierarchicalKMeans#cluster} checks emptiness against the top-level assignment and then replaces that
     * assignment with a coarse-to-fine one, so nothing verifies the result it actually returns.
     *
     * <p>Under an inner-product metric this breaks badly: centroids are means, so their norms differ with the population
     * that produced them, and inner product rewards a large norm regardless of direction. A handful of high-norm
     * centroids win nearly every vector and the rest are left with none. Euclidean assignment is consistent with means
     * and does not show it, which is why only a real inner-product corpus surfaced this.
     */
    @Test
    void testCluster_innerProduct_everyCentroidHasMembers() throws IOException {
        FloatVectorValues source = skewedPopulations(5000, 128, 20, 5L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(512)
            .metric(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT)
            .seed(42L)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config, null);

        int[] members = new int[result.numCentroids()];
        for (int assignment : result.assignments()) {
            members[assignment]++;
        }
        List<Integer> empty = new ArrayList<>();
        for (int centroid = 0; centroid < members.length; centroid++) {
            if (members[centroid] == 0) {
                empty.add(centroid);
            }
        }
        assertTrue(
            empty.isEmpty(),
            () -> empty.size()
                + " of "
                + result.numCentroids()
                + " centroids own no vectors "
                + empty
                + "; the posting writer rejects a zero-member cluster, so this cannot be written"
        );
    }

    /**
     * Groups of wildly uneven population: 80% of the vectors in one, the rest spread over the others. The imbalance is
     * what makes centroid norms differ, and one oversized group is what forces the split branch to run.
     */
    private static FloatVectorValues skewedPopulations(int size, int dim, int groups, long seed) {
        Random rng = new Random(seed);
        float[][] centres = new float[groups][dim];
        for (int group = 0; group < groups; group++) {
            for (int d = 0; d < dim; d++) {
                centres[group][d] = rng.nextFloat() * 2f - 1f;
            }
        }
        List<float[]> vectors = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            int group = rng.nextInt(10) < 8 ? 0 : 1 + rng.nextInt(groups - 1);
            float[] vector = new float[dim];
            for (int d = 0; d < dim; d++) {
                vector[d] = centres[group][d] + (float) rng.nextGaussian() * 0.05f;
            }
            vectors.add(vector);
        }
        return FloatVectorValues.fromFloats(vectors, dim);
    }

    // ========== cosine centroids ==========
    // The single-centroid path (n <= targetSize) and the splitRecursive leaves build centroids without running
    // KMeans. They used to return raw means under cosine while KMeans returned unit vectors, and the quantizer
    // asserts a unit centroid under cosine; both now go through VectorMath.centroidFromSum.

    @Test
    void testCluster_cosine_singleCentroidPath_isUnitLength() throws IOException {
        FloatVectorValues source = unitVectors(200, 16, 7L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(512)
            .metric(VectorSimilarityFunction.COSINE)
            .seed(7L)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertEquals(1, result.numCentroids());
        assertAllUnitLength(result.centroids());
    }

    @Test
    void testCluster_cosine_splitPaths_areUnitLength() throws IOException {
        // Well above targetSize: top-level KMeans plus recursive splits whose leaves come from centroidOf.
        FloatVectorValues source = unitVectors(4000, 16, 11L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(64)
            .metric(VectorSimilarityFunction.COSINE)
            .seed(11L)
            .maxIterations(5)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(source, config);

        assertTrue(result.numCentroids() > 1, "expected the split path, got " + result.numCentroids() + " centroid");
        assertAllUnitLength(result.centroids());
    }

    @Test
    void testCluster_euclidean_singleCentroidPath_isThePlainMean() throws IOException {
        List<float[]> vectors = List.of(new float[] { 0f, 0f }, new float[] { 2f, 0f }, new float[] { 1f, 3f });
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(512)
            .metric(VectorSimilarityFunction.EUCLIDEAN)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(FloatVectorValues.fromFloats(vectors, 2), config);

        assertEquals(1, result.numCentroids());
        assertArrayEquals(new float[] { 1f, 1f }, result.centroids()[0], 1e-6f);
    }

    private static void assertAllUnitLength(float[][] centroids) {
        for (int c = 0; c < centroids.length; c++) {
            float norm = (float) Math.sqrt(VectorUtil.dotProduct(centroids[c], centroids[c]));
            assertTrue(VectorUtil.isUnitVector(centroids[c]), "centroid " + c + " has norm " + norm);
        }
    }

    /** Random directions, all unit length, spread widely so their means are far from unit length. */
    private static FloatVectorValues unitVectors(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vectors = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] vector = new float[dim];
            for (int d = 0; d < dim; d++) {
                vector[d] = (float) rng.nextGaussian();
            }
            VectorUtil.l2normalize(vector);
            vectors.add(vector);
        }
        return FloatVectorValues.fromFloats(vectors, dim);
    }
}
