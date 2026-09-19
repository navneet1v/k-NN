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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KMeansTests {

    /** Shared executor for exercising the parallel assignment path. */
    private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);
    private static final TaskExecutor TASK_EXECUTOR = new TaskExecutor(EXECUTOR);

    @AfterAll
    static void shutdownExecutor() {
        EXECUTOR.shutdownNow();
    }

    @Test
    void cluster_emptyInput_returnsEmptyResult() throws java.io.IOException {
        FloatVectorValues source = FloatVectorValues.fromFloats(new java.util.ArrayList<>(), 8);
        KMeans.Config config = KMeans.Config.builder().seed(42L).build();

        KMeans.Result result = KMeans.cluster(source, 4, config);

        assertEquals(0, result.k());
        assertEquals(0, result.assignments().length);
        assertEquals(0, result.centroids().length);
    }

    // ========== Basic clustering ==========

    @Test
    void testCluster_twoWellSeparatedClusters() throws IOException {
        List<float[]> vectors = new ArrayList<>();
        // Cluster A: around (0, 0)
        for (int i = 0; i < 100; i++) {
            vectors.add(new float[] { (float) (Math.random() * 0.5 - 0.25), (float) (Math.random() * 0.5 - 0.25) });
        }
        // Cluster B: around (10, 10)
        for (int i = 0; i < 100; i++) {
            vectors.add(new float[] { (float) (10 + Math.random() * 0.5 - 0.25), (float) (10 + Math.random() * 0.5 - 0.25) });
        }

        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 2);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(20).build();
        KMeans.Result result = KMeans.cluster(source, 2, config);

        assertEquals(2, result.k());
        assertEquals(200, result.assignments().length);

        // All vectors 0-99 should be in same cluster, 100-199 in another
        int clusterA = result.assignments()[0];
        int clusterB = result.assignments()[100];
        assertFalse(clusterA == clusterB, "Two clusters should have different assignments");

        for (int i = 0; i < 100; i++) {
            assertEquals(clusterA, result.assignments()[i], "Vector " + i + " should be in cluster A");
        }
        for (int i = 100; i < 200; i++) {
            assertEquals(clusterB, result.assignments()[i], "Vector " + i + " should be in cluster B");
        }
    }

    @Test
    void testCluster_singleCluster_allAssignedToSame() throws IOException {
        List<float[]> vectors = createRandomVectors(50, 8, 42L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 8);
        KMeans.Config config = KMeans.Config.builder().seed(1L).build();

        KMeans.Result result = KMeans.cluster(source, 1, config);

        assertEquals(1, result.k());
        for (int a : result.assignments()) {
            assertEquals(0, a);
        }
    }

    @Test
    void testCluster_kEqualsN_eachVectorOwnCluster() throws IOException {
        int n = 10;
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float[] v = new float[4];
            v[0] = i * 100; // widely separated
            vectors.add(v);
        }
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 4);
        KMeans.Config config = KMeans.Config.builder().seed(7L).maxIterations(50).build();

        KMeans.Result result = KMeans.cluster(source, n, config);

        assertEquals(n, result.k());
        // Each vector in its own cluster (no two share)
        int[] counts = new int[n];
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < n);
            counts[a]++;
        }
        for (int c : counts) {
            assertEquals(1, c, "Each cluster should have exactly 1 member");
        }
    }

    // ========== Centroid quality ==========

    @Test
    void testCluster_centroidsNearClusterMeans() throws IOException {
        // 3 Gaussian blobs in 2D
        Random rng = new Random(42);
        float[][] centers = { { 0, 0 }, { 10, 0 }, { 5, 10 } };
        List<float[]> vectors = new ArrayList<>();
        for (float[] center : centers) {
            for (int i = 0; i < 100; i++) {
                vectors.add(new float[] { center[0] + (float) rng.nextGaussian() * 0.5f, center[1] + (float) rng.nextGaussian() * 0.5f });
            }
        }

        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 2);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(30).build();
        KMeans.Result result = KMeans.cluster(source, 3, config);

        // Each centroid should be within 1.0 of a true center
        for (float[] centroid : result.centroids()) {
            float minDist = Float.MAX_VALUE;
            for (float[] center : centers) {
                float d = (centroid[0] - center[0]) * (centroid[0] - center[0]) + (centroid[1] - center[1]) * (centroid[1] - center[1]);
                minDist = Math.min(minDist, d);
            }
            assertTrue(minDist < 1.0f, "Centroid should be near a true center, dist=" + minDist);
        }
    }

    // ========== Parallel vs Sequential consistency ==========

    @Test
    void testCluster_parallelAndSequential_sameResult() throws IOException {
        List<float[]> vectors = createRandomVectors(500, 16, 55L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 16);

        KMeans.Config seqConfig = KMeans.Config.builder().seed(42L).maxIterations(10).build();
        KMeans.Config parConfig = KMeans.Config.builder().seed(42L).executor(TASK_EXECUTOR).maxIterations(10).build();

        KMeans.Result seqResult = KMeans.cluster(source, 8, seqConfig);
        KMeans.Result parResult = KMeans.cluster(source, 8, parConfig);

        assertEquals(seqResult.k(), parResult.k());
        // Parallel sums per slice then reduces slices in a fixed order, which is a different
        // (but deterministic) addition tree than the flat sequential sum — so results match
        // closely, not bit-for-bit. Exact run-to-run stability of the parallel path is asserted
        // separately in testCluster_parallel_isReproducibleAcrossRuns.
        for (int c = 0; c < seqResult.k(); c++) {
            assertArrayEquals(
                seqResult.centroids()[c],
                parResult.centroids()[c],
                1e-4f,
                "Parallel centroids should match sequential closely, c=" + c
            );
        }
    }

    @Test
    void testCluster_parallel_isReproducibleAcrossRuns() throws IOException {
        List<float[]> vectors = createRandomVectors(500, 16, 55L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 16);
        KMeans.Config parConfig = KMeans.Config.builder().seed(42L).executor(TASK_EXECUTOR).maxIterations(10).build();

        // Same seed + input + executor must give identical results on every run: the fixed-order
        // partial reduction removes the thread-arrival-order dependence in float accumulation.
        KMeans.Result first = KMeans.cluster(source, 8, parConfig);
        for (int run = 0; run < 5; run++) {
            KMeans.Result again = KMeans.cluster(source, 8, parConfig);
            assertArrayEquals(first.assignments(), again.assignments(), "Parallel assignments must be identical across runs, run=" + run);
            for (int c = 0; c < first.k(); c++) {
                assertArrayEquals(
                    first.centroids()[c],
                    again.centroids()[c],
                    0f,
                    "Parallel centroids must be identical across runs, run=" + run + " c=" + c
                );
            }
        }
    }

    // ========== initialCentroids validation ==========

    @Test
    void testCluster_initialCentroids_lengthMismatchRejected() throws IOException {
        // n = 5 vectors, so k clamps to at most 5. Passing 8 initial centroids must be rejected
        // up front rather than throwing ArrayIndexOutOfBoundsException during assignment, because
        // clusterSums/clusterCounts are sized with the clamped k while assignmentStep derives its
        // cluster count from centroids.length.
        int n = 5;
        int dim = 4;
        FloatVectorValues source = FloatVectorValues.fromFloats(createRandomVectors(n, dim, 7L), dim);
        float[][] tooMany = createRandomVectors(8, dim, 8L).toArray(new float[0][]);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(5).build();

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> KMeans.cluster(source, 8, config, tooMany));
        assertTrue(ex.getMessage().contains("initialCentroids.length"), "message should explain the length mismatch: " + ex.getMessage());
    }

    @Test
    void testCluster_initialCentroids_notMutated() throws IOException {
        // The k-means iteration mutates its centroid rows in place. The caller-supplied
        // initialCentroids array (and its inner float[] rows) must be defensively deep-copied so a
        // caller — e.g. HierarchicalKMeans reusing a reservoir sample — can safely reuse it.
        int n = 300;
        int dim = 8;
        int k = 4;
        FloatVectorValues source = FloatVectorValues.fromFloats(createRandomVectors(n, dim, 11L), dim);
        float[][] seeds = createRandomVectors(k, dim, 12L).toArray(new float[0][]);

        // Snapshot the seeds (deep) before clustering.
        float[][] seedsBefore = new float[k][];
        for (int c = 0; c < k; c++) {
            seedsBefore[c] = seeds[c].clone();
        }

        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(10).build();
        KMeans.Result result = KMeans.cluster(source, k, config, seeds);

        // The caller's array is untouched...
        for (int c = 0; c < k; c++) {
            assertArrayEquals(seedsBefore[c], seeds[c], 0f, "initialCentroids row " + c + " must not be mutated by clustering");
        }
        // ...and the result's centroids are a distinct array (not aliased to the input rows).
        float[][] out = result.centroids();
        for (int c = 0; c < out.length; c++) {
            for (int s = 0; s < k; s++) {
                assertFalse(out[c] == seeds[s], "result centroid must not alias an input seed row");
            }
        }
    }

    // ========== Rebalance / counts consistency ==========

    @Test
    void testCluster_rebalance_countsConsistentWithAssignments() throws IOException {
        // Few distinct clusters but a larger k forces empty clusters, which triggers
        // rebalanceEmptyClusters. Rebalancing mutates centroids without re-assigning; if the loop
        // ends on a rebalance (converged or at the iteration cap) the returned counts/assignments
        // must still be reconciled with the returned centroids. Assert that the reported counts
        // exactly match a recomputation from assignments, and every assignment is in range.
        int dim = 4;
        List<float[]> vectors = new ArrayList<>();
        // 3 tight blobs, duplicated, so genuine clusters << k
        for (int blob = 0; blob < 3; blob++) {
            float base = blob * 10f;
            for (int i = 0; i < 20; i++) {
                float[] v = new float[dim];
                Arrays.fill(v, base);
                vectors.add(v);
            }
        }
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, dim);
        int k = 12;
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(10).build();

        KMeans.Result result = KMeans.cluster(source, k, config);

        int rk = result.k();
        int[] recomputed = new int[rk];
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < rk, "assignment index out of range: " + a);
            recomputed[a]++;
        }
        assertArrayEquals(recomputed, result.counts(), "reported counts must match counts recomputed from assignments after any rebalance");
    }

    // ========== Convergence ==========

    @Test
    void testCluster_converges() throws IOException {
        List<float[]> vectors = createRandomVectors(200, 8, 99L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 8);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(100).build();

        KMeans.Result result = KMeans.cluster(source, 5, config);

        assertTrue(result.converged() || result.iterations() < 100, "Should converge in < 100 iterations");
    }

    // ========== Dimension handling ==========

    @Test
    void testCluster_highDimensional_768d() throws IOException {
        List<float[]> vectors = createRandomVectors(200, 768, 42L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 768);
        KMeans.Config config = KMeans.Config.builder().seed(1L).maxIterations(5).build();

        KMeans.Result result = KMeans.cluster(source, 4, config);

        assertEquals(4, result.k());
        assertEquals(768, result.dimension());
        assertEquals(200, result.assignments().length);
        // All assignments valid
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 4);
        }
    }

    // ========== Edge cases ==========

    @Test
    void testCluster_kLargerThanN_clampedToN() throws IOException {
        List<float[]> vectors = createRandomVectors(5, 4, 1L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 4);
        KMeans.Config config = KMeans.Config.builder().seed(1L).build();

        KMeans.Result result = KMeans.cluster(source, 100, config);

        assertTrue(result.k() <= 5, "k should be clamped to n");
    }

    @Test
    void testCluster_allIdenticalVectors() throws IOException {
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            vectors.add(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        }
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 4);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(10).build();

        // Should not crash even with degenerate data
        KMeans.Result result = KMeans.cluster(source, 3, config);
        assertNotNull(result);
        assertEquals(50, result.assignments().length);
    }

    // ========== Distance metrics ==========

    @Test
    void testCluster_innerProduct() throws IOException {
        List<float[]> vectors = createRandomVectors(100, 8, 42L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 8);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(10).metric(VectorSimilarityFunction.DOT_PRODUCT).build();

        KMeans.Result result = KMeans.cluster(source, 4, config);
        assertEquals(4, result.k());
        // Should not crash and produce valid assignments
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 4);
        }
    }

    // ========== Initial centroids ==========

    @Test
    void testCluster_withInitialCentroids() throws IOException {
        List<float[]> vectors = new ArrayList<>();
        // Two clear clusters
        for (int i = 0; i < 50; i++)
            vectors.add(new float[] { 0.0f, 0.0f });
        for (int i = 0; i < 50; i++)
            vectors.add(new float[] { 10.0f, 10.0f });

        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 2);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(5).build();

        // Provide initial centroids close to true centers
        float[][] initCentroids = { { 0.1f, 0.1f }, { 9.9f, 9.9f } };
        KMeans.Result result = KMeans.cluster(source, 2, config, initCentroids);

        assertEquals(2, result.k());
        // Should converge immediately with good initial centroids
        assertTrue(result.iterations() <= 3, "Should converge quickly with good init");
    }

    // ========== Coverage: parallel, cosine, sampling, proximity map, rebalance ==========

    @Test
    void testCluster_parallel_manyClusters_exercisesProximityMap() throws IOException {
        // k > PROXIMITY_MAP_SIZE * 2 (=16) triggers the centroid-proximity-map path after iter 2,
        // run in parallel to also cover the multi-worker assignment branch.
        List<float[]> vectors = createRandomVectors(2000, 16, 123L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 16);
        KMeans.Config config = KMeans.Config.builder().seed(42L).executor(TASK_EXECUTOR).maxIterations(15).build();

        KMeans.Result result = KMeans.cluster(source, 32, config);

        assertEquals(32, result.k());
        assertEquals(2000, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 32);
        }
    }

    @Test
    void testCluster_cosineMetric_normalizesCentroids() throws IOException {
        // COSINE metric drives the spherical centroid normalization branch in updateCentroids.
        List<float[]> vectors = createRandomVectors(300, 8, 7L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 8);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(15).metric(VectorSimilarityFunction.COSINE).build();

        KMeans.Result result = KMeans.cluster(source, 5, config);

        assertEquals(5, result.k());
        // Non-empty centroids should be unit-norm under cosine (spherical k-means).
        int[] counts = new int[5];
        for (int a : result.assignments())
            counts[a]++;
        for (int c = 0; c < 5; c++) {
            if (counts[c] == 0) continue;
            float norm = 0f;
            for (float x : result.centroids()[c])
                norm += x * x;
            assertEquals(1.0f, norm, 1e-3f, "cosine centroid should be unit-norm");
        }
    }

    @Test
    void testCluster_samplingPath_largeInput() throws IOException {
        // n > 10_000 with default samplePercentage triggers the sampled-iterations path,
        // then the final full-data assignment pass.
        List<float[]> vectors = createRandomVectors(12000, 8, 321L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 8);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(5).samplePercentage(0.05f).build();

        KMeans.Result result = KMeans.cluster(source, 10, config);

        assertEquals(10, result.k());
        assertEquals(12000, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 10);
        }
    }

    @Test
    void testCluster_rebalanceEmptyClusters() throws IOException {
        // Fewer distinct points than k forces empty clusters, exercising the rebalance path.
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < 40; i++)
            vectors.add(new float[] { 0.0f, 0.0f });
        for (int i = 0; i < 40; i++)
            vectors.add(new float[] { 5.0f, 5.0f });
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 2);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(10).rebalanceEmpty(true).build();

        // Ask for more clusters than there are distinct points.
        KMeans.Result result = KMeans.cluster(source, 6, config);

        assertEquals(6, result.k());
        assertEquals(80, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 6);
        }
    }

    @Test
    void testConfig_defaults() {
        KMeans.Config config = KMeans.Config.defaults();
        // Exercises the defaults() factory and the @Builder defaults.
        assertNotNull(config);
    }

    @Test
    void testCluster_sequential_manyClusters_exercisesProximityMap() throws IOException {
        // Same proximity-map trigger (k > 16) but sequential, covering the non-parallel
        // proximity-map assignment branch and its nearest-neighbor update.
        List<float[]> vectors = createRandomVectors(1500, 16, 456L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 16);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(15).build();

        KMeans.Result result = KMeans.cluster(source, 24, config);

        assertEquals(24, result.k());
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 24);
        }
    }

    @Test
    void testCluster_innerProduct_parallel() throws IOException {
        // Inner-product metric under the parallel assignment path.
        List<float[]> vectors = createRandomVectors(1200, 16, 88L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 16);
        KMeans.Config config = KMeans.Config.builder()
            .seed(42L)
            .executor(TASK_EXECUTOR)
            .maxIterations(10)
            .metric(VectorSimilarityFunction.DOT_PRODUCT)
            .build();

        KMeans.Result result = KMeans.cluster(source, 8, config);

        assertEquals(8, result.k());
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 8);
        }
    }

    @Test
    void testCluster_rebalanceDisabled() throws IOException {
        // rebalanceEmpty(false) covers the branch that skips empty-cluster rebalancing.
        List<float[]> vectors = createRandomVectors(300, 8, 202L);
        FloatVectorValues source = FloatVectorValues.fromFloats(vectors, 8);
        KMeans.Config config = KMeans.Config.builder().seed(42L).maxIterations(10).rebalanceEmpty(false).build();

        KMeans.Result result = KMeans.cluster(source, 6, config);

        assertEquals(6, result.k());
        assertEquals(300, result.assignments().length);
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < 6);
        }
    }

    // ========== Helpers ==========

    private static List<float[]> createRandomVectors(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vectors = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++) {
                v[d] = (float) rng.nextGaussian();
            }
            vectors.add(v);
        }
        return vectors;
    }
}
