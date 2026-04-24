/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.opensearch.knn.index.clusterann.algorithm.*;
import org.opensearch.knn.KNNTestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Tests for {@link KMeans} covering initialization, convergence, rebalancing, and metrics.
 */
public class KMeansClustererTests extends KNNTestCase {

    private static final int DIM = 8;
    private static final long SEED = 123L;

    // ========== Basic Clustering ==========

    public void testClustersTwoBlobs() throws Exception {
        // Two well-separated clusters at (0,0,...) and (10,10,...)
        int n = 200;
        float[] data = new float[n * DIM];
        Random rng = new Random(SEED);
        for (int i = 0; i < n / 2; i++) {
            for (int d = 0; d < DIM; d++)
                data[i * DIM + d] = rng.nextFloat();
        }
        for (int i = n / 2; i < n; i++) {
            for (int d = 0; d < DIM; d++)
                data[i * DIM + d] = 10f + rng.nextFloat();
        }

        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(toVectorList(data, n, DIM), DIM);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).build();
        KMeans.Result result = KMeans.cluster(vectors, 2, config);

        assertEquals(2, result.k());
        assertEquals(n, result.assignments().length);

        // All vectors in first half should be in same cluster
        int cluster0 = result.assignments()[0];
        for (int i = 1; i < n / 2; i++) {
            assertEquals("Vector " + i + " should be in cluster " + cluster0, cluster0, result.assignments()[i]);
        }
        // All vectors in second half should be in the other cluster
        int cluster1 = result.assignments()[n / 2];
        assertNotEquals(cluster0, cluster1);
        for (int i = n / 2 + 1; i < n; i++) {
            assertEquals("Vector " + i + " should be in cluster " + cluster1, cluster1, result.assignments()[i]);
        }
    }

    public void testConvergence() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(4, 50, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).maxIterations(100).build();
        KMeans.Result result = KMeans.cluster(vectors, 4, config);

        assertTrue("Should converge in < 100 iterations", result.converged());
        assertTrue("Should take > 1 iteration", result.iterations() > 1);
    }

    public void testSingleVector() throws Exception {
        float[] data = new float[DIM];
        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(List.of(data), DIM);
        KMeans.Result result = KMeans.cluster(vectors, 5, KMeans.Config.defaults());

        assertEquals(1, result.k()); // clamped to n
        assertEquals(0, result.assignments()[0]);
    }

    public void testKGreaterThanN() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(10, DIM, SEED);
        KMeans.Result result = KMeans.cluster(vectors, 100, KMeans.Config.builder().parallel(false).build());

        assertEquals(10, result.k()); // clamped
    }

    // ========== Rebalancing ==========

    public void testEmptyClusterRebalancing() throws Exception {
        // Create data where k-means might produce empty clusters
        // All vectors near origin except one outlier
        int n = 50;
        float[] data = new float[n * DIM];
        Random rng = new Random(SEED);
        for (int i = 0; i < n - 1; i++) {
            for (int d = 0; d < DIM; d++)
                data[i * DIM + d] = rng.nextFloat() * 0.1f;
        }
        // One outlier far away
        for (int d = 0; d < DIM; d++)
            data[(n - 1) * DIM + d] = 100f;

        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(toVectorList(data, n, DIM), DIM);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).rebalanceEmpty(true).build();
        KMeans.Result result = KMeans.cluster(vectors, 3, config);

        // After rebalancing, no cluster should be empty (all 3 should have assignments)
        int[] counts = new int[result.k()];
        for (int a : result.assignments())
            counts[a]++;
        for (int c = 0; c < result.k(); c++) {
            assertTrue("Cluster " + c + " should not be empty after rebalancing", counts[c] > 0);
        }
    }

    public void testNoRebalancing() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(50, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).rebalanceEmpty(false).build();
        KMeans.Result result = KMeans.cluster(vectors, 5, config);

        // Should still produce valid assignments
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.k());
        }
    }

    // ========== Distance Metrics ==========

    public void testL2Metric() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(3, 100, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().metric(DistanceMetric.L2).seed(SEED).parallel(false).build();
        KMeans.Result result = KMeans.cluster(vectors, 3, config);

        assertEquals(3, result.k());
        assertTrue(result.converged());
    }

    public void testInnerProductMetric() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(3, 100, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().metric(DistanceMetric.INNER_PRODUCT).seed(SEED).parallel(false).build();
        KMeans.Result result = KMeans.cluster(vectors, 3, config);

        assertEquals(3, result.k());
        // IP metric may produce different assignments than L2
        assertNotNull(result.assignments());
    }

    public void testCosineMetric() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(3, 100, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().metric(DistanceMetric.COSINE).seed(SEED).parallel(false).build();
        KMeans.Result result = KMeans.cluster(vectors, 3, config);

        assertEquals(3, result.k());
    }

    // ========== Determinism ==========

    public void testDeterministicWithSameSeed() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(200, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().seed(42L).parallel(false).build();

        KMeans.Result r1 = KMeans.cluster(vectors, 5, config);
        KMeans.Result r2 = KMeans.cluster(vectors, 5, config);

        assertArrayEquals(r1.assignments(), r2.assignments());
        float[][] c1Centroids = r1.centroids();
        float[][] c2Centroids = r2.centroids();
        assertEquals(c1Centroids.length, c2Centroids.length);
        for (int i = 0; i < c1Centroids.length; i++) {
            assertFloatArrayEquals(c1Centroids[i], c2Centroids[i], 1e-6f);
        }
    }

    public void testDifferentSeedsDifferentResults() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(200, DIM, SEED);
        KMeans.Config c1 = KMeans.Config.builder().seed(1L).parallel(false).build();
        KMeans.Config c2 = KMeans.Config.builder().seed(999L).parallel(false).build();

        KMeans.Result r1 = KMeans.cluster(vectors, 5, c1);
        KMeans.Result r2 = KMeans.cluster(vectors, 5, c2);

        // Very unlikely to be identical with different seeds
        boolean allSame = true;
        for (int i = 0; i < r1.assignments().length; i++) {
            if (r1.assignments()[i] != r2.assignments()[i]) {
                allSame = false;
                break;
            }
        }
        assertFalse("Different seeds should produce different results", allSame);
    }

    // ========== Parallel ==========

    public void testParallelProducesSameAssignmentQuality() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(4, 200, DIM, SEED);
        KMeans.Config seqConfig = KMeans.Config.builder().seed(SEED).parallel(false).build();
        KMeans.Config parConfig = KMeans.Config.builder().seed(SEED).parallel(true).build();

        KMeans.Result seqResult = KMeans.cluster(vectors, 4, seqConfig);
        KMeans.Result parResult = KMeans.cluster(vectors, 4, parConfig);

        // Both should produce 4 clusters with valid assignments
        assertEquals(seqResult.k(), parResult.k());
        for (int a : parResult.assignments()) {
            assertTrue(a >= 0 && a < 4);
        }
    }

    // ========== Config ==========

    public void testMaxIterationsRespected() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(500, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).maxIterations(2).build();
        KMeans.Result result = KMeans.cluster(vectors, 10, config);

        assertTrue(result.iterations() <= 2);
    }

    public void testConvergenceThreshold() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(4, 200, DIM, SEED);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).convergenceThreshold(0.05f).maxIterations(100).build();
        KMeans.Result result = KMeans.cluster(vectors, 4, config);

        // Should converge early due to threshold
        assertTrue(result.iterations() < 100);
    }

    // ========== Centroid Quality ==========

    public void testCentroidsNearClusterCenters() throws Exception {
        // 4 blobs centered at known positions
        int perCluster = 100;
        float[][] centers = {
            { 0, 0, 0, 0, 0, 0, 0, 0 },
            { 10, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 10, 0, 0, 0, 0, 0, 0 },
            { 10, 10, 0, 0, 0, 0, 0, 0 } };
        int n = 4 * perCluster;
        float[] data = new float[n * DIM];
        Random rng = new Random(SEED);
        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < perCluster; i++) {
                int idx = c * perCluster + i;
                for (int d = 0; d < DIM; d++) {
                    data[idx * DIM + d] = centers[c][d] + rng.nextFloat() * 0.5f;
                }
            }
        }

        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(toVectorList(data, n, DIM), DIM);
        KMeans.Config config = KMeans.Config.builder().seed(SEED).parallel(false).build();
        KMeans.Result result = KMeans.cluster(vectors, 4, config);

        // Each centroid should be within 1.0 of one of the true centers
        for (int c = 0; c < 4; c++) {
            float[] centroid = result.getCentroid(c);
            boolean nearAny = false;
            for (float[] center : centers) {
                float dist = 0;
                for (int d = 0; d < DIM; d++) {
                    float diff = centroid[d] - center[d];
                    dist += diff * diff;
                }
                if (dist < 1.0f) {
                    nearAny = true;
                    break;
                }
            }
            assertTrue("Centroid " + c + " should be near a true center", nearAny);
        }
    }

    // ========== Helpers ==========

    private static List<float[]> toVectorList(float[] data, int n, int dim) {
        List<float[]> vecs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            System.arraycopy(data, i * dim, v, 0, dim);
            vecs.add(v);
        }
        return vecs;
    }

    private static ClusterANNVectorValues makeRandom(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vecs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++)
                v[d] = rng.nextFloat();
            vecs.add(v);
        }
        return ClusterANNVectorValues.fromList(vecs, dim);
    }

    private static ClusterANNVectorValues makeGaussianBlobs(int numBlobs, int perBlob, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vecs = new ArrayList<>(numBlobs * perBlob);
        for (int b = 0; b < numBlobs; b++) {
            float[] center = new float[dim];
            for (int d = 0; d < dim; d++)
                center[d] = rng.nextFloat() * 20f;
            for (int i = 0; i < perBlob; i++) {
                float[] v = new float[dim];
                for (int d = 0; d < dim; d++)
                    v[d] = center[d] + (float) rng.nextGaussian() * 0.5f;
                vecs.add(v);
            }
        }
        return ClusterANNVectorValues.fromList(vecs, dim);
    }

    private static void assertFloatArrayEquals(float[] a, float[] b, float delta) {
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            assertEquals("Mismatch at index " + i, a[i], b[i], delta);
        }
    }
}
