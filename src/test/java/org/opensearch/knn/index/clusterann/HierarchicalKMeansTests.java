/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.opensearch.knn.KNNTestCase;

import java.util.Random;

/**
 * Tests for {@link HierarchicalKMeans} covering recursive splitting, depth limits, and adaptive k.
 */
public class HierarchicalKMeansTests extends KNNTestCase {

    private static final int DIM = 8;
    private static final long SEED = 456L;

    // ========== Basic Splitting ==========

    public void testSplitsLargeClusters() {
        int n = 2000;
        int targetSize = 200;
        VectorData vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(targetSize)
            .maxK(32)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // Should produce multiple centroids
        assertTrue("Should have > 1 centroid for " + n + " vectors with targetSize=" + targetSize, result.numCentroids() > 1);
        // Expected: roughly n/targetSize = 10 centroids (±50%)
        assertTrue("Expected ~10 centroids, got " + result.numCentroids(), result.numCentroids() >= 5 && result.numCentroids() <= 30);
    }

    public void testSmallDatasetSingleCentroid() {
        int n = 50;
        int targetSize = 100;
        VectorData vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(targetSize)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals("Small dataset should produce 1 centroid", 1, result.numCentroids());
    }

    public void testEmptyInput() {
        VectorData vectors = new VectorData(new float[0], 0, DIM);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().build();
        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(0, result.numCentroids());
        assertEquals(0, result.assignments().length);
    }

    // ========== Depth Limit ==========

    public void testMaxDepthRespected() {
        int n = 10000;
        VectorData vectors = makeRandom(n, DIM, SEED);

        // Very small targetSize + depth=1 should limit splitting
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(10)  // would need many levels
            .maxK(128)
            .maxDepth(1)     // but limited to 1 level
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).maxIterations(5).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // With depth=1, max centroids = maxK = 128
        assertTrue("Depth limit should cap centroids, got " + result.numCentroids(), result.numCentroids() <= 128);
    }

    public void testDeepRecursion() {
        int n = 5000;
        VectorData vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(50)
            .maxK(4)  // small maxK forces deeper recursion
            .maxDepth(10)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).maxIterations(10).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // Should produce roughly n/targetSize = 100 centroids
        assertTrue("Expected many centroids from deep recursion, got " + result.numCentroids(), result.numCentroids() >= 30);
    }

    // ========== Assignment Quality ==========

    public void testAllVectorsAssigned() {
        int n = 1000;
        VectorData vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(100)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(n, result.assignments().length);
        for (int i = 0; i < n; i++) {
            int a = result.assignments()[i];
            assertTrue("Assignment " + a + " out of range [0, " + result.numCentroids() + ")", a >= 0 && a < result.numCentroids());
        }
    }

    public void testCentroidDimensionCorrect() {
        int n = 500;
        VectorData vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(100)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(result.numCentroids() * DIM, result.centroids().length);
        assertEquals(DIM, result.dimension());
    }

    // ========== Adaptive K ==========

    public void testAdaptiveKFormula() {
        // With targetSize=500 and 2000 vectors: k = min(maxK, (2000+250)/500) = min(128, 4) = 4
        int n = 2000;
        VectorData vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(500)
            .maxK(128)
            .maxDepth(1)  // single level to test formula directly
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // At depth=1 with targetSize=500, should get ~4 centroids (some may not split further)
        assertTrue("Expected 3-5 centroids, got " + result.numCentroids(), result.numCentroids() >= 3 && result.numCentroids() <= 6);
    }

    // ========== Determinism ==========

    public void testDeterministic() {
        VectorData vectors = makeRandom(1000, DIM, SEED);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(100)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result r1 = HierarchicalKMeans.cluster(vectors, config);
        HierarchicalKMeans.Result r2 = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(r1.numCentroids(), r2.numCentroids());
        assertArrayEquals(r1.assignments(), r2.assignments());
    }

    // ========== GetCentroid ==========

    public void testGetCentroid() {
        VectorData vectors = makeRandom(500, DIM, SEED);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(100)
            .kmeansConfig(KMeans.Config.builder().seed(SEED).parallel(false).build())
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        for (int c = 0; c < result.numCentroids(); c++) {
            float[] centroid = result.getCentroid(c);
            assertEquals(DIM, centroid.length);
            // Verify matches flat array
            for (int d = 0; d < DIM; d++) {
                assertEquals(result.centroids()[c * DIM + d], centroid[d], 0f);
            }
        }
    }

    // ========== Helpers ==========

    private static VectorData makeRandom(int n, int dim, long seed) {
        Random rng = new Random(seed);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++)
            data[i] = rng.nextFloat() * 10f;
        return new VectorData(data, n, dim);
    }
}
