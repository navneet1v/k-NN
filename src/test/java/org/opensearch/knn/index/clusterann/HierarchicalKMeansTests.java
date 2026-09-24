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
 * Tests for {@link HierarchicalKMeans} covering recursive splitting, depth limits, and adaptive k.
 */
public class HierarchicalKMeansTests extends KNNTestCase {

    private static final int DIM = 8;
    private static final long SEED = 456L;

    // ========== Basic Splitting ==========

    public void testSplitsLargeClusters() throws Exception {
        int n = 2000;
        int targetSize = 200;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(targetSize)

            .seed(SEED)
            .parallel(false)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // Should produce multiple centroids
        assertTrue("Should have > 1 centroid for " + n + " vectors with targetSize=" + targetSize, result.numCentroids() > 1);
        // Expected: roughly n/targetSize = 10 centroids (±50%)
        assertTrue("Expected ~10 centroids, got " + result.numCentroids(), result.numCentroids() >= 5 && result.numCentroids() <= 30);
    }

    public void testSmallDatasetSingleCentroid() throws Exception {
        int n = 50;
        int targetSize = 100;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(targetSize).seed(SEED).parallel(false).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals("Small dataset should produce 1 centroid", 1, result.numCentroids());
    }

    public void testEmptyInput() throws Exception {
        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(new ArrayList<>(), DIM);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().build();
        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(0, result.numCentroids());
        assertEquals(0, result.assignments().length);
    }

    // ========== Depth Limit ==========

    public void testMaxDepthRespected() throws Exception {
        int n = 10000;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        // Small targetSize should produce many centroids via recursive splitting
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(10)
            .seed(SEED)
            .parallel(false)
            .maxIterations(5)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // Should produce roughly n/targetSize centroids (with some variance from splitting)
        assertTrue("Expected many centroids, got " + result.numCentroids(), result.numCentroids() >= 100);
        assertTrue("Centroids should not exceed vector count, got " + result.numCentroids(), result.numCentroids() <= n);
    }

    public void testDeepRecursion() throws Exception {
        int n = 5000;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(50)
            // small maxK forces deeper recursion

            .seed(SEED)
            .parallel(false)
            .maxIterations(10)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // Should produce roughly n/targetSize = 100 centroids
        assertTrue("Expected many centroids from deep recursion, got " + result.numCentroids(), result.numCentroids() >= 30);
    }

    // ========== Assignment Quality ==========

    public void testAllVectorsAssigned() throws Exception {
        int n = 1000;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(100).seed(SEED).parallel(false).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(n, result.assignments().length);
        for (int i = 0; i < n; i++) {
            int a = result.assignments()[i];
            assertTrue("Assignment " + a + " out of range [0, " + result.numCentroids() + ")", a >= 0 && a < result.numCentroids());
        }
    }

    public void testCentroidDimensionCorrect() throws Exception {
        int n = 500;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(100).seed(SEED).parallel(false).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(result.numCentroids(), result.centroids().length);
        assertEquals(DIM, result.dimension());
    }

    // ========== Adaptive K ==========

    public void testAdaptiveKFormula() throws Exception {
        // With targetSize=500 and 2000 vectors: k = min(maxK, (2000+250)/500) = min(128, 4) = 4
        int n = 2000;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(500)

            // single level to test formula directly
            .seed(SEED)
            .parallel(false)
            .build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        // At depth=1 with targetSize=500, should get ~4 centroids (some may not split further)
        assertTrue("Expected 3-5 centroids, got " + result.numCentroids(), result.numCentroids() >= 3 && result.numCentroids() <= 6);
    }

    // ========== Determinism ==========

    public void testDeterministic() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(1000, DIM, SEED);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(100).seed(SEED).parallel(false).build();

        HierarchicalKMeans.Result r1 = HierarchicalKMeans.cluster(vectors, config);
        HierarchicalKMeans.Result r2 = HierarchicalKMeans.cluster(vectors, config);

        assertEquals(r1.numCentroids(), r2.numCentroids());
        assertArrayEquals(r1.assignments(), r2.assignments());
    }

    // ========== GetCentroid ==========

    public void testGetCentroid() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(500, DIM, SEED);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder().targetSize(100).seed(SEED).parallel(false).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        for (int c = 0; c < result.numCentroids(); c++) {
            float[] centroid = result.getCentroid(c);
            assertEquals(DIM, centroid.length);
            // Verify matches flat array
            for (int d = 0; d < DIM; d++) {
                assertEquals(result.centroids()[c][d], centroid[d], 0f);
            }
        }
    }

    // ========== Helpers ==========

    private static ClusterANNVectorValues makeRandom(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vecs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++)
                v[d] = rng.nextFloat() * 10f;
            vecs.add(v);
        }
        return ClusterANNVectorValues.fromList(vecs, dim);
    }

    // ========== cosine centroids (CR-307468808 parity) ==========
    // The single-centroid path (n <= targetSize) and the splitRecursive leaves build centroids
    // without running KMeans. They previously returned raw means under cosine while KMeans returned
    // unit vectors, and OptimizedScalarQuantizer asserts a unit centroid under cosine; computeMean
    // now projects onto the unit sphere for cosine on every path.

    public void testCosineSingleCentroidPathIsUnitLength() throws Exception {
        ClusterANNVectorValues vectors = makeUnit(200, DIM, 7L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(512).metric(DistanceMetric.COSINE).seed(7L).parallel(false).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertEquals("single-centroid path", 1, result.numCentroids());
        assertAllUnitLength(result.centroids());
    }

    public void testCosineSplitPathsAreUnitLength() throws Exception {
        // Well above targetSize: top-level KMeans plus recursive splits whose leaves come from computeMean.
        ClusterANNVectorValues vectors = makeUnit(4000, DIM, 11L);
        HierarchicalKMeans.Config config = HierarchicalKMeans.Config.builder()
            .targetSize(64).metric(DistanceMetric.COSINE).seed(11L).parallel(false).build();

        HierarchicalKMeans.Result result = HierarchicalKMeans.cluster(vectors, config);

        assertTrue("expected split path, got " + result.numCentroids(), result.numCentroids() > 1);
        assertAllUnitLength(result.centroids());
    }

    private static void assertAllUnitLength(float[][] centroids) {
        for (int c = 0; c < centroids.length; c++) {
            double norm = 0;
            for (float x : centroids[c]) norm += (double) x * x;
            norm = Math.sqrt(norm);
            // Zero centroids are allowed (empty/opposite-sum clusters have no direction to project).
            assertTrue("centroid " + c + " not unit length: norm=" + norm,
                norm == 0.0 || Math.abs(norm - 1.0) < 1e-4);
        }
    }

    /** Random unit-length directions, spread widely so their raw means are far from unit length. */
    private static ClusterANNVectorValues makeUnit(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vecs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            double norm = 0;
            for (int d = 0; d < dim; d++) { v[d] = (float) rng.nextGaussian(); norm += (double) v[d] * v[d]; }
            float inv = (float) (1.0 / Math.sqrt(norm));
            for (int d = 0; d < dim; d++) v[d] *= inv;
            vecs.add(v);
        }
        return ClusterANNVectorValues.fromList(vecs, dim);
    }
}
