/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.opensearch.knn.KNNTestCase;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Tests for {@link IVFIndexBuilder} covering clustering, SOAR, and all metrics.
 */
public class IVFIndexBuilderTests extends KNNTestCase {

    private static final int DIM = 16;
    private static final long SEED = 789L;

    public void testBuildSmallDataset() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(3, 30, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 50, DistanceMetric.L2, 1.0f, null, SEED, false);

        assertTrue(result.numCentroids() > 0);
        assertEquals(90, result.assignments().length);
        assertEquals(90, result.soarAssignments().length);
    }

    public void testBuildLargeDataset() throws Exception {
        ClusterANNVectorValues vectors = makeGaussianBlobs(5, 200, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 1.0f, null, SEED, false);

        assertTrue("Should have multiple centroids", result.numCentroids() > 1);
        // All vectors should appear in primary posting lists
        int totalInPostings = 0;
        for (int[] posting : result.primaryPostingLists()) {
            totalInPostings += posting.length;
        }
        assertEquals(1000, totalInPostings);
    }

    public void testAllVectorsAssigned() throws Exception {
        int n = 500;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 0f, null, SEED, false);

        // Every vector should have a valid assignment
        for (int i = 0; i < n; i++) {
            assertTrue("Assignment out of range", result.assignments()[i] >= 0 && result.assignments()[i] < result.numCentroids());
        }
    }

    public void testSOARAssignments() throws Exception {
        int n = 500;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 1.0f, null, SEED, false);

        // SOAR should assign most vectors to a different centroid than primary
        int soarCount = 0;
        for (int i = 0; i < n; i++) {
            if (result.soarAssignments()[i] >= 0) {
                soarCount++;
                assertNotEquals("SOAR should differ from primary", result.assignments()[i], result.soarAssignments()[i]);
            }
        }
        assertTrue("SOAR should assign most vectors", soarCount > n / 2);
    }

    public void testSOARDisabled() throws Exception {
        int n = 200;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 0f, null, SEED, false);

        // All SOAR assignments should be -1
        for (int s : result.soarAssignments()) {
            assertEquals(-1, s);
        }
    }

    public void testPostingListsComplete() throws Exception {
        int n = 300;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 1.0f, null, SEED, false);

        // Primary posting lists should contain all vectors exactly once
        Set<Integer> seen = new HashSet<>();
        for (int[] posting : result.primaryPostingLists()) {
            for (int ord : posting) {
                assertTrue("Duplicate in posting lists", seen.add(ord));
            }
        }
        assertEquals(n, seen.size());
    }

    public void testInitialCentroids() throws Exception {
        int n = 500;
        ClusterANNVectorValues vectors = makeRandom(n, DIM, SEED);

        // Provide initial centroids (simulating merge reservoir)
        float[][] initial = new float[5][DIM];
        Random rng = new Random(SEED);
        for (int i = 0; i < 5; i++) {
            for (int d = 0; d < DIM; d++) {
                initial[i][d] = rng.nextFloat();
            }
        }

        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 1.0f, initial, SEED, false);
        assertTrue(result.numCentroids() > 0);
        assertEquals(n, result.assignments().length);
    }

    public void testInnerProductMetric() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(200, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.INNER_PRODUCT, 1.0f, null, SEED, false);
        assertTrue(result.numCentroids() > 0);
    }

    public void testCosineMetric() throws Exception {
        ClusterANNVectorValues vectors = makeRandom(200, DIM, SEED);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.COSINE, 1.0f, null, SEED, false);
        assertTrue(result.numCentroids() > 0);
    }

    public void testEmptyInput() throws Exception {
        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(List.of(), DIM);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 1.0f, null, SEED, false);
        assertEquals(0, result.numCentroids());
    }

    public void testSingleVector() throws Exception {
        List<float[]> vecs = new ArrayList<>();
        vecs.add(new float[DIM]);
        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromList(vecs, DIM);
        ClusteringResult result = IVFIndexBuilder.build(vectors, 100, DistanceMetric.L2, 0f, null, SEED, false);
        assertEquals(1, result.numCentroids());
        assertEquals(0, result.assignments()[0]);
    }

    // ========== Helpers ==========

    private static ClusterANNVectorValues makeRandom(int n, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vectors = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++)
                v[d] = rng.nextFloat();
            vectors.add(v);
        }
        return ClusterANNVectorValues.fromList(vectors, dim);
    }

    private static ClusterANNVectorValues makeGaussianBlobs(int numBlobs, int perBlob, int dim, long seed) {
        Random rng = new Random(seed);
        List<float[]> vectors = new ArrayList<>(numBlobs * perBlob);
        for (int b = 0; b < numBlobs; b++) {
            float[] center = new float[dim];
            for (int d = 0; d < dim; d++)
                center[d] = rng.nextFloat() * 10;
            for (int i = 0; i < perBlob; i++) {
                float[] v = new float[dim];
                for (int d = 0; d < dim; d++)
                    v[d] = center[d] + (float) rng.nextGaussian() * 0.5f;
                vectors.add(v);
            }
        }
        return ClusterANNVectorValues.fromList(vectors, dim);
    }
}
