/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.opensearch.knn.KNNTestCase;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Tests for {@link IVFIndex} covering build, search, SOAR, recall, and all metrics.
 */
public class IVFIndexTests extends KNNTestCase {

    private static final int DIM = 16;
    private static final long SEED = 789L;

    // ========== Build ==========

    public void testBuildSmallDataset() {
        VectorData vectors = makeGaussianBlobs(3, 30, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder().numCentroids(10).metric(DistanceMetric.L2).seed(SEED).parallel(false).build();

        IVFIndex index = IVFIndex.build(vectors, config);

        assertTrue(index.numCentroids() > 0);
        assertTrue(index.numCentroids() <= 90); // clamped to n
        assertEquals(DIM, index.dimension());
    }

    public void testBuildLargeDataset() {
        VectorData vectors = makeGaussianBlobs(5, 200, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(50)
            .targetClusterSize(100)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .kmeansIterations(10)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);

        assertTrue("Should have multiple centroids", index.numCentroids() > 1);
        // All vectors should appear in posting lists
        int totalInPostings = 0;
        for (int[] posting : index.primaryPostings()) {
            totalInPostings += posting.length;
        }
        assertEquals(1000, totalInPostings);
    }

    // ========== Search ==========

    public void testSearchFindsExactMatch() {
        int n = 500;
        VectorData vectors = makeRandom(n, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(20)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);

        // Query with an existing vector — should find itself at distance 0
        float[] query = vectors.getVectorCopy(42);
        IVFIndex.SearchResult[] results = index.search(query, 1, 20, vectors);

        assertTrue(results.length > 0);
        assertEquals(42, results[0].docId);
        assertEquals(0f, results[0].distance, 1e-5f);
    }

    public void testSearchReturnsKResults() {
        VectorData vectors = makeRandom(1000, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(20)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);
        float[] query = vectors.getVectorCopy(0);

        IVFIndex.SearchResult[] results = index.search(query, 10, 20, vectors);

        assertEquals(10, results.length);
        // Results should be sorted by distance
        for (int i = 1; i < results.length; i++) {
            assertTrue("Results should be sorted", results[i].distance >= results[i - 1].distance);
        }
    }

    public void testSearchNprobeAffectsRecall() {
        VectorData vectors = makeGaussianBlobs(10, 100, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(50)
            .targetClusterSize(50)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);

        // Compute ground truth (brute force)
        float[] query = vectors.getVectorCopy(500);
        int[] groundTruth = bruteForceKNN(vectors, query, 10, DistanceMetric.L2);

        // Low nprobe
        IVFIndex.SearchResult[] lowProbe = index.search(query, 10, 1, vectors);
        float recallLow = computeRecall(lowProbe, groundTruth);

        // High nprobe
        IVFIndex.SearchResult[] highProbe = index.search(query, 10, 50, vectors);
        float recallHigh = computeRecall(highProbe, groundTruth);

        assertTrue("Higher nprobe should give better recall: low=" + recallLow + " high=" + recallHigh, recallHigh >= recallLow);
        assertTrue("Full probe should give perfect recall", recallHigh >= 0.9f);
    }

    // ========== SOAR ==========

    public void testSOARImprovesRecall() {
        VectorData vectors = makeGaussianBlobs(10, 100, DIM, SEED);

        IVFIndex.Config noSoar = IVFIndex.Config.builder()
            .numCentroids(30)
            .targetClusterSize(100)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();
        IVFIndex.Config withSoar = IVFIndex.Config.builder()
            .numCentroids(30)
            .targetClusterSize(100)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(1.0f)
            .build();

        IVFIndex indexNoSoar = IVFIndex.build(vectors, noSoar);
        IVFIndex indexWithSoar = IVFIndex.build(vectors, withSoar);

        // Average recall over multiple queries with low nprobe (where SOAR helps most)
        float avgRecallNoSoar = 0f, avgRecallWithSoar = 0f;
        int numQueries = 20;
        Random rng = new Random(SEED);
        for (int q = 0; q < numQueries; q++) {
            int queryIdx = rng.nextInt(vectors.numVectors());
            float[] query = vectors.getVectorCopy(queryIdx);
            int[] gt = bruteForceKNN(vectors, query, 10, DistanceMetric.L2);

            IVFIndex.SearchResult[] rNoSoar = indexNoSoar.search(query, 10, 3, vectors);
            IVFIndex.SearchResult[] rWithSoar = indexWithSoar.search(query, 10, 3, vectors);

            avgRecallNoSoar += computeRecall(rNoSoar, gt);
            avgRecallWithSoar += computeRecall(rWithSoar, gt);
        }
        avgRecallNoSoar /= numQueries;
        avgRecallWithSoar /= numQueries;

        assertTrue(
            "SOAR should improve recall at low nprobe: noSoar=" + avgRecallNoSoar + " withSoar=" + avgRecallWithSoar,
            avgRecallWithSoar >= avgRecallNoSoar
        );
    }

    public void testSOARPostingListsPopulated() {
        VectorData vectors = makeRandom(1000, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(10)
            .targetClusterSize(100)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(1.0f)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);

        int totalSoar = 0;
        for (int[] posting : index.soarPostings()) {
            totalSoar += posting.length;
        }
        assertTrue("SOAR posting lists should be populated", totalSoar > 0);
    }

    public void testNoSOARWhenLambdaZero() {
        VectorData vectors = makeRandom(500, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(10)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);

        for (int[] posting : index.soarPostings()) {
            assertEquals(0, posting.length);
        }
    }

    // ========== Distance Metrics ==========

    public void testSearchWithInnerProduct() {
        VectorData vectors = makeRandom(500, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(10)
            .targetClusterSize(100)
            .metric(DistanceMetric.INNER_PRODUCT)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);
        float[] query = vectors.getVectorCopy(0);

        // Use full probe to ensure we find the self-match
        IVFIndex.SearchResult[] results = index.search(query, 5, index.numCentroids(), vectors);
        assertEquals(5, results.length);
        // Self should be in results (highest dot product with itself)
        boolean foundSelf = false;
        for (IVFIndex.SearchResult r : results) {
            if (r.docId == 0) {
                foundSelf = true;
                break;
            }
        }
        assertTrue("Self should be in top-5 with full probe", foundSelf);
    }

    public void testSearchWithCosine() {
        VectorData vectors = makeRandom(500, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(10)
            .metric(DistanceMetric.COSINE)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);
        float[] query = vectors.getVectorCopy(0);

        IVFIndex.SearchResult[] results = index.search(query, 5, 10, vectors);
        assertEquals(5, results.length);
        assertEquals(0, results[0].docId);
        assertEquals(0f, results[0].distance, 1e-5f); // cosine distance to self = 0
    }

    // ========== Edge Cases ==========

    public void testSearchEmptyIndex() {
        VectorData vectors = makeRandom(10, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(5)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);
        float[] query = new float[DIM]; // zero vector

        IVFIndex.SearchResult[] results = index.search(query, 3, 5, vectors);
        assertTrue(results.length <= 3);
    }

    public void testNprobeGreaterThanCentroids() {
        VectorData vectors = makeRandom(100, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(5)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(0)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);
        float[] query = vectors.getVectorCopy(0);

        // nprobe > numCentroids should not crash
        IVFIndex.SearchResult[] results = index.search(query, 10, 1000, vectors);
        assertEquals(10, results.length);
    }

    // ========== Recall Validation ==========

    public void testRecallAbove90PercentWithFullProbe() {
        VectorData vectors = makeGaussianBlobs(10, 100, DIM, SEED);
        IVFIndex.Config config = IVFIndex.Config.builder()
            .numCentroids(30)
            .targetClusterSize(100)
            .metric(DistanceMetric.L2)
            .seed(SEED)
            .parallel(false)
            .soarLambda(1.0f)
            .build();

        IVFIndex index = IVFIndex.build(vectors, config);

        float totalRecall = 0f;
        int numQueries = 50;
        Random rng = new Random(SEED + 1);
        for (int q = 0; q < numQueries; q++) {
            float[] query = vectors.getVectorCopy(rng.nextInt(vectors.numVectors()));
            int[] gt = bruteForceKNN(vectors, query, 10, DistanceMetric.L2);
            IVFIndex.SearchResult[] results = index.search(query, 10, index.numCentroids(), vectors);
            totalRecall += computeRecall(results, gt);
        }
        float avgRecall = totalRecall / numQueries;

        assertTrue("Full-probe recall should be >= 0.95, got " + avgRecall, avgRecall >= 0.95f);
    }

    // ========== Helpers ==========

    private static VectorData makeRandom(int n, int dim, long seed) {
        Random rng = new Random(seed);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++)
            data[i] = rng.nextFloat();
        return new VectorData(data, n, dim);
    }

    private static VectorData makeGaussianBlobs(int numBlobs, int perBlob, int dim, long seed) {
        Random rng = new Random(seed);
        int n = numBlobs * perBlob;
        float[] data = new float[n * dim];
        for (int b = 0; b < numBlobs; b++) {
            float[] center = new float[dim];
            for (int d = 0; d < dim; d++)
                center[d] = rng.nextFloat() * 30f;
            for (int i = 0; i < perBlob; i++) {
                int idx = b * perBlob + i;
                for (int d = 0; d < dim; d++) {
                    data[idx * dim + d] = center[d] + (float) rng.nextGaussian() * 0.5f;
                }
            }
        }
        return new VectorData(data, n, dim);
    }

    private static int[] bruteForceKNN(VectorData vectors, float[] query, int k, DistanceMetric metric) {
        int n = vectors.numVectors();
        int dim = vectors.dimension();
        float[] data = vectors.data();

        float[] dists = new float[n];
        for (int i = 0; i < n; i++) {
            dists[i] = metric.distance(query, 0, data, i * dim, dim);
        }

        // Find top-k smallest
        int[] indices = new int[n];
        for (int i = 0; i < n; i++)
            indices[i] = i;

        // Partial sort
        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (dists[indices[j]] < dists[indices[minIdx]]) minIdx = j;
            }
            int tmp = indices[i];
            indices[i] = indices[minIdx];
            indices[minIdx] = tmp;
        }

        int[] result = new int[k];
        System.arraycopy(indices, 0, result, 0, k);
        return result;
    }

    private static float computeRecall(IVFIndex.SearchResult[] results, int[] groundTruth) {
        Set<Integer> gtSet = new HashSet<>();
        for (int id : groundTruth)
            gtSet.add(id);

        int hits = 0;
        for (IVFIndex.SearchResult r : results) {
            if (gtSet.contains(r.docId)) hits++;
        }
        return (float) hits / groundTruth.length;
    }
}
