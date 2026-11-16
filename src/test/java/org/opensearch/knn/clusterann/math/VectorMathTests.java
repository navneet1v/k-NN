/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.math;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorMathTests {

    // ========== squareDistance ==========

    @Test
    void testSquareDistance_identicalVectors_returnsZero() {
        float[] a = { 1.0f, 2.0f, 3.0f };
        assertEquals(0.0f, VectorMath.squareDistance(a, a), 1e-6f);
    }

    @Test
    void testSquareDistance_knownValues() {
        float[] a = { 1.0f, 0.0f, 0.0f };
        float[] b = { 0.0f, 1.0f, 0.0f };
        // (1-0)^2 + (0-1)^2 + (0-0)^2 = 2.0
        assertEquals(2.0f, VectorMath.squareDistance(a, b), 1e-6f);
    }

    @Test
    void testSquareDistance_symmetric() {
        float[] a = { 1.5f, -2.3f, 4.1f };
        float[] b = { -0.5f, 3.2f, 1.0f };
        assertEquals(VectorMath.squareDistance(a, b), VectorMath.squareDistance(b, a), 1e-6f);
    }

    @Test
    void testSquareDistance_highDimensional() {
        int dim = 768;
        float[] a = new float[dim];
        float[] b = new float[dim];
        for (int i = 0; i < dim; i++) {
            a[i] = (float) Math.sin(i);
            b[i] = (float) Math.cos(i);
        }
        float dist = VectorMath.squareDistance(a, b);
        assertTrue(dist > 0, "Distance should be positive");
    }

    // ========== dotProduct ==========

    @Test
    void testDotProduct_orthogonal_returnsZero() {
        float[] a = { 1.0f, 0.0f, 0.0f };
        float[] b = { 0.0f, 1.0f, 0.0f };
        assertEquals(0.0f, VectorMath.dotProduct(a, b), 1e-6f);
    }

    @Test
    void testDotProduct_parallel_returnsNormSquared() {
        float[] a = { 3.0f, 4.0f };
        assertEquals(25.0f, VectorMath.dotProduct(a, a), 1e-6f);
    }

    @Test
    void testDotProduct_knownValues() {
        float[] a = { 1.0f, 2.0f, 3.0f };
        float[] b = { 4.0f, 5.0f, 6.0f };
        // 1*4 + 2*5 + 3*6 = 32
        assertEquals(32.0f, VectorMath.dotProduct(a, b), 1e-6f);
    }

    // ========== cosine ==========

    @Test
    void testCosine_identicalVectors_returnsOne() {
        float[] a = { 1.0f, 2.0f, 3.0f };
        assertEquals(1.0f, VectorMath.cosine(a, a), 1e-5f);
    }

    @Test
    void testCosine_orthogonal_returnsZero() {
        float[] a = { 1.0f, 0.0f };
        float[] b = { 0.0f, 1.0f };
        assertEquals(0.0f, VectorMath.cosine(a, b), 1e-5f);
    }

    @Test
    void testCosine_opposite_returnsNegativeOne() {
        float[] a = { 1.0f, 0.0f };
        float[] b = { -1.0f, 0.0f };
        assertEquals(-1.0f, VectorMath.cosine(a, b), 1e-5f);
    }

    @Test
    void testCosine_scaleInvariant() {
        float[] a = { 1.0f, 2.0f, 3.0f };
        float[] b = { 2.0f, 4.0f, 6.0f }; // same direction, 2x scale
        assertEquals(1.0f, VectorMath.cosine(a, b), 1e-5f);
    }

    // ========== findNearestCentroidBulk ==========

    @Test
    void testFindNearestCentroidBulk_findsClosest() {
        float[] query = { 1.0f, 0.0f };
        float[][] centroids = { { 10.0f, 10.0f }, { 0.9f, 0.1f }, { -5.0f, -5.0f } };
        float[] flat = VectorMath.flattenCentroids(centroids);
        float[] dists = new float[centroids.length];
        assertEquals(1, VectorMath.findNearestCentroidBulk(query, flat, centroids.length, 2, dists));
        assertTrue(dists[1] < dists[0] && dists[1] < dists[2], "closest centroid has smallest distance");
    }

    @Test
    void testFindNearestCentroidBulk_exactMatchZeroDistance() {
        float[][] centroids = { { 0.0f, 0.0f }, { 3.0f, 4.0f }, { 6.0f, 8.0f } };
        float[] flat = VectorMath.flattenCentroids(centroids);
        float[] dists = new float[centroids.length];
        assertEquals(1, VectorMath.findNearestCentroidBulk(new float[] { 3.0f, 4.0f }, flat, 3, 2, dists));
        assertEquals(0.0f, dists[1], 1e-6f);
    }

    // ========== flattenCentroids ==========

    @Test
    void testFlattenCentroids() {
        float[][] centroids = { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } };
        float[] flat = VectorMath.flattenCentroids(centroids);
        assertEquals(6, flat.length);
        assertEquals(1.0f, flat[0], 1e-6f);
        assertEquals(2.0f, flat[1], 1e-6f);
        assertEquals(3.0f, flat[2], 1e-6f);
        assertEquals(4.0f, flat[3], 1e-6f);
        assertEquals(5.0f, flat[4], 1e-6f);
        assertEquals(6.0f, flat[5], 1e-6f);
    }

    // ========== distanceFunction (VectorSimilarityFunction -> lower-is-better distance) ==========

    @Test
    void testDistanceFunction_euclidean_identicalZero() {
        VectorMath.DistanceFunction fn = VectorMath.distanceFunction(VectorSimilarityFunction.EUCLIDEAN);
        float[] a = { 1.0f, 2.0f, 3.0f };
        assertEquals(0.0f, fn.distance(a, a), 1e-6f);
    }

    @Test
    void testDistanceFunction_euclidean_knownDistance() {
        VectorMath.DistanceFunction fn = VectorMath.distanceFunction(VectorSimilarityFunction.EUCLIDEAN);
        float[] a = { 0.0f, 0.0f };
        float[] b = { 3.0f, 4.0f };
        // 3^2 + 4^2 = 25 (squared L2)
        assertEquals(25.0f, fn.distance(a, b), 1e-5f);
    }

    @Test
    void testDistanceFunction_whenDotProduct_thenThrows() {
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> VectorMath.distanceFunction(VectorSimilarityFunction.DOT_PRODUCT)
        );

        assertTrue(e.getMessage().contains("MAXIMUM_INNER_PRODUCT"), e.getMessage());
    }

    @Test
    void testDistanceFunction_maximumInnerProduct_matchesDotProduct() {
        VectorMath.DistanceFunction mip = VectorMath.distanceFunction(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
        float[] a = { 1.5f, -2.0f, 0.5f };
        float[] b = { 0.25f, 1.0f, -3.0f };
        // Both DOT_PRODUCT and MAXIMUM_INNER_PRODUCT map to negated inner product for clustering.
        assertEquals(-VectorMath.dotProduct(a, b), mip.distance(a, b), 1e-6f);
    }

    @Test
    void testDistanceFunction_cosine_identicalZero_orthogonalOne_oppositeTwo() {
        VectorMath.DistanceFunction fn = VectorMath.distanceFunction(VectorSimilarityFunction.COSINE);
        float[] a = { 1.0f, 0.0f };
        assertEquals(0.0f, fn.distance(a, a), 1e-5f);
        assertEquals(1.0f, fn.distance(a, new float[] { 0.0f, 1.0f }), 1e-5f);
        assertEquals(2.0f, fn.distance(a, new float[] { -1.0f, 0.0f }), 1e-5f);
    }

    @Test
    void testDistanceFunction_cosine_scaleInvariant() {
        VectorMath.DistanceFunction fn = VectorMath.distanceFunction(VectorSimilarityFunction.COSINE);
        float[] a = { 1.0f, 2.0f, 3.0f };
        float[] b = { 2.0f, 4.0f, 6.0f };
        assertEquals(0.0f, fn.distance(a, b), 1e-4f);
    }

    @Test
    void testDistanceFunction_allMetrics_lowerIsBetter() {
        float[] query = { 1.0f, 0.0f, 0.0f };
        float[] close = { 0.9f, 0.1f, 0.0f };
        float[] far = { -1.0f, 0.0f, 0.0f };
        for (VectorSimilarityFunction metric : VectorSimilarityFunction.values()) {
            if (metric == VectorSimilarityFunction.DOT_PRODUCT) {
                continue;
            }
            VectorMath.DistanceFunction fn = VectorMath.distanceFunction(metric);
            assertTrue(fn.distance(query, close) < fn.distance(query, far), metric.name() + ": close should be less than far");
        }
    }

    @Test
    void testNearestCentroids_ordersByDistance() {
        float[][] centroids = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 5.0f, 0.0f }, { 10.0f, 0.0f } };
        VectorMath.DistanceFunction l2 = VectorMath.distanceFunction(VectorSimilarityFunction.EUCLIDEAN);
        // Nearest to centroid 0, excluding itself: 1 (d=1), then 2 (d=25).
        int[] nearest = VectorMath.nearestCentroids(centroids, 0, 2, l2);
        assertEquals(2, nearest.length);
        assertEquals(1, nearest[0]);
        assertEquals(2, nearest[1]);
    }

    // ========== centroidFromSum ==========

    @ParameterizedTest
    @EnumSource(value = VectorSimilarityFunction.class, names = { "EUCLIDEAN", "MAXIMUM_INNER_PRODUCT", "DOT_PRODUCT" })
    void testCentroidFromSum_isThePlainMean_forRawMetrics(VectorSimilarityFunction metric) {
        float[] sum = { 3.0f, -6.0f, 0.0f, 1.5f };

        float[] out = VectorMath.centroidFromSum(sum, 3, metric, new float[4]);

        assertArrayEquals(new float[] { 1.0f, -2.0f, 0.0f, 0.5f }, out, 0.0f);
    }

    @Test
    void testCentroidFromSum_projectsTheMeanOntoTheUnitSphere_forCosine() {
        // Two unit vectors 90 degrees apart: the mean has length sqrt(2)/2, the centroid must have length 1.
        float[] sum = { 1.0f, 1.0f, 0.0f };

        float[] out = VectorMath.centroidFromSum(sum, 2, VectorSimilarityFunction.COSINE, new float[3]);

        float inv = (float) (1.0 / Math.sqrt(2.0));
        assertArrayEquals(new float[] { inv, inv, 0.0f }, out, 1e-7f);
        assertTrue(VectorUtil.isUnitVector(out));
    }

    @Test
    void testCentroidFromSum_leavesAZeroSumAtZero_forCosine() {
        // Two opposite unit vectors: no direction to project onto.
        float[] out = VectorMath.centroidFromSum(new float[] { 0.0f, 0.0f }, 2, VectorSimilarityFunction.COSINE, new float[2]);

        assertArrayEquals(new float[] { 0.0f, 0.0f }, out, 0.0f);
    }

    @Test
    void testCentroidFromSum_canWriteInPlace() {
        float[] sum = { 2.0f, 4.0f };

        float[] out = VectorMath.centroidFromSum(sum, 2, VectorSimilarityFunction.EUCLIDEAN, sum);

        assertSame(sum, out);
        assertArrayEquals(new float[] { 1.0f, 2.0f }, sum, 0.0f);
    }
}
