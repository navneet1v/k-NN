/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * CLIP (Cosine-Law-based Inverted-list Pruning) metadata for a single field.
 *
 * <p>Stores per-cluster sorted centroid-to-vector L2 distances and empirically
 * calibrated λ parameters, enabling:
 * <ul>
 *   <li><b>Inter-cluster pruning</b> — skip entire cluster in O(1) if
 *       {@code (1-λ²)·‖q-c‖² > maxDist}</li>
 *   <li><b>Intra-cluster pruning</b> — binary search for valid vector range
 *       in O(log n), avoiding full cluster scan</li>
 *   <li><b>Cross-segment pruning</b> — skip entire segment using segment centroid bound</li>
 * </ul>
 *
 * <p>Based on: "CLIP: Lightweight Cosine-Law-Based Inverted-List Pruning for
 * IVF-Based Vector Search" (Song et al., 2026).
 *
 * <p>The bound works for inner product on normalized vectors via the identity:
 * {@code ⟨q,v⟩ = 1 - ‖q-v‖²/2}, so a lower bound on L2² gives an upper bound on IP.
 *
 * @see <a href="https://arxiv.org/abs/2606.29968">arXiv:2606.29968</a>
 */
public final class ClipPruningData {

    /** Number of distance slices for adaptive λ lookup. */
    public static final int NUM_SLICES = 20;

    /** Quantile for λ calibration — 0.0001 means 0.01% of angles may be below θ_min. */
    public static final float BETA = 0.0001f;

    /** Number of triplets sampled per cluster for λ calibration. */
    private static final int SAMPLES_PER_CLUSTER = 10000;

    // --- Per-field segment-level data ---
    private final float[] segmentCentroid;
    private final float segmentLambda;

    // --- Per-cluster data ---
    private final int numCentroids;
    private final float[][] sortedDistances;  // [centroid][sortedPosition] → ‖c-v‖
    private final int[][] sortPermutation;     // [centroid][sortedPosition] → original posting index
    private final float[][] lambdaSlices;      // [centroid][slice] → λ for that distance range
    private final float[] minDistSq;           // [centroid] → min ‖q-c‖² seen during calibration
    private final float[] maxDistSq;           // [centroid] → max ‖q-c‖² seen during calibration

    private ClipPruningData(
        float[] segmentCentroid,
        float segmentLambda,
        int numCentroids,
        float[][] sortedDistances,
        int[][] sortPermutation,
        float[][] lambdaSlices,
        float[] minDistSq,
        float[] maxDistSq
    ) {
        this.segmentCentroid = segmentCentroid;
        this.segmentLambda = segmentLambda;
        this.numCentroids = numCentroids;
        this.sortedDistances = sortedDistances;
        this.sortPermutation = sortPermutation;
        this.lambdaSlices = lambdaSlices;
        this.minDistSq = minDistSq;
        this.maxDistSq = maxDistSq;
    }

    // ========== Query-time API ==========

    /**
     * Get the calibrated λ for a given cluster and query-centroid squared distance.
     * Uses distance-adaptive slicing for tighter bounds.
     */
    public float getLambda(int centroidIdx, float distQC_sq) {
        float min = minDistSq[centroidIdx];
        float max = maxDistSq[centroidIdx];
        if (max <= min) return lambdaSlices[centroidIdx][0];

        float sliceLen = (max - min) / NUM_SLICES;
        int slice = Math.min((int) ((distQC_sq - min) / sliceLen), NUM_SLICES - 1);
        slice = Math.max(0, slice);
        return lambdaSlices[centroidIdx][slice];
    }

    /**
     * Inter-cluster pruning: compute upper bound on max IP for any vector in cluster.
     *
     * @param distQC_sq squared L2 distance from query to centroid (‖q-c‖²)
     * @param centroidIdx cluster index
     * @return upper bound on ⟨q,v⟩ for any v in this cluster; or Float.MAX_VALUE if no bound
     */
    public float upperBoundIP(int centroidIdx, float distQC_sq) {
        float lambda = getLambda(centroidIdx, distQC_sq);
        float lbMin = (1.0f - lambda * lambda) * distQC_sq;
        // ⟨q,v⟩ = 1 - ‖q-v‖²/2, so upper bound on IP = 1 - lbMin/2
        return 1.0f - lbMin / 2.0f;
    }

    /**
     * Inter-cluster pruning for L2 metric: compute lower bound on min L2² for any vector.
     *
     * @param distQC_sq squared L2 distance from query to centroid
     * @param centroidIdx cluster index
     * @return lower bound on ‖q-v‖² for any v in this cluster
     */
    public float lowerBoundL2Sq(int centroidIdx, float distQC_sq) {
        float lambda = getLambda(centroidIdx, distQC_sq);
        return (1.0f - lambda * lambda) * distQC_sq;
    }

    /**
     * Intra-cluster pruning: find the rightmost index in sorted distances
     * whose lower bound could still beat maxDist.
     *
     * <p>Returns the exclusive upper bound — scan vectors from [0, result).
     *
     * @param centroidIdx cluster index
     * @param distQC_sq squared L2 distance from query to centroid
     * @param maxDistL2Sq current k-th best L2² distance (threshold)
     * @return number of vectors to actually score (from sorted-nearest first)
     */
    public int intraPruneCount(int centroidIdx, float distQC_sq, float maxDistL2Sq) {
        float lambda = getLambda(centroidIdx, distQC_sq);
        float discriminant = maxDistL2Sq - (1.0f - lambda * lambda) * distQC_sq;
        if (discriminant < 0) return 0; // entire cluster pruned

        float distQC = (float) Math.sqrt(distQC_sq);
        float dRight = lambda * distQC + (float) Math.sqrt(discriminant);

        // Binary search in sorted distances for dRight
        float[] dists = sortedDistances[centroidIdx];
        int idx = Arrays.binarySearch(dists, dRight);
        if (idx < 0) idx = -idx - 1; // insertion point
        return Math.min(idx, dists.length);
    }

    /**
     * Get the original posting-list index for a vector at sorted position.
     * Used when posting lists are NOT physically reordered.
     */
    public int getOriginalIndex(int centroidIdx, int sortedPosition) {
        return sortPermutation[centroidIdx][sortedPosition];
    }

    /**
     * Get sorted distances array for a cluster (for external binary search).
     */
    public float[] getSortedDistances(int centroidIdx) {
        return sortedDistances[centroidIdx];
    }

    /**
     * Cross-segment pruning: upper bound on max IP for any vector in this segment.
     *
     * @param query query vector (unit-normalized)
     * @return upper bound on ⟨q,v⟩ for any v in this segment
     */
    public float segmentUpperBoundIP(float[] query) {
        float distSq = squaredL2(query, segmentCentroid);
        float lbMin = (1.0f - segmentLambda * segmentLambda) * distSq;
        return 1.0f - lbMin / 2.0f;
    }

    /**
     * Cross-segment pruning: lower bound on min L2² for any vector in this segment.
     */
    public float segmentLowerBoundL2Sq(float[] query) {
        float distSq = squaredL2(query, segmentCentroid);
        return (1.0f - segmentLambda * segmentLambda) * distSq;
    }

    /** Whether this data has segment-level centroid for cross-segment pruning. */
    public boolean hasSegmentCentroid() {
        return segmentCentroid != null;
    }

    // ========== Index-time calibration ==========

    /**
     * Calibrate CLIP pruning data from clustering results.
     *
     * @param centroids cluster centroids [numCentroids][dimension]
     * @param assignments per-vector cluster assignment
     * @param vectors all vectors in this segment
     * @param dimension vector dimensionality
     * @param seed random seed for sampling
     * @return calibrated ClipPruningData ready for writing
     */
    public static ClipPruningData calibrate(
        float[][] centroids,
        int[][] primaryPostings,
        float[][] vectors,
        int dimension,
        long seed
    ) {
        int numCentroids = centroids.length;
        int numVectors = vectors.length;
        Random rng = new Random(seed);

        // 1. Compute segment centroid (mean of all vectors)
        float[] segmentCentroid = new float[dimension];
        for (float[] v : vectors) {
            for (int d = 0; d < dimension; d++) {
                segmentCentroid[d] += v[d];
            }
        }
        for (int d = 0; d < dimension; d++) {
            segmentCentroid[d] /= numVectors;
        }

        // 2. Per-cluster: compute and sort centroid-to-vector distances
        float[][] sortedDistances = new float[numCentroids][];
        int[][] sortPermutation = new int[numCentroids][];

        for (int c = 0; c < numCentroids; c++) {
            int[] posting = primaryPostings[c];
            int count = posting.length;
            float[] dists = new float[count];
            int[] perm = new int[count];

            for (int i = 0; i < count; i++) {
                int ord = posting[i];
                dists[i] = (float) Math.sqrt(squaredL2(centroids[c], vectors[ord]));
                perm[i] = i; // maps to position in posting list
            }

            // Sort by distance, maintaining permutation
            sortByDistance(dists, perm, count);

            sortedDistances[c] = dists;
            sortPermutation[c] = perm;
        }

        // 3. Calibrate λ per cluster using empirical angle sampling
        float[][] lambdaSlices = new float[numCentroids][NUM_SLICES];
        float[] minDistSqArr = new float[numCentroids];
        float[] maxDistSqArr = new float[numCentroids];

        for (int c = 0; c < numCentroids; c++) {
            calibrateCluster(
                c, centroids[c], primaryPostings, vectors, dimension,
                lambdaSlices[c], minDistSqArr, maxDistSqArr, rng
            );
        }

        // 4. Calibrate segment-level λ
        float segLambda = calibrateSegmentLambda(
            segmentCentroid, vectors, dimension, rng
        );

        return new ClipPruningData(
            segmentCentroid, segLambda, numCentroids,
            sortedDistances, sortPermutation,
            lambdaSlices, minDistSqArr, maxDistSqArr
        );
    }

    /**
     * Calibrate λ slices for one cluster by sampling (query, centroid, vector) triplets
     * and measuring the angle ∠qcv.
     */
    private static void calibrateCluster(
        int clusterIdx,
        float[] centroid,
        int[][] allPostings,
        float[][] vectors,
        int dimension,
        float[] outLambdaSlices,
        float[] outMinDistSq,
        float[] outMaxDistSq,
        Random rng
    ) {
        int[] posting = allPostings[clusterIdx];
        int clusterSize = posting.length;
        if (clusterSize == 0) {
            Arrays.fill(outLambdaSlices, 0.0f);
            outMinDistSq[clusterIdx] = 0;
            outMaxDistSq[clusterIdx] = 1;
            return;
        }

        // Sample triplets: use vectors from OTHER clusters as proxy queries
        int totalVectors = vectors.length;
        int numSamples = Math.min(SAMPLES_PER_CLUSTER * clusterSize, totalVectors * 10);
        numSamples = Math.max(numSamples, 200);

        // Collect (distQC_sq, angle) pairs
        float[] sampleDistsSq = new float[numSamples];
        float[] sampleAngles = new float[numSamples];
        int sampleCount = 0;

        for (int s = 0; s < numSamples && sampleCount < numSamples; s++) {
            // Pick a random "query" vector (from any cluster, including this one)
            int qOrd = rng.nextInt(totalVectors);
            float[] q = vectors[qOrd];

            // Pick a random vector from THIS cluster
            int vIdx = rng.nextInt(clusterSize);
            int vOrd = posting[vIdx];
            float[] v = vectors[vOrd];

            // Compute ‖q-c‖² and ‖c-v‖²
            float distQC_sq = squaredL2(q, centroid);
            float distCV_sq = squaredL2(centroid, v);
            float distQV_sq = squaredL2(q, v);

            float distQC = (float) Math.sqrt(distQC_sq);
            float distCV = (float) Math.sqrt(distCV_sq);

            if (distQC < 1e-10f || distCV < 1e-10f) continue;

            // Compute angle ∠qcv via law of cosines:
            // cos(θ) = (‖q-c‖² + ‖c-v‖² - ‖q-v‖²) / (2·‖q-c‖·‖c-v‖)
            float cosTheta = (distQC_sq + distCV_sq - distQV_sq) / (2.0f * distQC * distCV);
            cosTheta = Math.max(-1.0f, Math.min(1.0f, cosTheta)); // clamp numerical noise
            float angle = (float) Math.acos(cosTheta);

            sampleDistsSq[sampleCount] = distQC_sq;
            sampleAngles[sampleCount] = angle;
            sampleCount++;
        }

        if (sampleCount == 0) {
            Arrays.fill(outLambdaSlices, 0.0f);
            outMinDistSq[clusterIdx] = 0;
            outMaxDistSq[clusterIdx] = 1;
            return;
        }

        // Find min/max distQC_sq for slicing
        float minDist = Float.MAX_VALUE, maxDist = Float.MIN_VALUE;
        for (int i = 0; i < sampleCount; i++) {
            if (sampleDistsSq[i] < minDist) minDist = sampleDistsSq[i];
            if (sampleDistsSq[i] > maxDist) maxDist = sampleDistsSq[i];
        }
        outMinDistSq[clusterIdx] = minDist;
        outMaxDistSq[clusterIdx] = maxDist;

        float sliceLen = (maxDist - minDist) / NUM_SLICES;
        if (sliceLen <= 0) sliceLen = 1.0f;

        // For each slice, find the β-quantile of angles and compute λ = cos(θ_β)
        for (int sl = 0; sl < NUM_SLICES; sl++) {
            float sliceStart = minDist + sl * sliceLen;
            float sliceEnd = sliceStart + sliceLen;

            // Collect angles in this slice
            int sliceCount = 0;
            float[] sliceAngles = new float[sampleCount]; // over-allocated
            for (int i = 0; i < sampleCount; i++) {
                if (sampleDistsSq[i] >= sliceStart && sampleDistsSq[i] < sliceEnd) {
                    sliceAngles[sliceCount++] = sampleAngles[i];
                }
            }

            if (sliceCount == 0) {
                // No samples in this slice — use conservative λ = 0 (triangle inequality)
                outLambdaSlices[sl] = 0.0f;
                continue;
            }

            // Sort angles and find β-quantile
            Arrays.sort(sliceAngles, 0, sliceCount);
            int quantileIdx = Math.max(0, (int) (BETA * sliceCount));
            float thetaMin = sliceAngles[quantileIdx];

            // λ = cos(θ_min): since cos is decreasing, smaller θ → larger λ → tighter bound
            outLambdaSlices[sl] = (float) Math.cos(thetaMin);
        }
    }

    /**
     * Calibrate a single λ for segment-level pruning.
     * Treats the entire segment as one "super-cluster" with the segment centroid.
     */
    private static float calibrateSegmentLambda(
        float[] segmentCentroid,
        float[][] vectors,
        int dimension,
        Random rng
    ) {
        int numVectors = vectors.length;
        int numSamples = Math.min(5000, numVectors * 5);
        float[] angles = new float[numSamples];
        int count = 0;

        for (int s = 0; s < numSamples && count < numSamples; s++) {
            int qOrd = rng.nextInt(numVectors);
            int vOrd = rng.nextInt(numVectors);
            if (qOrd == vOrd) continue;

            float[] q = vectors[qOrd];
            float[] v = vectors[vOrd];

            float distQC_sq = squaredL2(q, segmentCentroid);
            float distCV_sq = squaredL2(segmentCentroid, v);
            float distQV_sq = squaredL2(q, v);

            float distQC = (float) Math.sqrt(distQC_sq);
            float distCV = (float) Math.sqrt(distCV_sq);

            if (distQC < 1e-10f || distCV < 1e-10f) continue;

            float cosTheta = (distQC_sq + distCV_sq - distQV_sq) / (2.0f * distQC * distCV);
            cosTheta = Math.max(-1.0f, Math.min(1.0f, cosTheta));
            angles[count++] = (float) Math.acos(cosTheta);
        }

        if (count == 0) return 0.0f;

        Arrays.sort(angles, 0, count);
        int quantileIdx = Math.max(0, (int) (BETA * count));
        return (float) Math.cos(angles[quantileIdx]);
    }

    // ========== Serialization ==========

    /** File extension for CLIP pruning data. */
    public static final String EXTENSION = "clid";

    /**
     * Write CLIP pruning data to output.
     */
    public void write(IndexOutput output) throws IOException {
        int dimension = segmentCentroid != null ? segmentCentroid.length : 0;

        // Header
        output.writeInt(numCentroids);
        output.writeInt(dimension);

        // Segment centroid
        if (dimension > 0) {
            for (int d = 0; d < dimension; d++) {
                output.writeInt(Float.floatToIntBits(segmentCentroid[d]));
            }
            output.writeInt(Float.floatToIntBits(segmentLambda));
        }

        // Per-cluster data
        for (int c = 0; c < numCentroids; c++) {
            int count = sortedDistances[c].length;
            output.writeInt(count);

            // Sorted distances
            for (int i = 0; i < count; i++) {
                output.writeInt(Float.floatToIntBits(sortedDistances[c][i]));
            }

            // Sort permutation
            for (int i = 0; i < count; i++) {
                output.writeInt(sortPermutation[c][i]);
            }

            // Lambda slices
            for (int sl = 0; sl < NUM_SLICES; sl++) {
                output.writeInt(Float.floatToIntBits(lambdaSlices[c][sl]));
            }

            // Distance bounds for slicing
            output.writeInt(Float.floatToIntBits(minDistSq[c]));
            output.writeInt(Float.floatToIntBits(maxDistSq[c]));
        }
    }

    /**
     * Read CLIP pruning data from input.
     */
    public static ClipPruningData read(IndexInput input) throws IOException {
        int numCentroids = input.readInt();
        int dimension = input.readInt();

        // Segment centroid
        float[] segmentCentroid = null;
        float segmentLambda = 0.0f;
        if (dimension > 0) {
            segmentCentroid = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                segmentCentroid[d] = Float.intBitsToFloat(input.readInt());
            }
            segmentLambda = Float.intBitsToFloat(input.readInt());
        }

        // Per-cluster data
        float[][] sortedDistances = new float[numCentroids][];
        int[][] sortPermutation = new int[numCentroids][];
        float[][] lambdaSlices = new float[numCentroids][NUM_SLICES];
        float[] minDistSq = new float[numCentroids];
        float[] maxDistSq = new float[numCentroids];

        for (int c = 0; c < numCentroids; c++) {
            int count = input.readInt();

            sortedDistances[c] = new float[count];
            for (int i = 0; i < count; i++) {
                sortedDistances[c][i] = Float.intBitsToFloat(input.readInt());
            }

            sortPermutation[c] = new int[count];
            for (int i = 0; i < count; i++) {
                sortPermutation[c][i] = input.readInt();
            }

            for (int sl = 0; sl < NUM_SLICES; sl++) {
                lambdaSlices[c][sl] = Float.intBitsToFloat(input.readInt());
            }

            minDistSq[c] = Float.intBitsToFloat(input.readInt());
            maxDistSq[c] = Float.intBitsToFloat(input.readInt());
        }

        return new ClipPruningData(
            segmentCentroid, segmentLambda, numCentroids,
            sortedDistances, sortPermutation,
            lambdaSlices, minDistSq, maxDistSq
        );
    }

    // ========== Utilities ==========

    private static float squaredL2(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    private static void sortByDistance(float[] distances, int[] permutation, int count) {
        // Pack (distance, index) into longs for primitive sort
        long[] packed = new long[count];
        for (int i = 0; i < count; i++) {
            int bits = Float.floatToIntBits(distances[i]);
            long sortKey = (long) (bits ^ (bits >> 31) | 0x80000000) & 0xFFFFFFFFL;
            packed[i] = (sortKey << 32) | (permutation[i] & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);

        for (int i = 0; i < count; i++) {
            permutation[i] = (int) packed[i];
            int bits = (int) (packed[i] >>> 32);
            bits = bits ^ ((bits >> 31) | 0x80000000);
            distances[i] = Float.intBitsToFloat(bits);
        }
    }
}
