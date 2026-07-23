/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.codec.ClusterANNCentroidScanner;
import org.opensearch.knn.index.clusterann.codec.ClusterANNFieldState;
import org.opensearch.knn.index.clusterann.codec.OffHeapCentroids;

import java.io.IOException;
import java.util.Arrays;

/**
 * Core search stage: selects nearest centroids and scans them.
 *
 * <p>Computes distances to all centroids, determines adaptive nprobe from
 * the distance distribution, then scans each centroid via the scanner.
 *
 * <p>Optimizations inspired by ScaNN:
 * <ul>
 *   <li>L2 via dot product + precomputed norms: ||q-c||² = ||q||² + ||c||² - 2·q·c</li>
 *   <li>Primitive long[] sort (no Integer boxing)</li>
 *   <li>Precomputed posting sizes for exact prefetch</li>
 * </ul>
 */
public final class NearestProbeScheduler implements ProbeScheduler {

    private final ProbeTarget[] probes;
    private final int nprobe;
    private final ClusterANNCentroidScanner scanner;

    public NearestProbeScheduler(float[] query, ClusterANNFieldState fieldState, int k,
                                ClusterANNCentroidScanner scanner, OffHeapCentroids.Reader centroidReader) throws IOException {
        this.scanner = scanner;
        long[] offsets = fieldState.centroidOffsets;
        int[] postingSizes = fieldState.postingSizes;
        int numCentroids = fieldState.numCentroids;
        int dimension = fieldState.dimension;
        DistanceMetric metric = fieldState.metric;

        // Read all centroids from off-heap (.clac mmap) into temp buffer
        float[] flatCentroids = new float[numCentroids * dimension];
        centroidReader.readAllCentroids(flatCentroids);

        // Compute distances using temp centroid slices
        float[] dists = new float[numCentroids];
        float[] centroidBuf = new float[dimension];
        if (metric == DistanceMetric.L2 && fieldState.centroidNorms != null) {
            float queryNormSq = VectorUtil.dotProduct(query, query);
            for (int c = 0; c < numCentroids; c++) {
                System.arraycopy(flatCentroids, c * dimension, centroidBuf, 0, dimension);
                float dot = VectorUtil.dotProduct(query, centroidBuf);
                dists[c] = queryNormSq + fieldState.centroidNorms[c] - 2f * dot;
            }
        } else {
            for (int c = 0; c < numCentroids; c++) {
                System.arraycopy(flatCentroids, c * dimension, centroidBuf, 0, dimension);
                dists[c] = metric.distance(query, centroidBuf);
            }
        }

        // Sort by distance using primitive long[] packing
        long[] packed = new long[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            int floatBits = Float.floatToIntBits(dists[c]);
            long sortKey = floatBits ^ (floatBits >> 31) | 0x80000000;
            packed[c] = (sortKey << 32) | (c & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);

        int[] sortedIndices = new int[numCentroids];
        float[] sortedDists = new float[numCentroids];
        for (int i = 0; i < numCentroids; i++) {
            int c = (int) packed[i];
            sortedIndices[i] = c;
            sortedDists[i] = dists[c];
        }

        this.nprobe = calculateNprobe(sortedDists, numCentroids, k);

        this.probes = new ProbeTarget[nprobe];
        for (int i = 0; i < nprobe; i++) {
            int c = sortedIndices[i];
            this.probes[i] = new ProbeTarget(c, offsets[c], postingSizes[c], dists[c]);
        }
    }

    /**
     * Filter-aware constructor: ranks centroids by distance / matchCount.
     * Centroids with zero matches are skipped entirely.
     * Centroids with more filter matches are prioritized (probed earlier).
     */
    public NearestProbeScheduler(float[] query, ClusterANNFieldState fieldState, int k,
                                 ClusterANNCentroidScanner scanner,
                                 FixedBitSet acceptCentroids, int[] matchCounts,
                                 OffHeapCentroids.Reader centroidReader) throws IOException {
        this.scanner = scanner;
        long[] offsets = fieldState.centroidOffsets;
        int[] postingSizes = fieldState.postingSizes;
        int numCentroids = fieldState.numCentroids;
        int dimension = fieldState.dimension;
        DistanceMetric metric = fieldState.metric;

        // Read centroids from off-heap
        float[] flatCentroids = new float[numCentroids * dimension];
        centroidReader.readAllCentroids(flatCentroids);

        // Compute distances
        float[] dists = new float[numCentroids];
        float[] centroidBuf = new float[dimension];
        if (metric == DistanceMetric.L2 && fieldState.centroidNorms != null) {
            float queryNormSq = VectorUtil.dotProduct(query, query);
            for (int c = 0; c < numCentroids; c++) {
                System.arraycopy(flatCentroids, c * dimension, centroidBuf, 0, dimension);
                float dot = VectorUtil.dotProduct(query, centroidBuf);
                dists[c] = queryNormSq + fieldState.centroidNorms[c] - 2f * dot;
            }
        } else {
            for (int c = 0; c < numCentroids; c++) {
                System.arraycopy(flatCentroids, c * dimension, centroidBuf, 0, dimension);
                dists[c] = metric.distance(query, centroidBuf);
            }
        }

        // Count valid centroids (those with filter matches)
        int validCount = acceptCentroids.cardinality();
        if (validCount == 0) {
            this.nprobe = 0;
            this.probes = new ProbeTarget[0];
            return;
        }

        // Rank by adjustedScore = dist / log2(1 + matchCount)
        // This prioritizes clusters that are both close AND have many matching docs
        long[] packed = new long[validCount];
        int[] validCentroids = new int[validCount];
        int vi = 0;
        for (int c = acceptCentroids.nextSetBit(0); c != -1; c = acceptCentroids.nextSetBit(c + 1)) {
            validCentroids[vi] = c;
            float adjustedDist = dists[c] / (float) Math.log1p(matchCounts[c]);
            int floatBits = Float.floatToIntBits(adjustedDist);
            long sortKey = floatBits ^ (floatBits >> 31) | 0x80000000;
            packed[vi] = (sortKey << 32) | (c & 0xFFFFFFFFL);
            vi++;
        }
        Arrays.sort(packed);

        // Use adaptive nprobe on filtered set, but ensure we probe enough to get k results
        float[] sortedDists = new float[validCount];
        int[] sortedIndices = new int[validCount];
        for (int i = 0; i < validCount; i++) {
            int c = (int) packed[i];
            sortedIndices[i] = c;
            sortedDists[i] = dists[c];
        }

        // For filtered search: probe ALL matching centroids.
        // The filter already bounds the work — if only 20 centroids have matches, probe all 20.
        // This matches the "keep going until k results" approach without artificial caps.
        this.nprobe = validCount;

        this.probes = new ProbeTarget[nprobe];
        for (int i = 0; i < nprobe; i++) {
            int c = sortedIndices[i];
            this.probes[i] = new ProbeTarget(c, offsets[c], postingSizes[c], dists[c]);
        }
    }

    @Override
    public int execute(KnnCollector collector) throws IOException {
        int totalScored = 0;
        for (int i = 0; i < nprobe; i++) {
            scanner.prepare(probes[i]);
            totalScored += scanner.scan(collector);
            if (collector.earlyTerminated()) break;
        }
        return totalScored;
    }

    ProbeTarget[] probes() {
        return probes;
    }

    public int nprobe() {
        return nprobe;
    }

    private static volatile int NPROBE_MULTIPLIER = Integer.getInteger("clusterann.nprobe.multiplier", 2);

    public static void setNprobeMultiplier(int m) { NPROBE_MULTIPLIER = m; }
    public static int getNprobeMultiplier() { return NPROBE_MULTIPLIER; }

    private static int calculateNprobe(float[] sortedDists, int numCentroids, int k) {
        if (numCentroids <= 10) return numCentroids;

        int maxNprobe = Math.min(NPROBE_MULTIPLIER * (int) Math.sqrt(numCentroids), numCentroids);
        int minNprobe = Math.max(10, (int) Math.sqrt(numCentroids));

        // Adaptive: use the "knee" in the distance curve
        float closestDist = sortedDists[0];
        float range = sortedDists[maxNprobe - 1] - closestDist;
        if (range <= 0) return maxNprobe;

        float avgStep = range / maxNprobe;
        int adaptiveNprobe = maxNprobe;
        for (int i = minNprobe; i < maxNprobe - 1; i++) {
            float step = sortedDists[i + 1] - sortedDists[i];
            if (step > avgStep * 3.0f) {
                adaptiveNprobe = i + 1;
                break;
            }
        }

        return Math.max(minNprobe, adaptiveNprobe);
    }
}
