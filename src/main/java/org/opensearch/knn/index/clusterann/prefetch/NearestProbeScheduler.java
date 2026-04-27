/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.search.KnnCollector;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.codec.ClusterANNCentroidScanner;
import org.opensearch.knn.index.clusterann.codec.ClusterANNFieldState;

import java.io.IOException;
import java.util.Arrays;

/**
 * Core search stage: selects nearest centroids and scans them.
 *
 * <p>Computes distances to all centroids, determines adaptive nprobe from
 * the distance distribution, then scans each centroid via the scanner.
 */
public final class NearestProbeScheduler implements ProbeScheduler {

    private final ProbeTarget[] probes;
    private final int nprobe;
    private final ClusterANNCentroidScanner scanner;

    public NearestProbeScheduler(float[] query, ClusterANNFieldState fieldState, int k, ClusterANNCentroidScanner scanner) {
        this.scanner = scanner;
        float[][] centroids = fieldState.centroids;
        long[] offsets = fieldState.centroidOffsets;
        int[] postingSizes = fieldState.postingSizes;
        int numCentroids = fieldState.numCentroids;
        DistanceMetric metric = fieldState.metric;

        // Compute distances and sort by distance using primitive long[] packing
        float[] dists = new float[numCentroids];
        long[] packed = new long[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            dists[c] = metric.distance(query, centroids[c]);
            // Pack: sortable float bits in upper 32, index in lower 32
            int floatBits = Float.floatToIntBits(dists[c]);
            // Flip sign bit so IEEE 754 sorts correctly as unsigned long
            long sortKey = floatBits ^ (floatBits >> 31) | 0x80000000;
            packed[c] = (sortKey << 32) | (c & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);

        // Extract sorted indices
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

    int nprobe() {
        return nprobe;
    }

    private static int calculateNprobe(float[] sortedDists, int numCentroids, int k) {
        if (numCentroids <= 10) return numCentroids;
        int minProbe = Math.max(1, (int) Math.sqrt(k));
        int maxProbe = Math.min(numCentroids, Math.max(10, numCentroids / 4));
        float nearestDist = sortedDists[0];
        float cutoff = Math.max(nearestDist * 4f, 1e-6f);
        int nprobe = minProbe;
        for (int i = minProbe; i < maxProbe; i++) {
            if (sortedDists[i] > cutoff) break;
            nprobe = i + 1;
        }
        return nprobe;
    }
}
