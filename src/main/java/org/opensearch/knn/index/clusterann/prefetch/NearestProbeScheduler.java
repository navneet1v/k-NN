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
        int numCentroids = fieldState.numCentroids;
        DistanceMetric metric = fieldState.metric;

        float[] dists = new float[numCentroids];
        Integer[] indices = new Integer[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            dists[c] = metric.distance(query, centroids[c]);
            indices[c] = c;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(dists[a], dists[b]));

        float[] sortedDists = new float[numCentroids];
        for (int i = 0; i < numCentroids; i++) {
            sortedDists[i] = dists[indices[i]];
        }
        this.nprobe = calculateNprobe(sortedDists, numCentroids, k);

        this.probes = new ProbeTarget[nprobe];
        for (int i = 0; i < nprobe; i++) {
            int c = indices[i];
            long offset = offsets[c];
            long nextOffset = findNextOffset(offsets, c);
            probes[i] = new ProbeTarget(c, offset, nextOffset - offset, dists[c]);
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

    /** Exposed for wrapping stages that need the probe list. */
    ProbeTarget[] probes() {
        return probes;
    }

    int nprobe() {
        return nprobe;
    }

    private static long findNextOffset(long[] offsets, int centroidIdx) {
        long thisOffset = offsets[centroidIdx];
        long minNext = Long.MAX_VALUE;
        for (long offset : offsets) {
            if (offset > thisOffset && offset < minNext) {
                minNext = offset;
            }
        }
        return minNext == Long.MAX_VALUE ? thisOffset + 1024 * 1024 : minNext;
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
