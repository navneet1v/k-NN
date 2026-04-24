/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.opensearch.knn.index.clusterann.codec.ClusterANNFieldState;
import org.opensearch.knn.index.clusterann.*;
import org.opensearch.knn.index.clusterann.codec.ClusterANNFieldState;
import org.opensearch.knn.index.clusterann.*;
import org.opensearch.knn.index.clusterann.DistanceMetric;

import java.util.Arrays;

/**
 * Yields the nprobe nearest centroids with adaptive cutoff.
 * Computes distances once, sorts, and determines nprobe from the distance distribution.
 */
public final class NearestProbeScheduler implements ProbeScheduler {

    private final ProbeTarget[] probes;
    private final int nprobe;
    private int cursor;

    public NearestProbeScheduler(float[] query, ClusterANNFieldState fieldState, int k) {
        float[][] centroids = fieldState.centroids;
        long[] offsets = fieldState.centroidOffsets;
        int numCentroids = fieldState.numCentroids;
        DistanceMetric metric = fieldState.metric;

        // Compute distances
        float[] dists = new float[numCentroids];
        Integer[] indices = new Integer[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            dists[c] = metric.distance(query, centroids[c]);
            indices[c] = c;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(dists[a], dists[b]));

        // Adaptive nprobe
        float[] sortedDists = new float[numCentroids];
        for (int i = 0; i < numCentroids; i++) {
            sortedDists[i] = dists[indices[i]];
        }
        this.nprobe = calculateNprobe(sortedDists, numCentroids, k);

        // Build probes with posting size from offset table
        this.probes = new ProbeTarget[nprobe];
        for (int i = 0; i < nprobe; i++) {
            int c = indices[i];
            long offset = offsets[c];
            // Posting size: distance to next centroid's offset or estimate
            long nextOffset = findNextOffset(offsets, c, fieldState);
            long postingBytes = nextOffset - offset;
            probes[i] = new ProbeTarget(c, offset, postingBytes, dists[c]);
        }
        this.cursor = 0;
    }

    int nprobe() {
        return nprobe;
    }

    @Override
    public boolean hasNext() {
        return cursor < nprobe;
    }

    @Override
    public ProbeTarget next() {
        return probes[cursor++];
    }

    private static long findNextOffset(long[] offsets, int centroidIdx, ClusterANNFieldState fieldState) {
        // Find the smallest offset that is greater than offsets[centroidIdx]
        long thisOffset = offsets[centroidIdx];
        long minNext = Long.MAX_VALUE;
        for (int i = 0; i < offsets.length; i++) {
            if (offsets[i] > thisOffset && offsets[i] < minNext) {
                minNext = offsets[i];
            }
        }
        // If this is the last centroid by offset, estimate from postingsOffset context
        // Use a reasonable upper bound
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
