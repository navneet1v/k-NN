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

    /** POC: tiered centroid routing (2-bit Hamming shortlist -> int8 rerank). -Dclusterann.route.tiered=true */
    private static final boolean TIERED_ROUTE = Boolean.getBoolean("clusterann.route.tiered");
    /** Coarse shortlist size for tiered routing before int8 rerank. */
    private static final int ROUTE_COARSE_M = Integer.getInteger("clusterann.route.coarseM", 256);

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

        // Compute distances directly from flat buffer — no per-centroid copy
        float[] dists = new float[numCentroids];
        if (TIERED_ROUTE) {
            // POC tiered centroid routing: 2-bit thermometer Hamming shortlist -> int8 rerank.
            // Codes computed on the fly from full-precision centroids (few centroids, no format change).
            computeTieredDists(query, flatCentroids, numCentroids, dimension, k, dists);
        } else if (metric == DistanceMetric.L2 && fieldState.centroidNorms != null) {
            float queryNormSq = VectorUtil.dotProduct(query, query);
            for (int c = 0; c < numCentroids; c++) {
                int offset = c * dimension;
                float dot = dotProductWithOffset(query, flatCentroids, offset, dimension);
                dists[c] = queryNormSq + fieldState.centroidNorms[c] - 2f * dot;
            }
        } else {
            for (int c = 0; c < numCentroids; c++) {
                int offset = c * dimension;
                float dot = dotProductWithOffset(query, flatCentroids, offset, dimension);
                // IP metric: distance = -dot (lower is more similar)
                dists[c] = -dot;
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

        // Compute distances directly from flat buffer — no per-centroid copy
        float[] dists = new float[numCentroids];
        if (metric == DistanceMetric.L2 && fieldState.centroidNorms != null) {
            float queryNormSq = VectorUtil.dotProduct(query, query);
            for (int c = 0; c < numCentroids; c++) {
                int offset = c * dimension;
                float dot = dotProductWithOffset(query, flatCentroids, offset, dimension);
                dists[c] = queryNormSq + fieldState.centroidNorms[c] - 2f * dot;
            }
        } else {
            for (int c = 0; c < numCentroids; c++) {
                int offset = c * dimension;
                float dot = dotProductWithOffset(query, flatCentroids, offset, dimension);
                dists[c] = -dot;
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

    /**
     * Compute dot product between query[0..dim) and flat[offset..offset+dim).
     * Avoids System.arraycopy into a temporary buffer.
     */
    private static float dotProductWithOffset(float[] query, float[] flat, int offset, int dim) {
        float sum = 0f;
        for (int d = 0; d < dim; d++) {
            sum += query[d] * flat[offset + d];
        }
        return sum;
    }

    /**
     * POC tiered centroid routing. Rotate query + centroids (Hadamard), 2-bit thermometer Hamming
     * shortlist top-{@link #ROUTE_COARSE_M}, then int8 rerank the shortlist. Fills {@code dists}
     * with negative int8 score for shortlisted centroids (lower = nearer) and +INF for the rest,
     * so the existing sort/nprobe logic downstream works unchanged.
     */
    private static void computeTieredDists(float[] query, float[] flatCentroids, int numCentroids, int dim, int k, float[] dists) {
        final int OFFSET = 128;
        java.util.Arrays.fill(dists, Float.MAX_VALUE);

        org.opensearch.knn.index.clusterann.algorithm.HadamardRotation rot =
            org.opensearch.knn.index.clusterann.algorithm.HadamardRotation.create(dim);

        // Rotate + code the query (2-bit coarse + int8).
        float[] rq = new float[dim];
        rot.transform(query, rq);
        int cb = org.opensearch.knn.index.clusterann.codec.Nitrox2.bytesPerVector(dim);
        byte[] qCoarse = new byte[cb];
        org.opensearch.knn.index.clusterann.codec.Nitrox2.packPlanes(rq, dim, qCoarse, 0);
        byte[] qI8 = new byte[dim];
        int qSum = int8Encode(rq, dim, qI8);
        float qScale = int8Scale(rq, dim);

        // Coarse pass: Hamming(query, each centroid) -> keep top-M.
        int[] ham = new int[numCentroids];
        float[] cIn = new float[dim];
        float[] cRot = new float[dim];
        byte[] cCoarse = new byte[cb];
        for (int c = 0; c < numCentroids; c++) {
            System.arraycopy(flatCentroids, c * dim, cIn, 0, dim);
            rot.transform(cIn, cRot);
            org.opensearch.knn.index.clusterann.codec.Nitrox2.packPlanes(cRot, dim, cCoarse, 0);
            ham[c] = org.opensearch.knn.index.clusterann.codec.Nitrox2.hamming(qCoarse, 0, cCoarse, 0, cb);
        }
        int M = Math.min(Math.max(ROUTE_COARSE_M, k), numCentroids);
        int[] hamCopy = ham.clone();
        java.util.Arrays.sort(hamCopy);
        int hThresh = hamCopy[M - 1];

        // int8 rerank the shortlist.
        byte[] cI8 = new byte[dim];
        int kept = 0;
        for (int c = 0; c < numCentroids && kept < M; c++) {
            if (ham[c] <= hThresh) {
                System.arraycopy(flatCentroids, c * dim, cIn, 0, dim);
                rot.transform(cIn, cRot);
                int cSum = int8Encode(cRot, dim, cI8);
                float cScale = int8Scale(cRot, dim);
                long unsignedDot = 0;
                for (int d = 0; d < dim; d++) unsignedDot += (long) (qI8[d] & 0xFF) * (cI8[d] & 0xFF);
                long signedDot = unsignedDot - ((long) OFFSET * qSum + 16384L * dim) - (long) OFFSET * cSum;
                double dot = (double) signedDot * qScale * cScale;
                dists[c] = -(float) dot; // lower = nearer (IP)
                kept++;
            }
        }
    }

    private static float int8Scale(float[] v, int dim) {
        float maxAbs = 0f;
        for (int d = 0; d < dim; d++) { float a = Math.abs(v[d]); if (a > maxAbs) maxAbs = a; }
        return maxAbs == 0f ? 1f : maxAbs / 127f;
    }

    private static int int8Encode(float[] v, int dim, byte[] out) {
        float maxAbs = 0f;
        for (int d = 0; d < dim; d++) { float a = Math.abs(v[d]); if (a > maxAbs) maxAbs = a; }
        if (maxAbs == 0f) { java.util.Arrays.fill(out, (byte) 128); return 0; }
        float inv = 127f / maxAbs;
        int sum = 0;
        for (int d = 0; d < dim; d++) {
            int q = Math.round(v[d] * inv);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            sum += q;
            out[d] = (byte) (q + 128);
        }
        return sum;
    }
}
