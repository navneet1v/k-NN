/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import java.util.Arrays;

/**
 * Amortized O(1) top-N candidate collection inspired by ScaNN's TopNAmortizedConstant.
 *
 * <p>Instead of maintaining a heap (O(log n) per insert), uses a buffer that
 * accumulates candidates. When the buffer fills, partitions to keep only the
 * top half by score. Insertions are O(1) amortized with O(n) partition cost
 * spread across n insertions.
 *
 * <p>After collection, call {@link #topN(int)} to extract the best candidates.
 */
public final class CandidateCollector {

    private int[] ordinals;
    private float[] scores;
    private final int capacity;
    private final int targetSize;
    private int count;
    private float threshold;

    CandidateCollector(int maxCandidates) {
        this.targetSize = maxCandidates;
        this.capacity = maxCandidates * 2;
        this.ordinals = new int[capacity];
        this.scores = new float[capacity];
        this.count = 0;
        this.threshold = Float.NEGATIVE_INFINITY;
    }

    void add(int ordinal, float score) {
        if (score <= threshold) return;

        ordinals[count] = ordinal;
        scores[count] = score;
        count++;

        if (count == capacity) {
            partition();
        }
    }

    /** Current threshold — scores at or below this are rejected. */
    float threshold() {
        return threshold;
    }

    int count() {
        return count;
    }

    int ordinal(int i) {
        return ordinals[i];
    }

    float score(int i) {
        return scores[i];
    }

    /** Get indices of top-n candidates by score. */
    int[] topN(int n) {
        n = Math.min(n, count);
        if (count > targetSize) {
            partialSort(targetSize);
        }
        // Sort top targetSize by score descending, return top n
        long[] packed = new long[Math.min(count, targetSize)];
        for (int i = 0; i < packed.length; i++) {
            int floatBits = Float.floatToIntBits(scores[i]);
            long sortKey = ~((long) (floatBits ^ ((floatBits >> 31) | 0x80000000)) & 0xFFFFFFFFL);
            packed[i] = (sortKey << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = (int) packed[i];
        }
        return result;
    }

    /**
     * Partition: keep top targetSize elements, discard the rest.
     * Uses quickselect to find the targetSize-th element, then compact.
     */
    private void partition() {
        partialSort(targetSize);
        count = targetSize;
        threshold = scores[count - 1];
    }

    /**
     * Quickselect-based partial sort: rearranges so that elements [0..k-1]
     * all have scores >= elements [k..count-1].
     */
    private void partialSort(int k) {
        if (k >= count) return;
        int lo = 0, hi = count - 1;
        while (lo < hi) {
            // Median-of-three pivot
            int mid = lo + (hi - lo) / 2;
            if (scores[mid] < scores[lo]) swap(lo, mid);
            if (scores[hi] < scores[lo]) swap(lo, hi);
            if (scores[mid] < scores[hi]) swap(mid, hi);
            float pivot = scores[hi];

            int i = lo, j = hi - 1;
            while (i <= j) {
                while (i <= j && scores[i] > pivot)
                    i++;
                while (i <= j && scores[j] <= pivot)
                    j--;
                if (i < j) {
                    swap(i, j);
                    i++;
                    j--;
                }
            }
            swap(i, hi);

            if (i == k) break;
            else if (i < k) lo = i + 1;
            else hi = i - 1;
        }
    }

    private void swap(int a, int b) {
        int tmpOrd = ordinals[a];
        ordinals[a] = ordinals[b];
        ordinals[b] = tmpOrd;
        float tmpScore = scores[a];
        scores[a] = scores[b];
        scores[b] = tmpScore;
    }
}
