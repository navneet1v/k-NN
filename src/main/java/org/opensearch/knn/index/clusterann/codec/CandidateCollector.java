/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import java.util.Arrays;

/**
 * Bounded min-heap for ADC candidate collection. Keeps top {@code maxCandidates}
 * by score, evicting the worst when full.
 */
public final class CandidateCollector {

    private final int[] ordinals;
    private final float[] scores;
    private final int maxCandidates;
    private int count;

    CandidateCollector(int maxCandidates) {
        this.maxCandidates = maxCandidates;
        this.ordinals = new int[maxCandidates];
        this.scores = new float[maxCandidates];
        this.count = 0;
    }

    void add(int ordinal, float score) {
        if (count < maxCandidates) {
            ordinals[count] = ordinal;
            scores[count] = score;
            count++;
            if (count == maxCandidates) {
                buildMinHeap();
            }
        } else if (score > scores[0]) {
            ordinals[0] = ordinal;
            scores[0] = score;
            siftDown(0);
        }
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

    /** Get indices of top-n candidates by score (sort-based, O(n log n)). */
    int[] topN(int n) {
        n = Math.min(n, count);
        // Pack (score, index) into long[] for primitive sort — descending by score
        long[] packed = new long[count];
        for (int i = 0; i < count; i++) {
            // Negate float bits for descending sort
            int floatBits = Float.floatToIntBits(scores[i]);
            long sortKey = ~((long) (floatBits ^ ((floatBits >> 31) | 0x80000000)) & 0xFFFFFFFFL);
            packed[i] = (sortKey << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed, 0, count);
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = (int) packed[i];
        }
        return result;
    }

    private void buildMinHeap() {
        for (int i = count / 2 - 1; i >= 0; i--) {
            siftDown(i);
        }
    }

    private void siftDown(int i) {
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;
            if (left < count && scores[left] < scores[smallest]) smallest = left;
            if (right < count && scores[right] < scores[smallest]) smallest = right;
            if (smallest == i) break;
            swap(i, smallest);
            i = smallest;
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
