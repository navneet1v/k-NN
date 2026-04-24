/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.clusterann;

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

    /** Get indices of top-n candidates by score (partial select, descending). */
    int[] topN(int n) {
        n = Math.min(n, count);
        int[] idx = new int[count];
        for (int i = 0; i < count; i++)
            idx[i] = i;
        for (int i = 0; i < n; i++) {
            int bestIdx = i;
            for (int j = i + 1; j < count; j++) {
                if (scores[idx[j]] > scores[idx[bestIdx]]) bestIdx = j;
            }
            int tmp = idx[i];
            idx[i] = idx[bestIdx];
            idx[bestIdx] = tmp;
        }
        return java.util.Arrays.copyOf(idx, n);
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
