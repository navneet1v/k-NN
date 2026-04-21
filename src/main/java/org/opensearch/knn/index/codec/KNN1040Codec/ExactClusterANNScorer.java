/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * Exact scorer that uses Lucene's {@link RandomVectorScorer} with batch scoring via
 * {@code bulkScore()} for prefetch + native SIMD acceleration.
 *
 * <p>When the underlying scorer is a {@code PrefetchableRandomVectorScorer}, calling
 * {@code bulkScore()} triggers:
 * <ol>
 *   <li>Prefetch: groups ordinals by disk locality, issues prefetch hints</li>
 *   <li>Native SIMD scoring: delegates to C++ AVX2/AVX512/NEON code via JNI</li>
 * </ol>
 *
 * <p>This is the primary scoring path for IVF search — posting list ordinals are
 * collected into a batch, then scored in one bulk call for maximum throughput.
 */
final class ExactClusterANNScorer implements ClusterANNScorer {

    private final RandomVectorScorer scorer;

    // Reusable batch buffers (avoid allocation per posting list)
    private int[] batchOrdinals;
    private float[] batchScores;

    ExactClusterANNScorer(RandomVectorScorer scorer) {
        this.scorer = scorer;
        this.batchOrdinals = new int[256];
        this.batchScores = new float[256];
    }

    @Override
    public void prefetch(int[] ordinals, int count) throws IOException {
        // No-op: prefetch is handled inside bulkScore by PrefetchableRandomVectorScorer
    }

    @Override
    public float score(int ordinal) throws IOException {
        return scorer.score(ordinal);
    }

    @Override
    public int ordToDoc(int ordinal) {
        return scorer.ordToDoc(ordinal);
    }

    /**
     * Score a batch of ordinals using bulkScore (prefetch + SIMD).
     *
     * @param ordinals array of vector ordinals to score
     * @param count    number of valid entries
     * @param scores   output array for scores (must be >= count)
     * @throws IOException if an I/O error occurs
     */
    void bulkScore(int[] ordinals, float[] scores, int count) throws IOException {
        scorer.bulkScore(ordinals, scores, count);
    }

    /**
     * Ensure batch buffers are large enough for the given size.
     */
    void ensureBatchCapacity(int size) {
        if (batchOrdinals.length < size) {
            batchOrdinals = new int[size];
            batchScores = new float[size];
        }
    }

    int[] batchOrdinals() { return batchOrdinals; }
    float[] batchScores() { return batchScores; }

    @Override
    public void finish(ResultCollector collector) throws IOException {
        // No-op: exact scorer collects results inline
    }
}
