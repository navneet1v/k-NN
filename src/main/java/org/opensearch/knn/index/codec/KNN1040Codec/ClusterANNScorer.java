/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import java.io.IOException;

/**
 * Scoring interface for ClusterANN search. Abstracts the scoring strategy so the reader
 * doesn't need to know whether it's doing exact scoring, ADC scoring, or two-phase scoring.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link ExactClusterANNScorer} — full-precision scoring via Lucene's RandomVectorScorer</li>
 *   <li>{@link TwoPhaseClusterANNScorer} — ADC first pass + exact rescore (future)</li>
 * </ul>
 */
interface ClusterANNScorer {

    /**
     * Prefetch data for a batch of ordinals to optimize I/O.
     * Implementations may use this to issue prefetch hints for disk-based data.
     *
     * @param ordinals array of vector ordinals
     * @param count    number of valid entries in the array
     * @throws IOException if an I/O error occurs
     */
    void prefetch(int[] ordinals, int count) throws IOException;

    /**
     * Score a single vector ordinal against the query.
     *
     * @param ordinal the vector ordinal to score
     * @return similarity score (higher = more similar, Lucene convention)
     * @throws IOException if an I/O error occurs
     */
    float score(int ordinal) throws IOException;

    /**
     * Map a vector ordinal to its document ID.
     *
     * @param ordinal the vector ordinal
     * @return the document ID
     */
    int ordToDoc(int ordinal);

    /**
     * Flush any buffered ADC candidates and perform rescoring.
     * Called after all posting lists have been scanned.
     * For exact scorers this is a no-op.
     *
     * @param collector callback to collect final scored results
     * @throws IOException if an I/O error occurs
     */
    void finish(ResultCollector collector) throws IOException;

    /**
     * Callback for collecting scored results during the finish phase.
     */
    @FunctionalInterface
    interface ResultCollector {
        void collect(int docId, float score) throws IOException;
    }
}
