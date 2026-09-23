/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * A corpus to measure a format against: vectors to index, held-out queries, and each query's exact neighbours as row
 * numbers in {@link #train()} rather than doc ids, which merges do not preserve.
 */
public interface Corpus {
    String name();

    int dimension();

    VectorSimilarityFunction similarity();

    int size();

    float[] vector(int row);

    int queries();

    float[] query(int index);

    int[] neighbours(int index, int count);
}
