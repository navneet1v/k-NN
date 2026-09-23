/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.suite;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.List;
import java.util.Map;

/** One measurement's parameters, read from a suite file: which corpus, how to index it, how to search it, which codec. */
public record Scenario(String name, CorpusSpec corpus, IndexSpec index, SearchSpec search, CodecSpec codec) {
    public record CorpusSpec(String type, int size, int queries, int dimension, int clusters,

        double spread, int depth, long seed, VectorSimilarityFunction similarity, String file) {
    }

    public record IndexSpec(int docsPerSegment, List<String> expectFiles, Integer forceMerge) {
    }

    public record SearchSpec(int k, int queries, int repeat, int warmup, int threads, double scoreTolerance, double scoreBias) {
    }

    public record CodecSpec(String name, Map<String, Object> params) {
    }

    @Override
    public String toString() {
        return name;
    }
}
