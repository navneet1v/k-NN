/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.opensearch.knn.index.clusterann.prefetch.ProbeTarget;
import org.apache.lucene.search.KnnCollector;

import java.io.IOException;

/**
 * Visits one centroid's posting list: reads, filters, scores, collects.
 * Stateful per-query, reusable across centroids.
 */
public interface CentroidScanner {
    /**
     * Seek to centroid and read posting metadata (docIds, ordinals).
     * Returns expected doc count (primary + SOAR).
     */
    int prepare(ProbeTarget centroid) throws IOException;

    /**
     * Score and collect results for the current centroid.
     * Returns actual scored count.
     */
    int scan(KnnCollector collector) throws IOException;
}
