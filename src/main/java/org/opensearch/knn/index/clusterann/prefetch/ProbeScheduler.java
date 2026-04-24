/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.search.KnnCollector;

import java.io.IOException;

/**
 * A composable search stage that processes centroid probes.
 *
 * <p>Unlike iterator+visitor patterns, a ProbeScheduler is a self-contained
 * processing stage: it decides WHICH centroids to probe, HOW to order them,
 * and WHEN to stop. Stages compose via delegation, forming a pipeline.
 *
 * <p>Each stage calls {@code execute()} which drives the entire search
 * for that stage's responsibility. This is push-based (stage drives iteration)
 * rather than pull-based (caller pulls from iterator).
 */
public interface ProbeScheduler {

    /**
     * Execute this stage's search logic, collecting results into the collector.
     *
     * @return total number of vectors scored across all probed centroids
     */
    int execute(KnnCollector collector) throws IOException;
}
