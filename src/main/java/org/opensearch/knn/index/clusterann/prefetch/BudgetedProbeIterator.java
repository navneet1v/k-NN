/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.search.KnnCollector;

import java.io.IOException;

/**
 * Stops yielding probes once the scoring budget is exhausted.
 *
 * <p>Two phases:
 * <ol>
 *   <li>Visit until vector budget met AND competitive scores found</li>
 *   <li>If filtered, continue until at least k docs scored</li>
 * </ol>
 */
public final class BudgetedProbeIterator implements ProbeIterator {

    private final ProbeIterator delegate;
    private final KnnCollector collector;
    private final long budget;
    private final int k;
    private long docsExpected;
    private long docsScored;

    public BudgetedProbeIterator(ProbeIterator delegate, KnnCollector collector, int numVectors, int k) {
        this.delegate = delegate;
        this.collector = collector;
        this.k = k;
        // Budget: log²(n) × k vectors, ×2 for SOAR overlap
        double logN = Math.log10(Math.max(numVectors, 10));
        this.budget = Math.max(k * 4L, (long) (2.0 * logN * logN * k));
    }

    @Override
    public boolean hasNext() {
        if (!delegate.hasNext()) return false;

        // Always visit if we haven't scored k docs yet
        if (docsScored < k) return true;

        // Phase 1: keep going until budget met and competitive scores found
        if (docsExpected < budget) return true;
        if (collector.minCompetitiveSimilarity() == Float.NEGATIVE_INFINITY) return true;

        return false;
    }

    @Override
    public ProbedCentroid next() throws IOException {
        return delegate.next();
    }

    /** Called by visitor after processing a centroid. */
    public void recordVisit(int expectedCount, int scoredCount) {
        docsExpected += expectedCount;
        docsScored += scoredCount;
    }
}
