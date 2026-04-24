/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import java.io.IOException;

/**
 * Skips centroids that are unlikely to contain matching docs under a selective filter.
 *
 * <p>Uses centroid doc counts from .clam to estimate expected matches per centroid.
 * When filter selectivity is low (< 10%), centroids with very few docs are skipped
 * because the probability of any doc passing the filter is near zero.
 *
 * <p>Zero extra disk storage — uses doc counts already in .clam.
 */
public final class FilterAwareProbeScheduler implements ProbeScheduler {

    private static final float SELECTIVITY_THRESHOLD = 0.10f;

    private final ProbeScheduler delegate;
    private final int[] centroidDocCounts;
    private final float filterSelectivity;
    private final boolean active;
    private ProbeTarget buffered;

    /**
     * @param delegate          underlying probe iterator
     * @param centroidDocCounts docs per centroid from .clam
     * @param numVectors        total vectors in segment
     * @param filterCost        approximate number of docs passing the filter
     */
    public FilterAwareProbeScheduler(ProbeScheduler delegate, int[] centroidDocCounts, int numVectors, long filterCost) {
        this.delegate = delegate;
        this.centroidDocCounts = centroidDocCounts;
        this.filterSelectivity = numVectors > 0 ? (float) filterCost / numVectors : 1.0f;
        // Only activate for selective filters — unfiltered queries skip this entirely
        this.active = filterSelectivity < SELECTIVITY_THRESHOLD && filterSelectivity > 0;
    }

    @Override
    public boolean hasNext() {
        if (buffered != null) return true;
        while (delegate.hasNext()) {
            try {
                ProbeTarget probe = delegate.next();
                if (shouldVisit(probe)) {
                    buffered = probe;
                    return true;
                }
            } catch (IOException e) {
                return false;
            }
        }
        return false;
    }

    @Override
    public ProbeTarget next() throws IOException {
        if (buffered != null) {
            ProbeTarget result = buffered;
            buffered = null;
            return result;
        }
        return delegate.next();
    }

    private boolean shouldVisit(ProbeTarget probe) {
        if (!active) return true;
        int docCount = centroidDocCounts[probe.centroidIdx()];
        // Expected matching docs = docCount × selectivity
        // Skip if expected < 0.5 (likely zero matches)
        return docCount * filterSelectivity >= 0.5f;
    }
}
