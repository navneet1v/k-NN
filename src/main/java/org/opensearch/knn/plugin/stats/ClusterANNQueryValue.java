/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.concurrent.atomic.AtomicLong;

/**
 * What ClusterANN queries have cost this node: how much of the index they looked at, how many distances they
 * computed, and how much they read to do it.
 *
 * <p>Counted per segment scan rather than per query, because a query fans out over segments and it is the
 * per-segment work that the numbers are about. {@link #SEGMENT_SCANS} is the denominator that turns the totals
 * back into an average scan, which is what makes them comparable across runs of different sizes.
 *
 * <p>Node-level totals since startup, in the same shape as {@link KNNGraphValue} — monotonic, never reset, so a
 * benchmark takes two readings and subtracts.
 */
public enum ClusterANNQueryValue {

    /** Segment scans completed. The denominator for every other value here. */
    SEGMENT_SCANS("segment_scans"),

    /** Clusters the planner named, empty ones included — what {@code nprobe} actually worked out to. */
    CLUSTERS_PROBED("clusters_probed"),

    /** Clusters actually walked. Short of {@link #CLUSTERS_PROBED} by the probes that landed on empty clusters. */
    CLUSTERS_SCANNED("clusters_scanned"),

    /**
     * Distances computed. Not the hits returned: a posting is scored and may still lose to the threshold, so this
     * is the CPU the scan spent, where the collector's visited count is what survived.
     */
    VECTORS_SCORED("vectors_scored"),

    /**
     * Stored blocks read and decoded. The scan's I/O, and the number that says whether block skipping and pruning
     * are earning their keep — against {@code vectors_scored / block_size} as the floor.
     */
    BLOCKS_FETCHED("blocks_fetched");

    private final String name;
    private final AtomicLong value;

    ClusterANNQueryValue(final String name) {
        this.name = name;
        this.value = new AtomicLong(0);
    }

    /**
     * Get name of value
     *
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * Get the value
     *
     * @return value
     */
    public Long getValue() {
        return value.get();
    }

    /** Increment by one. */
    public void increment() {
        value.getAndIncrement();
    }

    /**
     * Increment by a specified amount.
     *
     * @param delta the amount to add
     */
    public void incrementBy(final long delta) {
        value.getAndAdd(delta);
    }

    /**
     * Set the value. Only for tests, which need a known starting point.
     *
     * @param newValue the value to set
     */
    public void set(final long newValue) {
        value.set(newValue);
    }
}
