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
    BLOCKS_FETCHED("blocks_fetched"),

    /**
     * Blocks a pruner cleared to read. Should track {@link #BLOCKS_FETCHED} exactly — a cleared block is always
     * fetched — so a gap between the two means a block was read without being tested, or tested twice.
     */
    PRUNER_SCORE("pruner_score"),

    /**
     * Blocks a pruner stepped over, which cost a test and no I/O. Pure CLIP when the query carries no filter,
     * since the filter contributes {@link org.opensearch.knn.clusterann.format.block.BlockPostingsPruner#NONE}
     * then; under a filter the two are summed here and cannot be told apart.
     */
    PRUNER_SKIP("pruner_skip"),

    /**
     * Postings cut short, one per cluster scan that ended because no later block could compete. Only ever CLIP:
     * a filter cannot rule out the rest of a posting, so it never terminates.
     */
    PRUNER_TERMINATE("pruner_terminate"),

    /**
     * Blocks never reached because a posting terminated — the tail that cost neither a test nor a read, and
     * usually where most of the saving is.
     *
     * <p>Together these account for every block of every scanned cluster:
     * {@code blocks_fetched + pruner_skip + blocks_terminated = total blocks probed}. A violated sum is a
     * bookkeeping bug, which is the point of carrying the third term rather than deriving it.
     */
    BLOCKS_TERMINATED("blocks_terminated"),

    /**
     * Blocks whose corrective terms were read but whose codes were not, because no vector in the block could
     * reach the threshold on the scorer's own ceiling. Counted against {@link #BLOCKS_FETCHED}, which still
     * includes them: the prefix was read either way, and the prefix is a few percent of a block. So this is the
     * share of fetched blocks that cost their corrections and nothing more.
     *
     * <p>Unlike the pruner counters this measures a bound in estimate space rather than a geometric one, so it
     * is the one saving here that needs no quantization margin to be sound.
     */
    CODE_READS_SKIPPED("code_reads_skipped"),

    /**
     * Positions in those blocks that would have been scored had the codes been read — what
     * {@link #CODE_READS_SKIPPED} actually bought, in the unit the cost is paid in.
     *
     * <p>Carried so a run checks itself instead of being differenced against another build:
     * {@code vectors_scored + vectors_skipped} is every accepted position in every fetched block, which is
     * exactly what a scan with no gate would report as {@code vectors_scored}. A violated sum means the gate is
     * dropping something it is not counting, and no second run is needed to see it.
     */
    VECTORS_SKIPPED("vectors_skipped");

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
