/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.model;

/**
 * Mutable carrier for the graph-derived page-locality permutation produced during a native
 * layer-0 HNSW build and consumed by the locality vectors writer afterwards.
 *
 * <h2>Why this exists</h2>
 * The permutation is produced deep inside the native build strategy
 * ({@code MemOptimizedScalarQuantizedIndexBuildStrategy}), which is the only place that holds the
 * native graph pointer. It is consumed by {@code Faiss1040SQHNSWReorderedWriter} several frames up,
 * <em>after</em> the build returns. Every frame in between
 * ({@code doFlush}/{@code doMergeOneField} → {@code NativeIndexWriter} → {@code NativeIndexBuildStrategy})
 * returns {@code void} and is shared with non-locality writers. Rather than changing all of those
 * signatures to return a value, the writer creates one of these, threads the reference down via
 * {@link BuildIndexParams}, and the strategy writes the result into it. The writer then reads it back
 * in its own flush/merge scope.
 *
 * <h2>Payload</h2>
 * Both payloads are computed at the source (natively, over the in-place layer-0 adjacency) and are
 * compact — the raw {@code 2*N*M} adjacency never crosses the JNI boundary or lands on the Java heap.
 * There are two arrays, and they store <b>different</b> things:
 *
 * <h3>1. {@code physicalOrdinals} — the ordering (a forward permutation)</h3>
 * The <b>forward</b> map {@code physicalOrdinals[originalOrdinal] = physicalPosition}: the array is
 * <b>indexed by the original</b> (insertion-order) ordinal, and each <b>value is the physical</b>
 * (on-disk) slot that vector's record occupies in the reordered store. It is <em>not</em> a list of
 * ordinals — the ordinals are the indices, the physical positions are the values. Length is
 * {@code ntotal} and it is bijective (every original ordinal maps to a distinct physical slot). This
 * matches the reader's {@code ordToPhysicalOrdMap}.
 * <ul>
 *   <li><b>Greedy strict-page layout</b> ({@code buildOrderingOfVectorsUsingIndexStructure}): the
 *       physical-position <b>values are sparse</b> — early-closed pages pad to the next page boundary,
 *       so the max value can exceed {@code ntotal-1} and the writer must zero-fill the padding slots.</li>
 *   <li><b>BFS layout</b> ({@code buildOrderingOfVectorsUsingBFS}): the physical-position <b>values are
 *       dense</b> — a contiguous {@code 0..ntotal-1} with no padding.</li>
 * </ul>
 *
 * <h3>2. {@code hubs} — search entry-point candidates (a list of ordinals)</h3>
 * {@code hubs[rank] = originalOrdinal}: the highest-degree (most connected) node ids, ordered by rank
 * (rank 0 = highest degree). Here the <b>values themselves are original ordinals</b> (unlike
 * {@code physicalOrdinals}). The array is small (capped at ~32); only {@code min(hubs.length, ntotal)}
 * entries are filled and any remaining slots are padded with {@code -1}. These are graph node ids
 * (original-ordinal space) intended as entry points for graph search; to get a hub's on-disk slot,
 * look it up as {@code physicalOrdinals[hub]}.
 *
 * <p>Not thread-safe; a single instance is scoped to one field's flush or merge.
 */
public final class Layer0LocalityOrdering {

    /** Forward permutation: {@code physicalOrdinals[originalOrdinal] = physicalPosition}. Length {@code ntotal}. */
    private int[] physicalOrdinals;
    /** Top-degree hub ordinals (values are original ordinals); {@code -1}-padded; length capped at ~32. */
    private int[] hubs;
    private boolean populated;

    /**
     * Records the computed ordering and hub list. Called once by the native build strategy after the
     * layer-0 graph is built and the ordering is derived.
     *
     * @param physicalOrdinals the forward permutation {@code physicalOrdinals[originalOrdinal] =
     *                         physicalPosition} (values are physical slots, indexed by original ordinal);
     *                         length equals the number of live vectors
     * @param hubs             top-degree hub node ids as search entry-point candidates
     *                         ({@code hubs[rank] = originalOrdinal}, values are original ordinals),
     *                         {@code -1}-padded; length capped at ~32
     * @throws IllegalStateException    if already populated
     * @throws NullPointerException     if {@code physicalOrdinals} is null
     */
    public void populate(final int[] physicalOrdinals, final int[] hubs) {
        if (populated) {
            throw new IllegalStateException("Layer0LocalityOrdering has already been populated");
        }
        if (physicalOrdinals == null) {
            throw new NullPointerException("physicalOrdinals must not be null");
        }
        this.physicalOrdinals = physicalOrdinals;
        this.hubs = hubs;
        this.populated = true;
    }

    /**
     * @return true once the strategy has written the permutation; false if the build was skipped,
     *         aborted, or has not run yet. Consumers must fall back to a default ordering when false.
     */
    public boolean isPopulated() {
        return populated;
    }

    /**
     * @return the forward permutation {@code physicalOrdinals[originalOrdinal] = physicalPosition}
     *         (indexed by original ordinal, values are physical slots); length {@code ntotal}
     * @throws IllegalStateException if not yet populated (guard with {@link #isPopulated()})
     */
    public int[] getPhysicalOrdinals() {
        if (!populated) {
            throw new IllegalStateException("Layer0LocalityOrdering has not been populated");
        }
        return physicalOrdinals;
    }

    /**
     * @return the hub entry-point candidates {@code hubs[rank] = originalOrdinal} (values are original
     *         ordinals, highest-degree first), {@code -1}-padded; length capped at ~32
     * @throws IllegalStateException if not yet populated (guard with {@link #isPopulated()})
     */
    public int[] getHubs() {
        if (populated == false) {
            throw new IllegalStateException("Layer0LocalityOrdering hubs are not populated");
        }
        return hubs;
    }

}
