/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.util.Bits;

import java.io.IOException;

/**
 * One IVF cluster — the data structure a query iterates over (the ClusterANN analogue of an HNSW
 * node / a graph you can walk). Everything scoped to the cluster is hidden behind it: its centroid
 * geometry, membership, on-disk postings and quantization.
 *
 * <p>A cluster hands out a {@link PostingScorer} over its own vectors, scored for a query — you
 * "iterate the postings, scoring while iterating." How the vectors are stored (block-columnar vs
 * row-major) and scored (bulk vs per-vector) lives entirely inside that iterator, so swapping the
 * storage family is invisible to the {@code ClusterSearcher} that drives it. The contract is purely
 * behavioral — there is deliberately no {@code centroid()} accessor; the centroid is used internally
 * to build the iterator, not exposed.
 *
 * <p>A cluster owns how its own vectors are scored — for quantized storage that is inseparable from how
 * they are laid out. What a query may vary arrives as data in {@link ScanParams}, so the searcher can
 * influence scoring without learning anything about storage.
 *
 * <p>Obtaining an instance reads <em>nothing</em> (see {@code Clusters}) — every read, including this
 * cluster's own centroid, is deferred to {@link #scorer}. That is what makes {@link #prefetch} and the
 * {@code size() == 0} / filter-skip checks free: a cluster the walk decides not to scan costs no I/O at
 * all. One instance per query; not thread-safe.
 */
public interface Cluster {

    /** This cluster's centroid ordinal within the field (identity; useful for bookkeeping/logging). */
    int ordinal();

    /** Number of vectors in this cluster (primary + SOAR). */
    int size();

    /**
     * Hint this cluster's posting into the buffer pool, before a scan reaches it. Obtaining a cluster does
     * no reading, so this is the cheap way to warm one the walk is only *approaching* — it needs nothing
     * but the cluster's own extent and layout, which is why it belongs here rather than in a component
     * that would have to be told how postings are laid out.
     *
     * @param partial hint only the prefix a scan is (near-)certain to read, letting the scan itself stream
     *     the remainder, so a posting that terminates early or is heavily pruned is not over-fetched.
     *     Otherwise hint the whole posting — worth it when a near-full scan is expected, since one large
     *     sequential hint beats many small ones.
     */
    void prefetch(boolean partial) throws IOException;

    /**
     * A scorer over this cluster's postings — a cursor that scores as it advances.
     *
     * @param params the prepared query plus the per-query scoring knobs this cluster should honour
     *     (see {@link ScanParams}).
     * @param wanted read-only ord-space membership test — {@code wanted.get(ord)} answers "may I score
     *     this vector?", or {@code null} to score every vector. It folds together the doc filter and
     *     SOAR dedup, both already resolved into ordinal space by the searcher; the cluster only reads
     *     it and never learns about docs or mutates the set. Applied over the cluster's
     *     {@code pos → ord} mapping <em>before</em> the (expensive) distance computation, so a rejected
     *     vector costs no dot product and a block with no wanted vector skips its codes entirely.
     */
    PostingScorer scorer(ScanParams params, Bits wanted) throws IOException;
}
