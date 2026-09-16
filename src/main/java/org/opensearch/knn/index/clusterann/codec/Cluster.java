/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * What one IVF cluster <em>is</em>: its identity, how many vectors it holds, where its posting sits, and
 * which centroid its codes were quantized against. All of it recorded by the writer and fixed for the
 * segment's life, so this answers questions about a cluster without a query and without touching a file.
 *
 * <p>Deliberately holds no centroid <em>vector</em> — only the row it lives in. A vector is bytes in
 * {@code .clac} that a mutable cursor has to reach, and keeping every cluster's centroid resident would put
 * the field's whole centroid set on heap, which is what the off-heap layout exists to avoid. So the address
 * is structural and free; the read belongs to a {@link ClusterScan}, which owns cursors.
 *
 * <p>That split is the whole division of labour: this says what a cluster is, a {@link ClusterScan} says how
 * one query traverses it. Immutable and free to construct, so it may be shared, cached, or rebuilt per probe
 * without consequence.
 *
 * @param ordinal identity, and equally the row of this cluster's <b>own</b> centroid — the one that decides
 *     membership, the posting's sort order, and the geometric pruning bound
 * @param size vectors in the posting, primary and SOAR spills counted alike
 * @param referenceRow row of the centroid this cluster's codes were <b>quantized against</b>. A distinct
 *     concept from {@link #ordinal()} even though it currently equals it: several clusters may come to share
 *     one quantization reference, in which case the query is quantized once for all of them, while
 *     {@link #ordinal()} stays per-cluster because geometry is
 * @param postingOffset start of the posting, relative to this field's {@code .clap} slice
 * @param postingBytes length of the posting
 */
public record Cluster(int ordinal, int size, int referenceRow, long postingOffset, long postingBytes) {

    /** Whether this cluster's codes were quantized against its own centroid, rather than a shared one. */
    boolean isOwnReference() {
        return referenceRow == ordinal;
    }
}
