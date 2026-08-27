/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOSupplier;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.algorithm.RandomRotation;

import java.io.IOException;

/**
 * The clusters of one field — the structure a {@code ClusterSearcher} searches over (the ClusterANN
 * analogue of an {@code HnswGraph}: a persistent, query-independent handle, so it lives on the reader,
 * built once per field and reused). Random access by ordinal: the planner sweeps geometry over all
 * clusters to rank them, the search touches only the {@code nprobe} probed ones via {@link #get};
 * iterating {@code 0..numClusters()-1} covers the whole field (an exhaustive/exact scan).
 *
 * <p>Immutable and thread-safe: it holds a {@code .clap} slice bounded to this field's posting region
 * (only ever cloned/sliced from, never read with a moving cursor), the immutable {@link
 * ClusterANNFieldState}, the stateless {@link ClusterFactory} (impl recipe + dispatch), and a
 * <em>base</em> transformed-centroid reader ({@code .clac}). The transformed centroids are field-level
 * data (their offset is fixed layout), but reading them needs a mutable cursor, so the base is never
 * read directly — {@link #get} clones a fresh single-use cursor from it per call. This keeps "how this
 * field's centroids are read" here rather than in the caller, while staying shareable across concurrent
 * searches.
 *
 * <p>Building a cluster is cheap (one centroid seek); the posting is parsed lazily inside
 * {@link Cluster#scorer}.
 */
public final class Clusters {

    private final IndexInput fieldPostings;                    // .clap sliced to this field's posting region
    private final IndexInput rotationInput;                    // .clar (shared; cloned per rotation read)
    private final ClusterANNFieldState fieldState;
    private final ClusterFactory factory;
    private final CentroidVectorValues transformedCentroidsBase; // .clac ADC-space; copy() per get()

    public Clusters(
        IndexInput postingsInput,
        IndexInput centroidsInput,
        IndexInput rotationInput,
        ClusterANNFieldState fieldState,
        ClusterFactory factory
    ) throws IOException {
        // Slice .clap to this field's posting region using the length stored in .clam (postings are
        // contiguous from postingsOffset). Cluster offsets rebase to slice-relative in get(). The slice
        // is only cloned/sliced from (in Cluster#scorer), so a single shared instance is thread-safe.
        this.fieldPostings = postingsInput.slice(
            "clap-field-" + fieldState.fieldNumber, fieldState.postingsOffset, fieldState.postingsLength);
        this.rotationInput = rotationInput;
        this.fieldState = fieldState;
        this.factory = factory;
        // Transformed-centroid region follows the raw region; the raw region carries a trailing norm
        // only for L2. This base is only copy()'d, never read directly, so sharing it is thread-safe.
        boolean rawHasNorm = fieldState.metric == DistanceMetric.L2;
        long transformedOffset = CentroidVectorValues.transformedOffset(
            fieldState.clacCentroidOffset, fieldState.numCentroids, fieldState.dimension, rawHasNorm);
        this.transformedCentroidsBase = new CentroidVectorValues(
            centroidsInput, transformedOffset, fieldState.numCentroids, fieldState.dimension, true);
    }

    /**
     * Number of clusters in this field. Iterating {@code 0..numClusters()-1} with {@link #get} visits
     * every cluster — the basis for an exhaustive/exact scan (recall ceiling, or future full-precision
     * rescoring) as opposed to the {@code nprobe}-limited search path.
     */
    public int numClusters() {
        return fieldState.numCentroids;
    }

    /**
     * Project a query into the space this field's stored vectors live in — the orthonormal rotation the
     * doc codes were written under — returning it unchanged for a field with no rotation (non-L2, or
     * disabled). Done once per query, not per cluster. Streamed from the {@code .clar} mmap; only a
     * permutation and one row buffer land on heap, never the matrix.
     *
     * <p>Unconditional for the scan path: anything that reads the stored codes must meet them in the
     * space they were written in. A future full-precision pass scores the flat vectors by ordinal instead
     * of scanning postings, so it simply never comes through here.
     */
    public float[] prepareQuery(float[] query) throws IOException {
        if (fieldState.rotationOffset < 0) {
            return query;
        }
        float[] rotated = new float[fieldState.dimension];
        IndexInput rotation = rotationInput.clone();
        rotation.seek(fieldState.rotationOffset);
        RandomRotation.transform(rotation, query, rotated);
        return rotated;
    }

    /** Number of vectors in this field, across all clusters (primary + SOAR counted once each). */
    public int numVectors() {
        return fieldState.numVectors;
    }

    /**
     * Number of vectors in the cluster with the given centroid ordinal (primary + SOAR). Available
     * without building the cluster, so the walk can decide whether a cluster is worth touching at all.
     */
    public int clusterSize(int ordinal) {
        return fieldState.centroidDocCounts[ordinal];
    }

    /**
     * Hint the posting of centroid {@code ordinal} into the buffer pool, before a scan reaches it. See
     * {@link Cluster#prefetch} for what {@code partial} covers.
     *
     * <p>Safe to call on this shared structure: like {@link #get}, it reads nothing through any shared
     * cursor — the cluster it goes through owns a private clone, so the hint is issued on that.
     */
    public void prefetch(int ordinal, boolean partial) throws IOException {
        get(ordinal).prefetch(partial);
    }

    /**
     * Hint several postings at once, issuing the hints in <b>file order</b> rather than the order given.
     *
     * <p>Reordering is safe here in a way it is not for the scan: a hint carries no ordering semantics, so
     * sorting them cannot affect which clusters the walk visits or in what sequence — and therefore cannot
     * cost recall. What it buys is that requests arrive at the layer below in increasing offset order,
     * where readahead and request coalescing can act on them. The walk still visits clusters closest-first,
     * which pruning and the competitive threshold depend on.
     *
     * <p>This is the batch form because ordering only exists across a set: given the ordinals it is about
     * to scan, this can put them in a sensible sequence, which per-ordinal calls cannot. A future step is
     * to merge adjacent ranges into single, larger hints — worth doing only if probe sets turn out to be
     * file-local enough to have adjacent ranges at all (see §5.4).
     */
    public void prefetch(int[] ordinals, boolean partial) throws IOException {
        // Insertion sort on a copy — n is the prefetch window (single digits), and this keeps the caller's
        // array (the probe order) untouched.
        int[] byOffset = ordinals.clone();
        for (int i = 1; i < byOffset.length; i++) {
            int ord = byOffset[i];
            long offset = fieldState.centroidOffsets[ord];
            int j = i - 1;
            while (j >= 0 && fieldState.centroidOffsets[byOffset[j]] > offset) {
                byOffset[j + 1] = byOffset[j];
                j--;
            }
            byOffset[j + 1] = ord;
        }
        for (int ord : byOffset) {
            prefetch(ord, partial);
        }
    }

    /**
     * The cluster with the given centroid ordinal. <b>Reads nothing</b> — it only looks up this field's
     * layout facts and hands the cluster private cursors ({@code .clap} and {@code .clac} clones) to read
     * through when it needs to. Both the posting and the cluster's own centroid are read lazily inside
     * {@link Cluster#scorer}.
     *
     * <p>That laziness is deliberate: it makes {@link Cluster#prefetch} and the walk's skip checks free, so
     * a cluster that is merely warmed ahead — or fetched and then skipped by the filter — costs no I/O.
     * The cursors must be private because concurrent searches share this structure and both an
     * {@link IndexInput} and a centroid cursor carry a moving pointer (and, for the latter, a reused
     * buffer); cloning allocates objects but reads nothing.
     */
    public Cluster get(int ordinal) throws IOException {
        int count = fieldState.centroidDocCounts[ordinal];
        // Rebase the absolute .clap offset to this field's slice (see ctor).
        long relativeOffset = fieldState.centroidOffsets[ordinal] - fieldState.postingsOffset;
        long postingBytes = fieldState.postingSizes[ordinal];
        // The cluster gets a capability to read its OWN centroid, not a view over the field's centroids —
        // it must not be able to address another cluster's geometry, or know centroids live in a file.
        // The cursor is minted inside the supplier, so a cluster that is only prefetched allocates none.
        IOSupplier<Centroid> centroid = () -> {
            CentroidVectorValues cursor = (CentroidVectorValues) transformedCentroidsBase.copy();
            float[] vector = cursor.vectorValue(ordinal); // reused buffer; read once, consumed immediately
            return new Centroid(vector, cursor.norm());
        };
        return factory.create(ordinal, count, fieldPostings.clone(), relativeOffset, postingBytes, centroid);
    }
}
