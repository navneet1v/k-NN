/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.algorithm.RandomRotation;

import java.io.IOException;

/**
 * The clusters of one field — the structure a {@code ClusterSearcher} searches over (the ClusterANN
 * analogue of an {@code HnswGraph}: a persistent, query-independent handle, so it lives on the reader,
 * built once per field and reused). Random access by ordinal: the planner sweeps geometry over all
 * clusters to rank them, and a query's {@link #scan} touches only the {@code nprobe} probed ones;
 * covering {@code 0..numClusters()-1} is the whole field (an exhaustive/exact scan).
 *
 * <p>Immutable and thread-safe: it holds a {@code .clap} slice bounded to this field's posting region
 * (only ever cloned/sliced from, never read with a moving cursor), the immutable {@link
 * ClusterANNFieldState}, the stateless {@link ClusterFactory} (impl recipe + dispatch), and a
 * <em>base</em> transformed-centroid reader ({@code .clac}). The transformed centroids are field-level
 * data (their offset is fixed layout), but reading them needs a mutable cursor, so the base is never
 * read directly — a fresh single-use cursor is cloned from it per read. This keeps "how this field's
 * centroids are read" here rather than in the caller, while staying shareable across concurrent searches.
 *
 * <p>Because it is shared, nothing here may vary per query. A query's own state — the projected and
 * quantized query, and anything derived from it — lives in the {@link ClusterScan} that {@link #scan}
 * hands out. This is only reachable through that: {@link #prefetch} is the one thing a caller can do
 * without a query, because a hint needs no query to be useful.
 */
public final class Clusters {

    private final IndexInput fieldPostings;                    // .clap sliced to this field's posting region
    private final IndexInput rotationInput;                    // .clar (shared; cloned per rotation read)
    private final ClusterANNFieldState fieldState;
    private final ClusterFactory factory;
    private final CentroidVectorValues transformedCentroidsBase; // .clac ADC-space; copy() per centroid read

    public Clusters(
        IndexInput postingsInput,
        IndexInput centroidsInput,
        IndexInput rotationInput,
        ClusterANNFieldState fieldState,
        ClusterFactory factory
    ) throws IOException {
        // Slice .clap to this field's posting region using the length stored in .clam (postings are
        // contiguous from postingsOffset). Cluster offsets rebase to slice-relative per posting. The slice
        // is only cloned/sliced from (in ClusterScan#scorer), so a single shared instance is thread-safe.
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
     * One query's pass over this field. Reads nothing itself, but it is where per-query work that many
     * postings share gets done once — projecting and quantizing the query against a reference centroid.
     * Not thread-safe: one per query per field, per search thread.
     */
    public ClusterScan scan(ScanParams params) {
        return factory.scan(params, fieldPostings.clone(), transformedCentroidsBase);
    }

    /**
     * Number of clusters in this field. Scanning {@code 0..numClusters()-1} covers every cluster — the basis
     * for an exhaustive/exact scan (recall ceiling, or future full-precision rescoring) as opposed to the
     * {@code nprobe}-limited search path.
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
     * What the cluster at the given centroid ordinal <em>is</em> — size, posting extent, quantization
     * reference. <b>Reads nothing:</b> every field comes from this field's {@code .clam} state, which is why
     * the walk can decide whether a cluster is worth touching, or hint it, without any I/O.
     */
    public Cluster cluster(int ordinal) {
        return new Cluster(
            ordinal,
            fieldState.centroidDocCounts[ordinal],
            ordinal, // reference == own centroid, until several clusters share one
            fieldState.centroidOffsets[ordinal] - fieldState.postingsOffset, // rebase to this field's slice
            fieldState.postingSizes[ordinal]
        );
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
     * <p>This is the batch form because ordering only exists across a set: given the clusters it is about to
     * scan, this can put them in a sensible sequence, which per-cluster calls cannot. A future step is to
     * merge adjacent ranges into single, larger hints — worth doing only if probe sets turn out to be
     * file-local enough to have adjacent ranges at all (see §5.4).
     *
     * <p>Takes {@link Cluster}s rather than ordinals because a cluster already carries its own extent, which
     * is all a hint needs. No query, and no {@link ClusterScan}: warming a cluster the filter is about to
     * skip should not require setting one up.
     */
    public void prefetch(Cluster[] clusters, boolean partial) throws IOException {
        // Insertion sort on a copy — n is the prefetch window (single digits), and this keeps the caller's
        // array (the probe order) untouched.
        Cluster[] byOffset = clusters.clone();
        for (int i = 1; i < byOffset.length; i++) {
            Cluster c = byOffset[i];
            int j = i - 1;
            while (j >= 0 && byOffset[j].postingOffset() > c.postingOffset()) {
                byOffset[j + 1] = byOffset[j];
                j--;
            }
            byOffset[j + 1] = c;
        }
        // One clone for the batch: the hints are offset-addressed, so they move no cursor.
        IndexInput cursor = fieldPostings.clone();
        for (Cluster c : byOffset) {
            if (c.size() == 0) {
                continue;
            }
            long length = partial ? Math.min(factory.guaranteedBytes(c.size()), c.postingBytes()) : c.postingBytes();
            if (length <= 0 || c.postingOffset() < 0 || c.postingOffset() + length > cursor.length()) {
                continue; // never let a hint reach past the field's slice
            }
            cursor.prefetch(c.postingOffset(), length);
        }
    }
}
