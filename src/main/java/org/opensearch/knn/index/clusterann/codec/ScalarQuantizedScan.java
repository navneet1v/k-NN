/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link ClusterScan} over block-columnar, scalar-quantized postings in {@code .clap} — this family's
 * scan, end to end. Per posting it parses the {@link PostingHeader}, slices the quantized payload,
 * and assembles the block reader, scorer and pruners behind a {@link BlockPostingScorer}.
 *
 * <p>Owning all of that here is deliberate. Scoring quantized codes is inseparable from how they are laid
 * out — that fusion is the point of the format — so there is no useful seam between "the storage" and "the
 * scoring of it" to abstract over. Nothing about blocks, readers or corrections escapes to the searcher,
 * which only ever sees a {@link PostingScorer}.
 *
 * <p><b>The centroid has two jobs, and a {@link Cluster} names both.</b> {@link Cluster#ordinal()} is the
 * <em>partition</em> centroid — membership, the posting's sort key, the geometric (CLIP) bound.
 * {@link Cluster#referenceRow()} is the <em>quantization reference</em>, the point residuals were measured
 * from. They are equal today, so each posting quantizes the query for itself and {@link #cachedContext} never
 * hits; once several clusters share a reference they arrive in runs and it does. Nothing here changes for
 * that to happen — the writer decides it and {@code referenceRow} carries it.
 *
 * <p>One instance per query per field; not thread-safe. Construction reads nothing.
 */
final class ScalarQuantizedScan implements ClusterScan {

    private final ScanParams params;
    private final IndexInput clap;                     // private clone of this field's .clap slice
    private final CentroidVectorValues centroidsBase;  // copy() per read; never read directly
    private final ScalarQuantizedLayout layout;

    // The query quantized against the last reference used. Probes arrive closest-first, so clusters sharing a
    // reference arrive in runs and one entry catches them; while every cluster is its own reference this never
    // hits and costs one int compare.
    private int cachedRow = -1;
    private AdcQueryContext cachedContext;

    ScalarQuantizedScan(
        ScanParams params,
        IndexInput clap,
        CentroidVectorValues centroidsBase,
        ScalarQuantizedLayout layout
    ) {
        this.params = params;
        this.clap = clap;
        this.centroidsBase = centroidsBase;
        this.layout = layout;
    }

    @Override
    public PostingScorer scorer(Cluster cluster, Bits wanted) throws IOException {
        int count = cluster.size();
        PostingHeader header = PostingHeader.parse(clap, cluster.postingOffset(), count);

        // Bound slice over just this posting's quantized payload.
        IndexInput payload = clap.slice("clap-quantized", header.payloadOffset(), (long) count * layout.recordBytes());

        // This cluster's own geometry — read now that scoring needs it, which is what lets an unscanned or
        // merely prefetched cluster cost no I/O. Needed here for the geometric bound.
        Centroid partition = centroid(cluster.ordinal());

        AdcQueryContext ctx = contextFor(cluster, partition);
        ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(payload, count, layout.packedBytes());
        ADCBlockScorer scorer = new ADCBlockScorer(reader, ctx, layout);
        PostingPruner[] pruners = pruners(header, scorer, partition);

        return new BlockPostingScorer(scorer, pruners, header.ordinals(), wanted);
    }

    /**
     * The query quantized against {@code cluster}'s reference, reusing the last one when the reference
     * repeats. A cluster referencing its own centroid reuses the geometry read just above rather than reading
     * the same row twice.
     */
    private AdcQueryContext contextFor(Cluster cluster, Centroid partition) throws IOException {
        int row = cluster.referenceRow();
        if (row != cachedRow) {
            Centroid reference = cluster.isOwnReference() ? partition : centroid(row);
            cachedContext = AdcQueryContext.quantize(params.query(), reference, layout.quantizer(), params.queryBits());
            cachedRow = row;
        }
        return cachedContext;
    }

    /** Read one centroid's geometry through a private cursor cloned from the shared base. */
    private Centroid centroid(int ordinal) throws IOException {
        CentroidVectorValues cursor = (CentroidVectorValues) centroidsBase.copy();
        float[] vector = cursor.vectorValue(ordinal); // reused buffer; read once, consumed immediately
        return new Centroid(vector, cursor.norm());
    }

    /**
     * Which block-skip bounds are sound for this scan. A bound may only prune against the collector's
     * threshold if it lives in the same space as the scores feeding that threshold, and those scores are
     * quantization-approximate — so the choice depends on the metric <em>and</em> on how coarse this scan's
     * scores are, which is why it is made here (where {@code queryBits} is known) rather than at segment
     * open.
     *
     * <ul>
     *   <li><b>L2</b> — the geometric (CLIP) bound on exact distance, against the <em>partition</em>
     *       centroid. Safe because it only fires deep in the tail, where its margin dwarfs the quantization
     *       error.</li>
     *   <li><b>Inner product</b> — the corrections bound, computed in the same approximate space as the
     *       scores. The exact geometric ceiling is <em>not</em> used: compared against a threshold inflated
     *       by quantization over-estimation it prunes true neighbours, and that inflation grows as
     *       {@code queryBits}/{@code docBits} shrink.</li>
     * </ul>
     */
    private PostingPruner[] pruners(PostingHeader header, ADCBlockScorer scorer, Centroid partition) {
        List<PostingPruner> pruners = new ArrayList<>(2);
        if (layout.sim() == VectorSimilarityFunction.EUCLIDEAN) {
            pruners.add(new ClipPostingPruner(params.query(), partition.vector(), header.distancesToCentroid(), layout.sim()));
        } else {
            pruners.add(scorer.correctionsBound());
        }
        return pruners.toArray(new PostingPruner[0]);
    }
}
