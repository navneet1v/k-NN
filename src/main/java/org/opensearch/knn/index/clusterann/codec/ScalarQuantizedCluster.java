/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * A {@link Cluster} whose posting is a block-columnar, scalar-quantized run in {@code .clap} — this
 * family's cluster, end to end. It parses the posting's {@link PostingHeader}, slices the quantized
 * payload, quantizes the query against its own centroid, and assembles the block reader, scorer, and
 * pruners behind a {@link BlockPostingScorer}.
 *
 * <p>Owning all of that here is deliberate. Scoring quantized codes is inseparable from how they are
 * laid out — that fusion is the point of the format — so there is no useful seam between "the storage"
 * and "the scoring of it" to abstract over. What a query <em>can</em> vary is carried in
 * {@link ScanParams} as data, and honoured here; nothing about blocks, readers, or corrections escapes
 * to the searcher, which only ever sees a {@link PostingScorer}.
 *
 * <p>Construction is cheap (it holds the recipe); all posting I/O happens inside {@link #scorer}.
 * One instance per query; not thread-safe.
 */
final class ScalarQuantizedCluster implements Cluster {

    private static final int CORRECTION_BYTES = 4 * Integer.BYTES; // lower, upper, add, sum

    private final int ordinal;
    private final int count;
    private final IndexInput clap;              // private clone of the field's .clap slice
    private final long postingOffset;           // this posting's start within the slice
    private final long postingBytes;            // this posting's byte length (from .clam)
    private final IOSupplier<Centroid> centroid; // this cluster's own geometry, read on demand

    // Field-level layout facts (from .clam): fixed for the segment, stamped in at construction.
    private final int dimension;
    private final ScalarBitEncoding encoding;
    private final VectorSimilarityFunction sim;
    private final OptimizedScalarQuantizer quantizer;
    private final int packedBytes;
    private final long recordBytes; // per-vector payload stride: corrections + packed codes

    ScalarQuantizedCluster(
        int ordinal,
        int count,
        IndexInput clap,
        long postingOffset,
        long postingBytes,
        IOSupplier<Centroid> centroid,
        int dimension,
        ScalarBitEncoding encoding,
        VectorSimilarityFunction sim,
        OptimizedScalarQuantizer quantizer
    ) {
        this.ordinal = ordinal;
        this.count = count;
        this.clap = clap;
        this.postingOffset = postingOffset;
        this.postingBytes = postingBytes;
        this.centroid = centroid;
        this.dimension = dimension;
        this.encoding = encoding;
        this.sim = sim;
        this.quantizer = quantizer;
        this.packedBytes = encoding.docPackedBytes(dimension);
        this.recordBytes = recordBytes(encoding, dimension);
    }

    /** Bytes one vector occupies in the payload: its corrective terms plus its packed codes. */
    private static long recordBytes(ScalarBitEncoding encoding, int dimension) {
        return CORRECTION_BYTES + encoding.docPackedBytes(dimension);
    }

    @Override
    public void prefetch(boolean partial) throws IOException {
        if (count == 0) {
            return;
        }
        long length = partial ? Math.min(guaranteedBytes(), postingBytes) : postingBytes;
        if (length <= 0 || postingOffset < 0 || postingOffset + length > clap.length()) {
            return; // never let a hint reach past the field's slice
        }
        clap.prefetch(postingOffset, length);
    }

    /**
     * Bytes of this posting a scan is (near-)certain to read: the metadata header — always parsed in full —
     * plus the first block. Everything after that is streamed by the block reader's own prefetch as the
     * scan advances, so a posting that terminates early or is mostly pruned is never over-fetched.
     */
    private long guaranteedBytes() {
        return PostingHeader.byteLength(count) + (long) Math.min(BLOCK_SIZE, count) * recordBytes;
    }

    @Override
    public int ordinal() {
        return ordinal;
    }

    @Override
    public int size() {
        return count;
    }

    @Override
    public PostingScorer scorer(ScanParams params, Bits wanted) throws IOException {
        PostingHeader header = PostingHeader.parse(clap, postingOffset, count);

        // Bound slice over just this posting's quantized payload.
        IndexInput payload = clap.slice("clap-quantized", header.payloadOffset(), count * recordBytes);

        // Read this cluster's own geometry, now that scoring actually needs it — deferring it is what lets
        // an unscanned or merely prefetched cluster cost no I/O. Read once; nothing below retains the array.
        Centroid c = centroid.get();
        float[] transformedCentroid = c.vector();
        float centroidNormSq = c.normSq();

        // Quantize the query against this cluster's centroid once, at the width the query asked for;
        // shared by the scorer and the corrections pruner so neither re-quantizes.
        AdcQueryContext ctx = AdcQueryContext.quantize(
            params.query(), transformedCentroid, centroidNormSq, quantizer, dimension, sim, params.queryBits());

        ScalarQuantizedBlockReader reader = new ScalarQuantizedBlockReader(payload, count, packedBytes);
        // The scorer pairs the field's doc width with this query's width — that pair picks the dot kernel
        // and is what makes the scan asymmetric (ADC) or symmetric (SDC).
        ScalarQuantizedBlockScorer scorer = new ADCBlockScorer(reader, ctx, encoding, sim, dimension, params.queryBits());
        PostingPruner[] pruners = pruners(params, header, ctx, reader, transformedCentroid);

        return new BlockPostingScorer(reader, scorer, pruners, header.ordinals(), wanted);
    }

    /**
     * Which block-skip bounds are sound for this scan. A bound may only prune against the collector's
     * threshold if it lives in the same space as the scores feeding that threshold, and those scores are
     * quantization-approximate — so the choice depends on the metric <em>and</em> on how coarse this
     * scan's scores are, which is why it is made here (where {@code queryBits} is known) rather than at
     * segment open.
     *
     * <ul>
     *   <li><b>L2</b> — the geometric (CLIP) bound on exact distance. Safe because it only fires deep in
     *       the tail, where its margin dwarfs the quantization error.</li>
     *   <li><b>Inner product</b> — the corrections bound, which is computed in the same approximate space
     *       as the scores. The exact geometric ceiling is <em>not</em> used: compared against a threshold
     *       inflated by quantization over-estimation it prunes true neighbours, and that inflation grows
     *       as {@code queryBits}/{@code docBits} shrink.</li>
     * </ul>
     */
    private PostingPruner[] pruners(
        ScanParams params,
        PostingHeader header,
        AdcQueryContext ctx,
        ScalarQuantizedBlockReader reader,
        float[] transformedCentroid
    ) {
        List<PostingPruner> pruners = new ArrayList<>(2);
        if (sim == VectorSimilarityFunction.EUCLIDEAN) {
            pruners.add(new ClipPostingPruner(params.query(), transformedCentroid, header.distancesToCentroid(), sim));
        } else {
            pruners.add(new AdcCorrectionsPruner(ctx, encoding, sim, dimension, reader));
        }
        return pruners.toArray(new PostingPruner[0]);
    }
}
