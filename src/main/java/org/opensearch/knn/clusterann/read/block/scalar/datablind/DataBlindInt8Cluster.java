/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.read.ScoreComputer;
import org.opensearch.knn.clusterann.read.block.BlockPostingScorer;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;
import org.opensearch.knn.quantization.DataBlindInt8Quantizer;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.IOException;

/**
 * One IVF cluster whose vectors are data-blind int8 quantized and laid out in blocks, scored by the absolute
 * (centroid-free) int8 estimate.
 *
 * <p>Its posting is the slice handed in, covering exactly this cluster:
 *
 * <pre>
 * ordinals          clusterSize ints              // ascending ‖c−v‖, primary and SOAR entries mixed together
 * soarBitset        bits2words(clusterSize) longs // FixedBitSet words, bit set = this entry is a SOAR copy
 * sortedDistances   clusterSize floats            // ‖c−v‖ ascending, parallel to ordinals
 * block 0           scale[BS] | sum[BS] | sqNorm[BS] | codes[BS × dim]
 * block 1           …
 * </pre>
 *
 * <p><b>No centroid.</b> Unlike
 * {@link org.opensearch.knn.clusterann.read.block.scalar.ScalarQuantizedCluster}, this cluster never
 * reads its centroid to score:
 * the scheme is absolute. The query is rotated with the field's rotation and quantized <em>once</em> in
 * {@link #prepareScan} — the same context then scores every cluster, independent of where the query routed.
 *
 * <p>Nothing is read until {@link #scorer} is iterated; the constructor only records where things are, which
 * it can do because the header's size follows from {@code clusterSize} alone. One instance per query; not
 * thread-safe.
 *
 * <p><b>Block buffers belong to the scan.</b> They are sized by the field's geometry, so one set serves every
 * cluster a query probes: {@link org.opensearch.knn.clusterann.read.ClusterFactory#newScanBuffers()}
 * makes them from the field's metadata and {@link #scorer} resets them onto this cluster.
 * That keeps a probe from allocating a block's worth of buffers, which dominated the search path's allocation. The
 * consequence is that the scorer {@link #scorer} hands out must be drained before the next cluster is scanned —
 * {@link org.opensearch.knn.clusterann.read.orchestration.ClusterSearcher} does exactly that, and
 * {@link DataBlindInt8BlockScorer} throws rather than score on if it does not.
 */
@NotThreadSafe
public class DataBlindInt8Cluster implements Cluster {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(DataBlindInt8Cluster.class);

    private final IndexInput posting;
    private final int ordinal;
    private final int clusterSize;
    private final int dimension;
    private final VectorSimilarityFunction similarityFunction;
    private final DataBlindInt8Quantizer quantizer;

    /** Bytes before block 0. Arithmetic, not measured, which keeps the constructor free of IO. */
    private final long headerBytes;

    private final ScoreComputer computer;

    /** Global vector ordinals, ascending by distance from the centroid. Read on the first {@link #scorer}. */
    private int[] ordinals;

    /**
     * Records where this cluster's data is. Reads nothing.
     *
     * @param posting a slice covering exactly this cluster, its header first and block 0 at {@code headerBytes}
     * @param ordinal this cluster's centroid ordinal within the field
     * @param clusterSize entries in the posting, primary and SOAR together
     * @param dimension the field's vector dimension
     * @param similarityFunction the field's similarity, which selects the score transform
     * @param computer scores each block's accepted positions; the field's choice at reader open
     */
    public DataBlindInt8Cluster(
        final IndexInput posting,
        final int ordinal,
        final int clusterSize,
        final int dimension,
        final VectorSimilarityFunction similarityFunction,
        final ScoreComputer computer
    ) throws IOException {
        this.posting = posting;
        this.ordinal = ordinal;
        this.clusterSize = clusterSize;
        this.dimension = dimension;
        this.similarityFunction = similarityFunction;
        this.computer = computer;
        this.quantizer = new DataBlindInt8Quantizer(dimension);

        this.headerBytes = (long) clusterSize * Integer.BYTES              // ordinals
            + (long) FixedBitSet.bits2words(clusterSize) * Long.BYTES      // soarBitset
            + (long) clusterSize * Float.BYTES;                            // sortedDistances

        if (posting.length() < headerBytes) {
            throw new IllegalArgumentException(
                "posting is " + posting.length() + " bytes, too short for a " + clusterSize + "-entry header needing " + headerBytes
            );
        }
    }

    @Override
    public int ordinal() {
        return ordinal;
    }

    @Override
    public int size() {
        return clusterSize;
    }

    @Override
    public PostingScorer scorer(final ScanContext scanContext, final Bits acceptedOrds, final ScanBuffers scanBuffers) throws IOException {
        if (!(scanContext instanceof DataBlindInt8ScanContext int8Context)) {
            throw new IllegalArgumentException(
                "A data-blind int8 cluster needs a data-blind int8 scan context, got: "
                    + (scanContext == null ? "null" : scanContext.getClass().getSimpleName())
            );
        }
        if (!(scanBuffers instanceof DataBlindInt8ScanBuffers int8Buffers)) {
            throw new IllegalArgumentException(
                "A data-blind int8 cluster needs data-blind int8 scan buffers, got: "
                    + (scanBuffers == null ? "null" : scanBuffers.getClass().getSimpleName())
            );
        }
        load(int8Buffers);

        final DataBlindInt8BlockReader reader = int8Buffers.reader();
        reader.reset(posting.slice("blocks", headerBytes, posting.length() - headerBytes), clusterSize);

        final DataBlindInt8ComputeContext computeContext = new DataBlindInt8ComputeContext(
            reader.codes(),
            reader.scale(),
            reader.sum(),
            reader.sqNorm(),
            dimension,
            int8Context
        );

        final DataBlindInt8BlockScorer scorer = new DataBlindInt8BlockScorer(reader, computeContext, computer);
        return new BlockPostingScorer(scorer, ordinals, clusterSize, acceptedOrds);
    }

    /**
     * Quantize the query <em>once</em>, absolutely — no centroid. The query arrives already rotated by the
     * field's rotation (the reader rotates it before the scan, exactly as it does for the optimized-scalar
     * family), so this does not rotate again; it just applies the same {@code maxAbs/127} absolute encoding the
     * stored codes use. The resulting context is centroid-independent, so it is valid for every cluster this
     * query visits.
     */
    @Override
    public ScanContext prepareScan(final ScanParams scanParams) throws IOException {
        final float[] rotatedQuery = scanParams.query();
        if (rotatedQuery.length != dimension) {
            throw new IllegalArgumentException("query is " + rotatedQuery.length + "-dimensional, field is " + dimension);
        }
        final byte[] queryCode = new byte[dimension];
        final DataBlindInt8Quantizer.Int8Code corrections = quantizer.encode(rotatedQuery, queryCode);
        return new DataBlindInt8ScanContext(rotatedQuery, queryCode, quantizer.queryTerms(corrections), similarityFunction);
    }

    /**
     * What this cluster itself holds, which is nothing that outlives it: the block buffers and the ordinals both
     * belong to the scan's {@link DataBlindInt8ScanBuffers}, shared by every cluster the query probes, so counting
     * them here would report one scan's buffers once per probed cluster.
     */
    @Override
    public long ramBytesUsed() {
        return BASE_RAM_USAGE;
    }

    /**
     * Reads this cluster's ordinals into the scan's buffer. Idempotent within one cluster, but the buffer is shared:
     * it holds these ordinals only until the next cluster of the scan loads its own, which is why the scorer handed
     * out by {@link #scorer} must be drained first.
     */
    private void load(final DataBlindInt8ScanBuffers int8Buffers) throws IOException {
        if (ordinals != null) {
            return;
        }
        posting.seek(0);
        final int[] readOrdinals = int8Buffers.clusterOrdinals(clusterSize);
        posting.readInts(readOrdinals, 0, clusterSize);
        ordinals = readOrdinals;
    }
}
