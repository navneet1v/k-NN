/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.reader.Centroid;
import org.opensearch.knn.clusterann.reader.Cluster;
import org.opensearch.knn.clusterann.reader.PostingScorer;
import org.opensearch.knn.clusterann.reader.ScanParams;
import org.opensearch.knn.clusterann.reader.block.BlockPostingScorer;

import java.io.IOException;

/**
 * One IVF cluster whose vectors are scalar quantized and laid out in blocks, scored by ADC.
 *
 * <p>Its posting is the slice handed in, covering exactly this cluster:
 *
 * <pre>
 * ordinals          clusterSize ints     // ascending ‖c−v‖, primary and SOAR entries mixed together
 * soarBitset        ceil(clusterSize/8)  // one bit per entry, set = this entry is a SOAR copy
 * sortedDistances   clusterSize floats   // ‖c−v‖ ascending, parallel to ordinals
 * block 0           lower[BS] | upper[BS] | add[BS] | sum[BS] | codes[BS × packedBytes]
 * block 1           …
 * </pre>
 *
 * <p>Nothing is read until {@link #scorer} is iterated. The constructor only records where things are, which it can do
 * because the header's size follows from {@code clusterSize} alone — so a scan may obtain every cluster it might
 * visit, but nothing is read until the cluster is actually scanned.
 *
 * <p>One instance per query; not thread-safe.
 */
public class ScalarQuantizedCluster implements Cluster {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(ScalarQuantizedCluster.class);

    private final IndexInput posting;
    private final int ordinal;
    private final int clusterSize;
    private final IOSupplier<Centroid> centroidSupplier;
    private final int blockSize;
    private final int dimension;
    private final ScalarEncoding encoding;
    private final OptimizedScalarQuantizer quantizer;
    private final VectorSimilarityFunction similarityFunction;

    /** Bytes before block 0. Arithmetic, not measured, which is what keeps the constructor free of IO. */
    private final long headerBytes;

    /** Global vector ordinals, ascending by distance from the centroid. Read on the first {@link #scorer}. */
    private int[] ordinals;

    private final ScalarQuantizedBlockReader reader;

    private Centroid centroid;

    /**
     * Records where this cluster's data is. Reads nothing.
     *
     * @param posting a slice covering exactly this cluster, its header first and block 0 at {@code headerBytes}
     * @param ordinal this cluster's centroid ordinal within the field
     * @param clusterSize entries in the posting, primary and SOAR together
     * @param centroidSupplier reads this cluster's centroid and ‖c‖² from {@code .clac}, called at most once and
     *     only if this cluster is actually scanned
     * @param blockSize vectors per block, the last block excepted
     * @param dimension the field's vector dimension
     * @param encoding stored code width, which also fixes the packed bytes per vector
     * @param quantizer quantizes the query into the same space as the stored codes
     * @param similarityFunction the field's similarity, which selects the ADC transform
     */
    @SuppressWarnings("checkstyle:ParameterNumber")
    public ScalarQuantizedCluster(
        IndexInput posting,
        int ordinal,
        int clusterSize,
        IOSupplier<Centroid> centroidSupplier,
        int blockSize,
        int dimension,
        ScalarEncoding encoding,
        OptimizedScalarQuantizer quantizer,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        this.posting = posting;
        this.ordinal = ordinal;
        this.clusterSize = clusterSize;
        this.centroidSupplier = centroidSupplier;
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.encoding = encoding;
        this.quantizer = quantizer;
        this.similarityFunction = similarityFunction;

        this.headerBytes = (long) clusterSize * Integer.BYTES   // ordinals
            + (clusterSize + 7) / 8                             // soarBitset
            + (long) clusterSize * Float.BYTES;                 // sortedDistances

        IndexInput blocks = posting.slice("blocks", headerBytes, posting.length() - headerBytes);
        this.reader = new ScalarQuantizedBlockReader(blocks, blockSize, clusterSize, dimension, encoding);
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
    public void prefetch(boolean partial) throws IOException {
        if (partial) {
            posting.prefetch(0, headerBytes);
            reader.prefetchBlock(0);
            return;
        }
        posting.prefetch(0, posting.length());
    }

    @Override
    public PostingScorer scorer(ScanParams params, Bits acceptedOrds) throws IOException {
        load();

        ADCScalarQuantizedBlockScorer scorer = new ADCScalarQuantizedBlockScorer(
            reader,
            params,
            centroid,
            quantizer,
            encoding,
            similarityFunction
        );
        return new BlockPostingScorer(scorer, ordinals, acceptedOrds);
    }

    /**
     * What this cluster holds on the heap, which is not fixed: an unscanned cluster is little more than its
     * reader's buffers, and only a scan adds the ordinals and the centroid. The inputs are not counted — they are
     * slices of a mapped file, not heap.
     */
    @Override
    public long ramBytesUsed() {
        long bytes = BASE_RAM_USAGE + reader.ramBytesUsed();
        if (ordinals != null) {
            bytes += RamUsageEstimator.sizeOf(ordinals);
        }
        if (centroid != null) {
            bytes += RamUsageEstimator.shallowSizeOf(centroid) + RamUsageEstimator.sizeOf(centroid.vector());
        }
        return bytes;
    }

    /**
     * Reads what a scan needs and keeps it, so repeated scans of one cluster pay once. Idempotent, and the only
     * place this class does IO.
     */
    private void load() throws IOException {
        if (ordinals != null) {
            return;
        }

        posting.seek(0);
        int[] readOrdinals = new int[clusterSize];
        posting.readInts(readOrdinals, 0, clusterSize);

        centroid = centroidSupplier.get();

        // Assigned last: it is the flag that says the rest is ready, so a failed read leaves nothing half-loaded.
        ordinals = readOrdinals;
    }
}
