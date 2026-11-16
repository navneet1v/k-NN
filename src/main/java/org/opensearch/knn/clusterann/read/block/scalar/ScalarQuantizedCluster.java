/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.read.Centroid;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.read.ScoreComputer;
import org.opensearch.knn.clusterann.read.block.BlockPostingScorer;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.IOException;

/**
 * One IVF cluster whose vectors are scalar quantized and laid out in blocks, scored by ADC.
 *
 * <p>Its posting is the slice handed in, covering exactly this cluster:
 *
 * <pre>
 * ordinals          clusterSize ints     // ascending ‖c−v‖, primary and SOAR entries mixed together
 * soarBitset        bits2words(clusterSize) longs  // FixedBitSet words, bit set = this entry is a SOAR copy
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
 *
 * <p><b>Block buffers belong to the scan.</b> They are sized by the field's geometry, so one set serves every cluster
 * a query probes: {@link org.opensearch.knn.clusterann.read.ClusterFactory#newScanBuffers()} makes them
 * from the field's metadata and {@link #scorer} resets them onto this cluster. The consequence is that the scorer
 * {@link #scorer} hands out must be drained before the next cluster is scanned, and
 * {@link ScalarQuantizedBlockScorer} throws rather than score on if it is not.
 */
@NotThreadSafe
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
    private final ScoreComputer computer;

    /** Bytes before block 0. Arithmetic, not measured, which is what keeps the constructor free of IO. */
    private final long headerBytes;

    /** Global vector ordinals, ascending by distance from the centroid. Read on the first {@link #scorer}. */
    private int[] ordinals;

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
     * @param computer scores each block's accepted positions; the field's choice at reader open
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
        VectorSimilarityFunction similarityFunction,
        ScoreComputer computer
    ) throws IOException {
        this.computer = computer;
        this.posting = posting;
        this.ordinal = ordinal;
        this.clusterSize = clusterSize;
        this.centroidSupplier = centroidSupplier;
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.encoding = encoding;
        this.quantizer = quantizer;
        this.similarityFunction = similarityFunction;

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
    public PostingScorer scorer(ScanContext scanContext, Bits acceptedOrds, ScanBuffers scanBuffers) throws IOException {
        if (!(scanContext instanceof SQScanContext sqScanContext)) {
            throw new IllegalArgumentException(
                "A scalar quantized cluster needs a scalar quantized scan context, got: "
                    + (scanContext == null ? "null" : scanContext.getClass().getSimpleName())
            );
        }
        if (!(scanBuffers instanceof SQScanBuffers sqBuffers)) {
            throw new IllegalArgumentException(
                "A scalar quantized cluster needs scalar quantized scan buffers, got: "
                    + (scanBuffers == null ? "null" : scanBuffers.getClass().getSimpleName())
            );
        }

        load();

        ScalarQuantizedBlockReader reader = sqBuffers.reader();
        reader.reset(posting.slice("blocks", headerBytes, posting.length() - headerBytes), clusterSize);

        SQComputeContext computeContext = new SQComputeContext(
            reader.codes(),
            reader.lower(),
            reader.upper(),
            reader.addCor(),
            reader.sum(),
            encoding,
            encoding.getDocPackedLength(dimension),
            sqScanContext
        );
        ScalarQuantizedBlockScorer scorer = new ScalarQuantizedBlockScorer(reader, computeContext, computer);
        return new BlockPostingScorer(scorer, ordinals, clusterSize, acceptedOrds);
    }

    /**
     * Quantize the query into the space this cluster's codes live in: centred on <em>this</em> centroid, at the
     * query width, and transposed into bit planes so the dot product can read a plane at a time.
     */
    @Override
    public ScanContext prepareScan(ScanParams scanParams) throws IOException {
        int queryBits = scanParams.queryBits();
        if (queryBits != encoding.getQueryBitsPerDim()) {
            throw new IllegalArgumentException("This encoding scores a " + encoding.getQueryBitsPerDim() + "-bit query, got: " + queryBits);
        }

        Centroid centroid = centroid();
        float[] query = scanParams.query();

        // multiScalarQuantize centres in place, so it gets a copy
        float[] centred = query.clone();
        byte[] codes = new byte[encoding.getDiscreteDimensions(query.length)];
        OptimizedScalarQuantizer.QuantizationResult quantized = quantizer.multiScalarQuantize(
            centred,
            new byte[][] { codes },
            new byte[] { (byte) queryBits },
            centroid.vector()
        )[0];

        final byte[] queryCodes;
        if (encoding.isAsymmetric()) {
            queryCodes = new byte[encoding.getQueryPackedLength(dimension)];
            OptimizedScalarQuantizer.transposeHalfByte(codes, queryCodes);
        } else {
            queryCodes = codes;
        }

        float lower = quantized.lowerInterval();
        float scale = (quantized.upperInterval() - lower) / ((1 << queryBits) - 1);
        return new SQScanContext(
            query,
            queryBits,
            centroid.normSq(),
            queryCodes,
            lower,
            scale,
            quantized.quantizedComponentSum(),
            quantized.additionalCorrection(),
            similarityFunction
        );
    }

    /** This cluster's centroid, read once and kept: every scan of this cluster centres on the same point. */
    private Centroid centroid() throws IOException {
        if (centroid == null) {
            centroid = centroidSupplier.get();
        }
        return centroid;
    }

    /**
     * What this cluster holds on the heap, which is not fixed: only a scan adds the ordinals and the centroid. The
     * block buffers are deliberately absent — they belong to the scan's {@link SQScanBuffers}, shared by every
     * cluster the query probes, so counting them here would report one scan's buffers once per probed cluster. The
     * inputs are not counted either; they are slices of a mapped file, not heap.
     */
    @Override
    public long ramBytesUsed() {
        long bytes = BASE_RAM_USAGE;
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

        centroid = centroid();

        // Assigned last: it is the flag that says the rest is ready, so a failed read leaves nothing half-loaded.
        ordinals = readOrdinals;
    }
}
