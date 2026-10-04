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
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;
import org.opensearch.knn.clusterann.read.Centroid;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.read.block.BlockPostingScorer;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

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

    /**
     * The header's {@code sortedDistances}: {@code ‖c−v‖²} per member, parallel to {@link #ordinals}. Read only
     * where a pruner can use it — the similarities whose member order makes it a bound — and {@code null}
     * otherwise. What a block's worth of it means is the pruner's business, not this class's.
     */
    private float[] sortedDistances;

    /**
     * Whether to inject the geometric CLIP pruners at all.
     *
     * <p>Off leaves the scan with only its two in-hand guards — the filter's accepted positions and the
     * corrections ceiling — which is what isolates block-based pruning for a measurement. The accepted-docs
     * pruner is unaffected either way: {@code BlockPostingScorer} composes that one in itself, so a filtered
     * query still steps over blocks the filter rejects.
     *
     * <p>A compile-time constant rather than a query knob on purpose. The arms of a comparison are separate
     * builds already, and a switch on the read path is not something worth carrying into production to settle a
     * benchmark question.
     */
    static final boolean CLIP_PRUNING_ENABLED = true;

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
     * @param ioFetchBytes the field's byte budget for one block; this cluster sizes its own blocks from it, by
     *     the same arithmetic the writer used, so nothing above it handles a vector count
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
        int ioFetchBytes,
        int dimension,
        ScalarEncoding encoding,
        OptimizedScalarQuantizer quantizer,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        this.posting = posting;
        this.ordinal = ordinal;
        this.clusterSize = clusterSize;
        this.centroidSupplier = centroidSupplier;
        this.dimension = dimension;
        this.encoding = encoding;
        this.quantizer = quantizer;
        this.similarityFunction = similarityFunction;
        this.blockSize = ScalarBlockLayout.blockSize(ioFetchBytes, encoding, dimension);

        this.headerBytes = (long) clusterSize * Integer.BYTES              // ordinals
            + (long) FixedBitSet.bits2words(clusterSize) * Long.BYTES      // soarBitset
            + (long) clusterSize * Float.BYTES;                            // sortedDistances

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
    public PostingScorer scorer(ScanContext scanContext, Bits acceptedOrds) throws IOException {
        if (!(scanContext instanceof SQScanContext sqScanContext)) {
            throw new IllegalArgumentException(
                "A scalar quantized cluster needs a scalar quantized scan context, got: "
                    + (scanContext == null ? "null" : scanContext.getClass().getSimpleName())
            );
        }

        load();
        ScalarQuantizedBlockScorer scorer = new ScalarQuantizedBlockScorer(reader, sqScanContext, encoding, similarityFunction);
        return new BlockPostingScorer(scorer, ordinals, acceptedOrds, clipPruner(sqScanContext));
    }

    /**
     * Pruning on how far this cluster's members sit from its centroid, which only bounds anything where the
     * members are ordered by that distance. Which bound that is, and what {@code correction()} holds, both
     * follow from the similarity:
     *
     * <ul>
     *   <li><b>EUCLIDEAN</b> — {@code correction()} is {@code ‖q−c‖²}, members are nearest-first, and the
     *       bound is a V-shaped floor that skips on one arm and terminates on the other.
     *   <li><b>MAXIMUM_INNER_PRODUCT</b> — {@code correction()} is {@code ⟨q,c⟩}, members are farthest-first,
     *       and the bound is a ceiling that only falls along the walk, so it terminates and never skips.
     * </ul>
     *
     * <p>Anything else prunes nothing: {@link #sortedDistances} is only read where the member order makes it a
     * bound, so there is nothing here to bound with.
     */
    private BlockPostingsPruner clipPruner(final SQScanContext scanContext) {
        // TODO: return NONE for sparse filters as well so recall doesn't drop
        if (!CLIP_PRUNING_ENABLED) {
            return BlockPostingsPruner.NONE;
        }
        if (sortedDistances == null) {
            return BlockPostingsPruner.NONE;
        }
        return switch (similarityFunction) {
            case EUCLIDEAN -> new EuclideanClipPruner(
                sortedDistances,
                blockSize,
                (float) Math.sqrt(Math.max(0f, scanContext.correction()))
            );
            // correction() is ⟨q,c⟩ here, and ‖q‖ turns a shell's radius into the most its residual can add.
            case MAXIMUM_INNER_PRODUCT -> new MaximumInnerProductClipPruner(
                sortedDistances,
                blockSize,
                scanContext.correction(),
                queryNorm(scanContext.query())
            );
            default -> BlockPostingsPruner.NONE;
        };
    }

    /** {@code ‖q‖}, from the query as it arrived — the same space the stored residuals were measured in. */
    private static float queryNorm(final float[] query) {
        return (float) Math.sqrt(VectorUtil.dotProduct(query, query));
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
            quantized.additionalCorrection()
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
        if (sortedDistances != null) {
            bytes += RamUsageEstimator.sizeOf(sortedDistances);
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

        // Read only where the member order makes it a bound: ascending for L2, descending for inner product.
        if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN
            || similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            sortedDistances = readSortedDistances();
        }

        centroid = centroid();

        // Assigned last: it is the flag that says the rest is ready, so a failed read leaves nothing half-loaded.
        ordinals = readOrdinals;
    }

    /** The header's squared distances, in member order — the region after the ordinals and the SOAR bitset. */
    private float[] readSortedDistances() throws IOException {
        long distancesOffset = (long) clusterSize * Integer.BYTES + (long) FixedBitSet.bits2words(clusterSize) * Long.BYTES;
        posting.seek(distancesOffset);
        float[] distances = new float[clusterSize];
        posting.readFloats(distances, 0, clusterSize);
        return distances;
    }
}
