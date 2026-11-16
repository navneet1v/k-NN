/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.IOException;

/**
 * {@link BlockVectorFormat.Reader} over one cluster's data-blind int8 vectors — the read counterpart of
 * {@link org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8BlockWriter}. A block
 * stores the per-vector corrections ahead of the whole-byte codes, so {@link #fetchBlock()} pays only for the
 * three cheap values a scorer needs and the codes stay unread until {@link #readBlockVectors()}.
 *
 * <p>One unsigned byte per dimension (no bit-plane transpose, no nibble packing), so a block's stride is fixed
 * and positioning a block is arithmetic rather than a walk. Buffers are sized once at construction and refilled
 * in place, keeping a scan free of per-block allocation.
 *
 * <p><b>Reused across clusters.</b> The buffers depend only on the field's geometry ({@code blockSize},
 * {@code dimension}), which every cluster of a field shares, so one reader serves a whole scan: the constructor
 * sizes the buffers and {@link #reset} points them at the next cluster's blocks. A scan holds one of these (see
 * {@link DataBlindInt8ScanContext}) rather than allocating a block's worth of buffers per probed cluster.
 *
 * <pre>
 *   one block of n vectors (n = blockSize, last block may be partial):
 *     scale   n x int   (floatToIntBits of maxAbs/127)
 *     sum     n x int   (signed component sum Σq)
 *     sqNorm  n x int   (floatToIntBits of ‖rotated vector‖²)
 *     codes   n x dim bytes
 * </pre>
 */
@NotThreadSafe
public class DataBlindInt8BlockReader implements BlockVectorFormat.Reader {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(DataBlindInt8BlockReader.class);

    private final int blockSize;
    private final int dimension;
    private final long fixedBlockBytes;

    private final int[] intScratch;

    private final float[] scale;
    private final int[] sum;
    private final float[] sqNorm;
    private final byte[] codes;

    /** The cluster this reader currently points at. Null until the first {@link #reset}. */
    private IndexInput in;
    private int vectorCount;
    private int numOfBlocks;

    private int blockCursor = -1;

    /**
     * Bumped by every {@link #reset}, so a scorer built against one cluster can tell that the reader has since
     * been pointed at another. See {@link DataBlindInt8BlockScorer}.
     */
    private int generation;

    /**
     * Sizes the block buffers for one field's geometry. Reads nothing and points at no cluster — call
     * {@link #reset} before any other method.
     *
     * @param blockSize vectors per block, the last block of a cluster excepted
     * @param dimension the field's vector dimension, which is also one vector's code length
     */
    public DataBlindInt8BlockReader(final int blockSize, final int dimension) {
        this.blockSize = blockSize;
        this.dimension = dimension;

        this.intScratch = new int[blockSize];
        this.scale = new float[blockSize];
        this.sum = new int[blockSize];
        this.sqNorm = new float[blockSize];
        this.codes = new byte[blockSize * dimension];

        this.fixedBlockBytes = fixedBlockBytes(blockSize, dimension);
    }

    /**
     * Bytes one whole block occupies: the three corrections per vector then the codes. Every block but a
     * cluster's last is exactly this wide, which is what makes positioning arithmetic.
     */
    private static long fixedBlockBytes(final int blockSize, final int dimension) {
        return (long) blockSize * Integer.BYTES       // scale
            + (long) blockSize * Integer.BYTES        // sum
            + (long) blockSize * Integer.BYTES        // sqNorm
            + (long) blockSize * dimension;           // codes
    }

    /**
     * Points this reader at another cluster's blocks, keeping the buffers. The cursor rewinds, so the next
     * {@link #advance} starts that cluster from the top.
     *
     * <p>Every scorer already built against this reader is invalidated: its {@link #generation} no longer
     * matches, and it will refuse to score rather than read a mixture of two clusters. So reset only between
     * clusters, never while a scorer is still being drained.
     *
     * @param blocks a slice covering exactly this cluster's blocks, block 0 first
     * @param vectorCount entries in the cluster, primary and SOAR together
     */
    public void reset(final IndexInput blocks, final int vectorCount) {
        if (blocks == null) {
            throw new IllegalArgumentException("blocks must not be null");
        }
        if (vectorCount < 0) {
            throw new IllegalArgumentException("vectorCount must not be negative, got " + vectorCount);
        }
        this.in = blocks;
        this.vectorCount = vectorCount;
        this.numOfBlocks = (vectorCount + blockSize - 1) / blockSize;
        this.blockCursor = -1;
        this.generation++;
    }

    /** Which cluster this reader is on, as an opaque counter. Changes on every {@link #reset}. */
    int generation() {
        return generation;
    }

    @Override
    public int numBlocks() {
        return numOfBlocks;
    }

    @Override
    public int blockSize() {
        return blockSize;
    }

    @Override
    public boolean advance(final int blockPos) throws IOException {
        ensureReset();
        if (blockPos >= numOfBlocks) {
            in.seek(in.length());
            blockCursor = numOfBlocks;
            return false;
        }
        blockCursor = blockPos;
        ensureInRange();
        in.seek(fixedBlockBytes * blockCursor);
        return true;
    }

    @Override
    public int blockVectorCount() {
        ensureInRange();
        if (blockCursor < numOfBlocks - 1) {
            return blockSize;
        }
        return vectorCount - blockCursor * blockSize;
    }

    @Override
    public void fetchBlock() throws IOException {
        final int length = blockVectorCount();
        readFloatsFromInts(scale, length);
        in.readInts(sum, 0, length);
        readFloatsFromInts(sqNorm, length);
    }

    @Override
    public void readBlockVectors() throws IOException {
        final int size = blockVectorCount();
        in.readBytes(codes, 0, size * dimension);
    }

    @Override
    public void prefetchBlock(final int blockPos) throws IOException {
        ensureReset();
        if (blockPos < 0 || blockPos >= numOfBlocks) {
            return;
        }
        final long offset = (long) blockPos * fixedBlockBytes;
        final long len = Math.min(fixedBlockBytes, in.length() - offset);
        if (len > 0) {
            in.prefetch(offset, len);
        }
    }

    @Override
    public long ramBytesUsed() {
        long buffers = RamUsageEstimator.sizeOf(intScratch);
        buffers += RamUsageEstimator.sizeOf(scale);
        buffers += RamUsageEstimator.sizeOf(sum);
        buffers += RamUsageEstimator.sizeOf(sqNorm);
        buffers += RamUsageEstimator.sizeOf(codes);
        return BASE_RAM_USAGE + buffers;
    }

    /** Per-vector {@code maxAbs/127} scales for the current block. */
    float[] scale() {
        return scale;
    }

    /** Per-vector signed component sums {@code Σq} for the current block. */
    int[] sum() {
        return sum;
    }

    /** Per-vector squared norms {@code ‖v‖²} for the current block. */
    float[] sqNorm() {
        return sqNorm;
    }

    /** The codes loaded by the last {@link #readBlockVectors()}. */
    byte[] codes() {
        return codes;
    }

    int dimension() {
        return dimension;
    }

    private void readFloatsFromInts(final float[] out, final int n) throws IOException {
        in.readInts(intScratch, 0, n);
        for (int i = 0; i < n; i++) {
            out[i] = Float.intBitsToFloat(intScratch[i]);
        }
    }

    private void ensureReset() {
        if (in == null) {
            throw new IllegalStateException("reset(blocks, vectorCount) must be called before reading a cluster");
        }
    }

    private boolean inRange() {
        return blockCursor > -1 && blockCursor < numOfBlocks;
    }

    private void ensureInRange() {
        if (!inRange()) {
            throw new IllegalStateException("blockCursor is not in range [0," + numOfBlocks + "]");
        }
    }
}
