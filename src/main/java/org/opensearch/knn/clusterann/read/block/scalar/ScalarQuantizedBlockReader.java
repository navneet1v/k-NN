/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;

import javax.annotation.concurrent.NotThreadSafe;
import java.io.IOException;

/**
 * {@link BlockVectorFormat.Reader} over one cluster's scalar-quantized vectors. A block stores the
 * per-vector corrections ahead of the packed codes, so {@link #fetchBlock()} pays only for the handful of
 * cheap values a scorer needs and the payload stays unread until {@link #readBlockVectors()}.
 *
 * <p>Every block occupies the same byte stride, the last and partial one included, so positioning a block is
 * arithmetic rather than a walk. Every buffer is sized once at construction and refilled in place, which is
 * what keeps a scan free of per-block allocation.
 *
 * <p><b>Reused across clusters.</b> The buffers depend only on the field's geometry ({@code blockSize},
 * {@code dimension}, {@code encoding}), which every cluster of a field shares, so one reader serves a whole scan:
 * the constructor sizes the buffers and {@link #reset} points them at the next cluster's blocks. A scan holds one
 * of these (see {@link SQScanBuffers}) rather than allocating a block's worth of buffers per probed cluster.
 */
@NotThreadSafe
public class ScalarQuantizedBlockReader implements BlockVectorFormat.Reader {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(ScalarQuantizedBlockReader.class);

    private final int blockSize;
    private final int packedBytesPerVector;
    private final long fixedBlockBytes;

    /** The cluster this reader currently points at. Null until the first {@link #reset}. */
    private IndexInput in;
    private int vectorCount;
    private int numOfBlocks;

    /**
     * Bumped by every {@link #reset}, so a scorer built against one cluster can tell that the reader has since been
     * pointed at another. See {@link ScalarQuantizedBlockScorer}.
     */
    private int generation;

    // buffer for floats
    private final int[] intScratch;

    // Scalar corrections
    private final float[] lower;
    private final float[] upper;
    private final float[] add;
    private final int[] sum;

    private final byte[] codes;

    private int blockCursor = -1;

    /**
     * Sizes the block buffers for one field's geometry. Reads nothing and points at no cluster — call
     * {@link #reset} before any other method.
     *
     * @param blockSize vectors per block, the last block of a cluster excepted
     * @param dimension the field's vector dimension
     * @param encoding the stored code width, which fixes the packed bytes per vector
     */
    public ScalarQuantizedBlockReader(int blockSize, int dimension, ScalarEncoding encoding) {
        this.blockSize = blockSize;

        this.packedBytesPerVector = encoding.getDocPackedLength(dimension);

        this.intScratch = new int[this.blockSize];
        this.lower = new float[this.blockSize];
        this.upper = new float[this.blockSize];
        this.add = new float[this.blockSize];
        this.sum = new int[this.blockSize];

        this.fixedBlockBytes = (long) this.blockSize * packedBytesPerVector + (long) lower.length * Float.BYTES + (long) upper.length
            * Float.BYTES + (long) add.length * Float.BYTES + (long) sum.length * Integer.BYTES;

        this.codes = new byte[this.blockSize * this.packedBytesPerVector];

    }

    /**
     * Points this reader at another cluster's blocks, keeping the buffers. The cursor rewinds, so the next
     * {@link #advance} starts that cluster from the top.
     *
     * <p>Every scorer already built against this reader is invalidated: its {@link #generation} no longer matches,
     * and it will refuse to score rather than read a mixture of two clusters. So reset only between clusters, never
     * while a scorer is still being drained.
     *
     * @param blocks a slice covering exactly this cluster's blocks, block 0 first
     * @param vectorCount entries in the cluster, primary and SOAR together
     */
    public void reset(IndexInput blocks, int vectorCount) {
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
    public boolean advance(int blockPos) throws IOException {
        ensureReset();
        if (blockPos >= numOfBlocks) {
            in.seek(in.length());
            blockCursor = numOfBlocks;
            return false;
        }
        blockCursor = blockPos;
        ensureInRange();
        long start = fixedBlockBytes * blockCursor;
        in.seek(start);
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
        int length = blockVectorCount();
        readFloatsFromInts(lower, length);
        readFloatsFromInts(upper, length);
        readFloatsFromInts(add, length);
        in.readInts(sum, 0, length);
    }

    @Override
    public void readBlockVectors() throws IOException {
        int size = blockVectorCount();
        in.readBytes(codes, 0, size * packedBytesPerVector);
    }

    @Override
    public void prefetchBlock(int blockPos) throws IOException {
        ensureReset();
        if (blockPos < 0 || blockPos >= numOfBlocks) {
            return;
        }
        long offset = (long) blockPos * fixedBlockBytes;
        long len = Math.min(fixedBlockBytes, in.length() - offset);
        if (len > 0) {
            in.prefetch(offset, len);
        }
    }

    /**
     * Every buffer this reader holds, all of them sized once at construction and reused for the whole scan.
     * {@link #codes} dominates: it is the block's whole packed payload, while the corrections are a handful of
     * values per vector.
     */
    @Override
    public long ramBytesUsed() {
        long buffers = RamUsageEstimator.sizeOf(intScratch);
        buffers += RamUsageEstimator.sizeOf(lower);
        buffers += RamUsageEstimator.sizeOf(upper);
        buffers += RamUsageEstimator.sizeOf(add);
        buffers += RamUsageEstimator.sizeOf(sum);
        buffers += RamUsageEstimator.sizeOf(codes);
        return BASE_RAM_USAGE + buffers;
    }

    float[] lower() {
        return lower;
    }

    float[] upper() {
        return upper;
    }

    /** EUCLIDEAN: ‖v−c‖² and IP/COSINE: ⟨v,c⟩ */
    float[] addCor() {
        return add;
    }

    /** Σᵢaᵢ where aᵢ is the quantized code */
    int[] sum() {
        return sum;
    }

    /** The codes loaded by the last {@link #readBlockVectors()}. */
    byte[] codes() {
        return codes;
    }

    private void readFloatsFromInts(float[] out, int n) throws IOException {
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
