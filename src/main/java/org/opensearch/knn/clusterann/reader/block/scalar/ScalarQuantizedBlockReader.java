/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;

import java.io.IOException;

/**
 * {@link BlockVectorFormat.Reader} over one cluster's scalar-quantized vectors. A block stores the
 * per-vector corrections ahead of the packed codes, so {@link #fetchBlock()} pays only for the handful of
 * cheap values a scorer needs and the payload stays unread until {@link #readBlockVectors()}.
 *
 * <p>Every block occupies the same byte stride, the last and partial one included, so positioning a block is
 * arithmetic rather than a walk. Every buffer is sized once at construction and refilled in place, which is
 * what keeps a scan free of per-block allocation.
 */
public class ScalarQuantizedBlockReader implements BlockVectorFormat.Reader {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(ScalarQuantizedBlockReader.class);

    private final IndexInput in;
    private final int blockSize;
    private final int vectorCount;
    private final int numOfBlocks;
    private final int packedBytesPerVector;
    private final long fixedBlockBytes;

    // buffer for floats
    private final int[] intScratch;

    // Scalar corrections
    private final float[] lower;
    private final float[] upper;
    private final float[] add;
    private final int[] sum;

    private final byte[] codes;

    private int blockCursor = -1;

    public ScalarQuantizedBlockReader(IndexInput quantizedSlice, int blockSize, int vectorCount, int dimension, ScalarEncoding encoding) {
        this.in = quantizedSlice;
        this.blockSize = blockSize;

        this.vectorCount = vectorCount;
        this.numOfBlocks = (vectorCount + blockSize - 1) / blockSize;

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

    private boolean inRange() {
        return blockCursor > -1 && blockCursor < numOfBlocks;
    }

    private void ensureInRange() {
        if (!inRange()) {
            throw new IllegalStateException("blockCursor is not in range [0," + numOfBlocks + "]");
        }
    }
}
