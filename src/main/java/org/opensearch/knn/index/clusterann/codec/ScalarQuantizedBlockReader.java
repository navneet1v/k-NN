/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * Block iterator over one posting's block-columnar scalar-quantized section in {@code .clap}.
 *
 * <p>Sole responsibility is reading the on-disk layout: position on a block, exposing its corrections
 * columns ({@code lower/upper/add/sum}) and, on demand, its packed codes. The section is
 * <b>fixed-stride</b> — every vector occupies {@code 4·4 + packedBytes} bytes — so block {@code b}
 * begins at exactly {@code b·BLOCK_SIZE·perVectorBytes} and {@link #seekToBlock(int)} is pure
 * arithmetic plus one seek. Blocks that are never seeked to cost nothing: not their codes, not even
 * their corrections. Having landed on a block the caller reads the corrections it needs and then
 * either calls {@link #readCodes()} to score it or seeks past it. No scoring, no doc mapping — those
 * belong to the {@link ScalarQuantizedBlockScorer} and {@link PostingScorer} respectively.
 *
 * <p>Backed by a clone of the bounded quantized-section slice; not thread-safe. Implements the generic
 * {@link BlockReader} cursor ({@link #seekToBlock(int)}/{@link #blockStart()}/{@link #blockLength()});
 * the columnar accessors below ({@code lower/upper/add/sum/readCodes}) are the scalar-quant specifics
 * its {@link ADCBlockScorer} and {@link AdcCorrectionsPruner} read.
 */
final class ScalarQuantizedBlockReader implements BlockReader {

    private static final int CORR_INTS_PER_VECTOR = 4; // lower, upper, add, sum

    private final IndexInput in;
    private final int count;
    private final int packedBytes;
    private final int numBlocks;
    private final long perVectorBytes;  // corrections + codes, per vector (block byte stride unit)

    private final int[] intScratch = new int[BLOCK_SIZE];
    private final float[] lower = new float[BLOCK_SIZE];
    private final float[] upper = new float[BLOCK_SIZE];
    private final float[] add = new float[BLOCK_SIZE];
    private final int[] sum = new int[BLOCK_SIZE];
    private final byte[] codes;

    private int currentBlock = -1;
    private int blockStart;
    private int blockLen;

    ScalarQuantizedBlockReader(IndexInput quantizedSlice, int count, int packedBytes) {
        this.in = quantizedSlice.clone();
        this.count = count;
        this.packedBytes = packedBytes;
        this.numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this.perVectorBytes = (long) CORR_INTS_PER_VECTOR * Integer.BYTES + packedBytes;
        this.codes = new byte[BLOCK_SIZE * packedBytes];
    }

    @Override
    public int numBlocks() {
        return numBlocks;
    }

    /**
     * Position on {@code block} and read its corrections columns ({@code lower/upper/add/sum});
     * returns {@code false} when out of range. The section is fixed-stride, so this is one absolute
     * seek — blocks jumped over are never touched. After returning, the file pointer sits exactly at
     * the block's codes, ready for {@link #readCodes()}.
     */
    @Override
    public boolean seekToBlock(int block) throws IOException {
        currentBlock = block;
        if (block < 0 || block >= numBlocks) {
            return false;
        }
        blockStart = block * BLOCK_SIZE;
        blockLen = Math.min(BLOCK_SIZE, count - blockStart);
        in.seek((long) blockStart * perVectorBytes);
        readFloatsFromInts(lower, blockLen);
        readFloatsFromInts(upper, blockLen);
        readFloatsFromInts(add, blockLen);
        in.readInts(sum, 0, blockLen);
        return true;
    }

    @Override
    public boolean nextBlock() throws IOException {
        return seekToBlock(currentBlock + 1);
    }

    /**
     * Level-2 (block) prefetch: hint a block's bytes while the current one is scored, so the
     * intra-posting stream stays ahead of the reader (L1 only prefetched the posting's prefix). The
     * caller hints the block it will actually visit next, so hints aren't spent on skipped blocks.
     */
    @Override
    public void prefetchBlock(int block) throws IOException {
        if (block < 0 || block >= numBlocks) {
            return;
        }
        long offset = (long) block * BLOCK_SIZE * perVectorBytes;
        long len = Math.min((long) BLOCK_SIZE * perVectorBytes, in.length() - offset);
        if (len > 0) {
            in.prefetch(offset, len);
        }
    }


    /** Posting-local index of the first vector in the current block. */
    @Override
    public int blockStart() {
        return blockStart;
    }

    /** Number of vectors in the current block. */
    @Override
    public int blockLength() {
        return blockLen;
    }

    float[] lower() {
        return lower;
    }

    float[] upper() {
        return upper;
    }

    float[] add() {
        return add;
    }

    int[] sum() {
        return sum;
    }

    /**
     * Read the current block's packed codes; valid until the next {@link #seekToBlock(int)}. Call at
     * most once per block — the pointer was left at the codes by the seek, and this consumes them.
     */
    byte[] readCodes() throws IOException {
        in.readBytes(codes, 0, blockLen * packedBytes);
        return codes;
    }

    private void readFloatsFromInts(float[] out, int n) throws IOException {
        in.readInts(intScratch, 0, n);
        for (int i = 0; i < n; i++) {
            out[i] = Float.intBitsToFloat(intScratch[i]);
        }
    }
}
