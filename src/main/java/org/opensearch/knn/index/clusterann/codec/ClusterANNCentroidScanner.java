/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.opensearch.knn.index.clusterann.prefetch.ProbeTarget;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.util.BitSet;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * Reads one centroid's columnar posting data (primary + SOAR), filters, scores, collects.
 * Quantized section uses block-columnar layout (BLOCK_SIZE=32) for SIMD scoring.
 */
public final class ClusterANNCentroidScanner {

    private final IndexInput postingsInput;
    private final ClusterANNFieldState fieldState;
    private final RandomVectorScorer exactScorer;
    private final QuantizedVectorReader adcReader;
    private final float[] target;
    private final Bits acceptBits;
    private final BitSet visited;
    private final boolean useADC;
    private final int packedBytes;

    // Reusable buffers
    private int[] docIdBuf = new int[1024];
    private int[] ordBuf = new int[1024];
    private boolean[] validBuf = new boolean[1024];
    private float[] scoreBuf = new float[1024];

    // State set by prepare()
    private int centroidIdx;

    public ClusterANNCentroidScanner(
        IndexInput postingsInput,
        ClusterANNFieldState fieldState,
        RandomVectorScorer exactScorer,
        QuantizedVectorReader adcReader,
        float[] target,
        Bits acceptBits,
        BitSet visited,
        boolean useADC
    ) {
        this.postingsInput = postingsInput;
        this.fieldState = fieldState;
        this.exactScorer = exactScorer;
        this.adcReader = adcReader;
        this.target = target;
        this.acceptBits = acceptBits;
        this.visited = visited;
        this.useADC = useADC;
        this.packedBytes = fieldState.docBits > 0
            ? ScalarBitEncoding.fromDocBits(fieldState.docBits).docPackedBytes(fieldState.dimension)
            : 0;
    }

    public int prepare(ProbeTarget centroid) throws IOException {
        this.centroidIdx = centroid.centroidIdx();
        postingsInput.seek(centroid.fileOffset());
        return 0;
    }

    public int scan(KnnCollector collector) throws IOException {
        int totalScored = 0;
        totalScored += scanOnePosting(collector);
        totalScored += scanOnePosting(collector);
        return totalScored;
    }

    private int scanOnePosting(KnnCollector collector) throws IOException {
        int count = PostingListCodec.read(postingsInput, docIdBuf);
        if (count == 0) return 0;

        // Bulk read ordinals (fixed-width ints)
        int ordCount = postingsInput.readVInt();
        ensureCapacity(Math.max(count, ordCount));
        if (ordCount > 0) {
            postingsInput.readInts(ordBuf, 0, ordCount);
        }

        int validCount = 0;
        for (int i = 0; i < count; i++) {
            int doc = docIdBuf[i];
            validBuf[i] = false;
            if (visited.get(doc)) continue;
            visited.set(doc);
            if (acceptBits != null && !acceptBits.get(doc)) continue;
            validBuf[i] = true;
            validCount++;
        }

        if (validCount == 0) {
            skipQuantizedBlocks(count);
            return 0;
        }

        if (useADC && adcReader != null) {
            return scoreADC(collector, count, validCount);
        } else {
            return scoreExact(collector, count);
        }
    }

    private int scoreADC(KnnCollector collector, int count, int validCount) throws IOException {
        float[] centroid = fieldState.centroids[centroidIdx];
        float centroidDp = 0f;
        if (adcReader.getSimFunc() != VectorSimilarityFunction.EUCLIDEAN) {
            centroidDp = VectorUtil.dotProduct(target, centroid);
        }

        int scored = 0;
        int pos = 0;
        while (pos < count) {
            int blockSize = Math.min(BLOCK_SIZE, count - pos);
            adcReader.scoreBlock(postingsInput, pos, blockSize, docIdBuf, ordBuf, validBuf, centroid, centroidDp);
            for (int j = 0; j < blockSize; j++) {
                if (validBuf[pos + j]) scored++;
            }
            pos += blockSize;
        }
        collector.incVisitedCount(scored);
        return scored;
    }

    private int scoreExact(KnnCollector collector, int count) throws IOException {
        skipQuantizedBlocks(count);

        // Compact valid entries
        int batchCount = 0;
        for (int i = 0; i < count; i++) {
            if (!validBuf[i]) continue;
            ordBuf[batchCount] = ordBuf[i];
            docIdBuf[batchCount] = docIdBuf[i];
            batchCount++;
        }

        if (batchCount > 0) {
            exactScorer.bulkScore(ordBuf, scoreBuf, batchCount);
            float minCompetitive = collector.minCompetitiveSimilarity();
            for (int i = 0; i < batchCount; i++) {
                if (scoreBuf[i] > minCompetitive) {
                    collector.collect(docIdBuf[i], scoreBuf[i]);
                }
            }
            collector.incVisitedCount(batchCount);
        }
        return batchCount;
    }

    private void skipQuantizedBlocks(int count) throws IOException {
        long totalBytes = (long) count * packedBytes + (long) count * Integer.BYTES * 4;
        postingsInput.skipBytes(totalBytes);
    }

    private void ensureCapacity(int needed) {
        if (docIdBuf.length < needed) {
            int newSize = Math.max(needed, docIdBuf.length * 2);
            docIdBuf = new int[newSize];
            ordBuf = new int[newSize];
            validBuf = new boolean[newSize];
            scoreBuf = new float[newSize];
        }
    }
}
