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
    private final OffHeapCentroids.Reader centroidReader;
    private boolean forceExact;

    /**
     * Flow B ("IVFaster absolute") scan tier. Non-null iff {@code fieldState.quantizerId ==
     * QUANTIZER_IVFASTER_ABSOLUTE}: the posting blocks are thermometer+int8, scored coarse by
     * Hamming then rescored exactly. UNVERIFIED (written blind).
     */
    private final ThermometerVectorReader thermoReader;

    /** Similarity function for int8 rerank (flow B), derived from fieldState.metric. */
    private final VectorSimilarityFunction simFunc;

    /**
     * Per-posting coarse Hamming shortlist width before exact rescore (flow B). The 2-bit
     * thermometer Hamming is a weak ranking signal, so the shortlist must be generous enough that
     * the true neighbors survive it; the exact rescore then ranks them. Overridable via
     * {@code -Dclusterann.thermo.shortlist=N} for tuning the recall/latency tradeoff.
     */
    private static final int FLOWB_SHORTLIST = Integer.getInteger("clusterann.thermo.shortlist", 2000);

    /**
     * Flow B rerank tier: false = full-precision exact rescore (default, higher recall), true = int8
     * rerank (IVFaster-style, faster, slightly lower recall). Toggle: {@code -Dclusterann.thermo.rerank=int8}.
     */
    private static final boolean RERANK_INT8 = "int8".equalsIgnoreCase(System.getProperty("clusterann.thermo.rerank"));

    // Reusable buffers
    private int[] docIdBuf = new int[1024];
    private int[] ordBuf = new int[1024];
    private boolean[] validBuf = new boolean[1024];
    private float[] scoreBuf = new float[1024];

    /**
     * Per-segment token for the shard-level int8 stash key, so the same segment-local docId in two
     * segments does not collide. Set once per segment by the reader; defaults to 0 (single-segment).
     */
    private int segmentToken = 0;

    public void setSegmentToken(int token) {
        this.segmentToken = token;
    }

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
        boolean useADC,
        OffHeapCentroids.Reader centroidReader,
        ThermometerVectorReader thermoReader
    ) {
        this(postingsInput, fieldState, exactScorer, adcReader, target, acceptBits, visited, useADC, centroidReader, thermoReader, null);
    }

    private final PQScanState pqState;

    public ClusterANNCentroidScanner(
        IndexInput postingsInput,
        ClusterANNFieldState fieldState,
        RandomVectorScorer exactScorer,
        QuantizedVectorReader adcReader,
        float[] target,
        Bits acceptBits,
        BitSet visited,
        boolean useADC,
        OffHeapCentroids.Reader centroidReader,
        ThermometerVectorReader thermoReader,
        PQScanState pqState
    ) {
        this.postingsInput = postingsInput;
        this.fieldState = fieldState;
        this.exactScorer = exactScorer;
        this.adcReader = adcReader;
        this.target = target;
        this.acceptBits = acceptBits;
        this.visited = visited;
        this.useADC = useADC;
        this.centroidReader = centroidReader;
        this.thermoReader = thermoReader;
        this.pqState = pqState;
        switch (fieldState.metric) {
            case L2: this.simFunc = VectorSimilarityFunction.EUCLIDEAN; break;
            case INNER_PRODUCT: this.simFunc = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT; break;
            default: this.simFunc = VectorSimilarityFunction.COSINE; break;
        }
        this.packedBytes = fieldState.docBits > 0
            ? ScalarBitEncoding.fromDocBits(fieldState.docBits).docPackedBytes(fieldState.dimension)
            : 0;
    }

    public int prepare(ProbeTarget centroid) throws IOException {
        this.centroidIdx = centroid.centroidIdx();
        postingsInput.seek(centroid.fileOffset());
        return 0;
    }

    /** Adaptive precision: force exact scoring for this cluster (sparse filter matches). */
    public void setForceExact(boolean force) {
        this.forceExact = force;
    }

    // Reusable centroid buffer — allocated once, reused across primary + SOAR postings
    private float[] centroidBuf;
    private float centroidDp;
    private float centroidNormSq;
    private int lastCentroidLoaded = -1;

    public int scan(KnnCollector collector) throws IOException {
        // Flow C: residualize the query against THIS cell's raw centroid and build the AH lookup table.
        if (pqState != null && centroidIdx != lastCentroidLoaded) {
            if (centroidBuf == null) centroidBuf = new float[fieldState.dimension];
            centroidReader.readCentroid(centroidIdx, centroidBuf);
            pqState.prepareCell(centroidBuf);
            lastCentroidLoaded = centroidIdx;
        }
        // Load centroid ONCE for both primary and SOAR postings (same centroidIdx)
        if (useADC && adcReader != null && thermoReader == null && pqState == null && centroidIdx != lastCentroidLoaded) {
            if (centroidBuf == null) {
                centroidBuf = new float[fieldState.dimension];
            }
            centroidReader.readTransformedCentroid(centroidIdx, centroidBuf);
            centroidDp = 0f;
            if (adcReader.getSimFunc() != VectorSimilarityFunction.EUCLIDEAN) {
                centroidDp = VectorUtil.dotProduct(target, centroidBuf);
            }
            // Use precomputed norm from fieldState (avoids 768 FP ops)
            centroidNormSq = fieldState.centroidNorms[centroidIdx];
            lastCentroidLoaded = centroidIdx;
        }

        int totalScored = 0;
        totalScored += scanOnePosting(collector);
        totalScored += scanOnePosting(collector);
        return totalScored;
    }

    private int scanOnePosting(KnnCollector collector) throws IOException {
        // Peek count first to ensure buffers are large enough
        int count = postingsInput.readVInt();
        if (count == 0) return 0;
        ensureCapacity(count);
        PostingListCodec.readBody(postingsInput, count, docIdBuf);

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

        if (pqState != null) {
            return scoreScannPQ(collector, count);
        }
        if (thermoReader != null) {
            return scoreThermometer(collector, count, validCount);
        }
        if (useADC && adcReader != null && !forceExact) {
            return scoreADC(collector, count, validCount);
        } else {
            return scoreExact(collector, count);
        }
    }

    /**
     * Flow B scan: coarse-scan the thermometer blocks (Hamming query-vs-doc), keep the best
     * {@link #FLOWB_SHORTLIST} valid docs, then rescore those exactly and collect. UNVERIFIED.
     */
    private int scoreThermometer(KnnCollector collector, int count, int validCount) throws IOException {
        // DIAGNOSTIC: bypass the coarse shortlist — exact-score EVERY valid doc. If recall jumps to
        // ~flow A, the coarse Hamming shortlist is the culprit; if it stays low, the bug is in
        // rotation/ordinal/scoring plumbing. Toggle via -Dclusterann.thermo.noshortlist=true.
        if (Boolean.getBoolean("clusterann.thermo.noshortlist")) {
            // consume the coarse blocks to keep the stream aligned
            int[] hthrow = new int[BLOCK_SIZE];
            int p2 = 0;
            while (p2 < count) {
                int bs = Math.min(BLOCK_SIZE, count - p2);
                thermoReader.scanBlockCoarse(postingsInput, bs, hthrow);
                p2 += bs;
            }
            int[] allOrds = new int[validCount];
            int[] allDocs = new int[validCount];
            int b2 = 0;
            for (int i = 0; i < count; i++) {
                if (validBuf[i]) { allOrds[b2] = ordBuf[i]; allDocs[b2] = docIdBuf[i]; b2++; }
            }
            float[] sc2 = new float[b2];
            exactScorer.bulkScore(allOrds, sc2, b2);
            float mc = collector.minCompetitiveSimilarity();
            for (int i = 0; i < b2; i++) if (sc2[i] > mc) collector.collect(allDocs[i], sc2[i]);
            collector.incVisitedCount(b2);
            return b2;
        }

        int[] hamming = new int[BLOCK_SIZE];
        // First pass: compute each valid doc's Hamming, stored by ORIGINAL position (no reorder).
        // For int8 rerank, also buffer the int8 code + corrections per position.
        int[] hamByPos = new int[count];
        java.util.Arrays.fill(hamByPos, Integer.MAX_VALUE);
        byte[] int8Buf = RERANK_INT8 ? new byte[count * fieldState.dimension] : null;
        float[] scaleBuf = RERANK_INT8 ? new float[count] : null;
        int[] sumBuf = RERANK_INT8 ? new int[count] : null;
        float[] normBuf = RERANK_INT8 ? new float[count] : null;
        int nCand = 0;
        int pos = 0;
        while (pos < count) {
            int blockSize = Math.min(BLOCK_SIZE, count - pos);
            if (RERANK_INT8) {
                thermoReader.scanBlockCoarseAndInt8(postingsInput, blockSize, pos, hamming, int8Buf, scaleBuf, sumBuf, normBuf);
            } else {
                thermoReader.scanBlockCoarse(postingsInput, blockSize, hamming);
            }
            for (int j = 0; j < blockSize; j++) {
                if (validBuf[pos + j]) {
                    hamByPos[pos + j] = hamming[j];
                    nCand++;
                }
            }
            pos += blockSize;
        }

        // Hamming threshold = the keep-th smallest, via a sorted copy of the valid distances only.
        int keep = Math.min(FLOWB_SHORTLIST, nCand);
        int threshold;
        if (keep >= nCand) {
            threshold = Integer.MAX_VALUE; // keep everything
        } else {
            int[] sortedHam = new int[nCand];
            int t = 0;
            for (int i = 0; i < count; i++) if (hamByPos[i] != Integer.MAX_VALUE) sortedHam[t++] = hamByPos[i];
            java.util.Arrays.sort(sortedHam);
            threshold = sortedHam[keep - 1];
        }

        float minCompetitive = collector.minCompetitiveSimilarity();
        int batch = 0;

        if (RERANK_INT8) {
            if (org.opensearch.knn.index.clusterann.codec.ShardInt8Stash.SHARD_SCOPE) {
                // POC shard-level: phase 1 collects by COARSE HAMMING (as a similarity: smaller
                // Hamming -> higher score) and STASHES the int8 code for phase-2 shard rescore.
                byte[] one = new byte[fieldState.dimension];
                for (int i = 0; i < count && batch < keep; i++) {
                    if (validBuf[i] && hamByPos[i] <= threshold) {
                        // phase-1 score: negative Hamming so the collector keeps smallest-Hamming.
                        float phase1 = -(float) hamByPos[i];
                        System.arraycopy(int8Buf, i * fieldState.dimension, one, 0, fieldState.dimension);
                        org.opensearch.knn.index.clusterann.codec.ShardInt8Stash.put(
                            org.opensearch.knn.index.clusterann.codec.ShardInt8Stash.key(segmentToken, docIdBuf[i]),
                            one, scaleBuf[i], sumBuf[i], normBuf[i]);
                        collector.collect(docIdBuf[i], phase1);
                        batch++;
                    }
                }
                collector.incVisitedCount(batch);
                return batch;
            }
            // Segment-level int8 rerank: score each shortlisted doc from its buffered code + corrections.
            for (int i = 0; i < count && batch < keep; i++) {
                if (validBuf[i] && hamByPos[i] <= threshold) {
                    float score = thermoReader.int8Score(int8Buf, i * fieldState.dimension, scaleBuf[i], sumBuf[i], normBuf[i], simFunc);
                    if (score > minCompetitive) {
                        collector.collect(docIdBuf[i], score);
                    }
                    batch++;
                }
            }
            collector.incVisitedCount(batch);
            return batch;
        }

        // Full-precision rerank: gather shortlist ordinals in original order, exact-score.
        int[] shortlistOrds = new int[nCand];
        int[] shortlistDocs = new int[nCand];
        for (int i = 0; i < count && batch < keep; i++) {
            if (validBuf[i] && hamByPos[i] <= threshold) {
                shortlistOrds[batch] = ordBuf[i];
                shortlistDocs[batch] = docIdBuf[i];
                batch++;
            }
        }
        if (batch > 0) {
            float[] sc = new float[batch];
            exactScorer.bulkScore(shortlistOrds, sc, batch);
            for (int i = 0; i < batch; i++) {
                if (sc[i] > minCompetitive) {
                    collector.collect(shortlistDocs[i], sc[i]);
                }
            }
            collector.incVisitedCount(batch);
        }
        return batch;
    }

    private int scoreADC(KnnCollector collector, int count, int validCount) throws IOException {
        // Use pre-loaded centroid buffer from scan() — no allocation, enables cache hit in ensureQueryQuantized
        int scored = 0;
        int pos = 0;
        while (pos < count) {
            int blockSize = Math.min(BLOCK_SIZE, count - pos);
            adcReader.scoreBlock(postingsInput, pos, blockSize, docIdBuf, ordBuf, validBuf, centroidBuf, centroidDp, centroidNormSq, centroidIdx);
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

    /**
     * Flow C scan: read the posting's PQ codes and score each valid doc via the AH lookup table
     * ({@code Σ_block lut[block][code[block]]} = approx residual dot). Collects the similarity for
     * the field's metric. The LUT was built for this cell in {@link #scan} via prepareCell.
     */
    private int scoreScannPQ(KnnCollector collector, int count) throws IOException {
        int codeBytes = pqState.codebook().codeBytes();
        byte[] codes = new byte[count * codeBytes];
        postingsInput.readBytes(codes, 0, codes.length);
        float minCompetitive = collector.minCompetitiveSimilarity();
        int scored = 0;
        for (int i = 0; i < count; i++) {
            if (!validBuf[i]) continue;
            // pqState.score() returns the final metric-appropriate similarity (IP or L2).
            float sim = pqState.score(codes, i * codeBytes);
            if (sim > minCompetitive) {
                collector.collect(docIdBuf[i], sim);
            }
            scored++;
        }
        collector.incVisitedCount(scored);
        return scored;
    }

    private void skipQuantizedBlocks(int count) throws IOException {
        long totalBytes;
        if (pqState != null) {
            // Flow C: count · codeBytes (one byte per subspace per doc).
            totalBytes = (long) count * pqState.codebook().codeBytes();
        } else if (thermoReader != null) {
            // Flow B: sum thermometer block bytes over full blocks (matches ThermometerVectorWriter).
            totalBytes = 0;
            int pos = 0;
            while (pos < count) {
                int blockSize = Math.min(BLOCK_SIZE, count - pos);
                totalBytes += thermoReader.blockBytes(blockSize);
                pos += blockSize;
            }
        } else {
            totalBytes = (long) count * packedBytes + (long) count * Integer.BYTES * 4;
        }
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
