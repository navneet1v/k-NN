package org.opensearch.knn.clusterann.read.block;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;

import java.io.IOException;

/**
 * Walks one sequence block by block, scoring the positions it wants and handing them out one at a time.
 *
 * <p>Block geometry — how the sequence is divided and how many vectors the block it is on holds — comes from
 * the {@link BlockVectorFormat.Reader}. Turning a block-local position back into an ordinal is this class's
 * own business, since the ordinal mapping belongs to the posting rather than to the block storage.
 *
 * <p>Per block it positions (free), builds the set of positions the filter accepts, and only then fetches and
 * decodes (expensive) — so a block with nothing wanted costs a position and no more.
 */
public class BlockPostingScorer implements PostingScorer {

    private final BlockVectorScorer scorer;
    private final BlockVectorFormat.Reader reader;
    private final int[] ordinals;
    private final int vectorCount;
    private final Bits acceptedOrds;
    private final int numBlocks;
    private final int blockSize;

    private final BlockVectorScorer.BlockCandidates candidates = new BlockVectorScorer.BlockCandidates();

    /** Positions of the current block the filter accepts. Sized to a full block; a partial one leaves a tail unset. */
    private final FixedBitSet validPos;

    /** Next block to visit. */
    private int blockIndex = 0;

    /**
     * Where the block in {@link #candidates} starts, so {@link #ord()} maps back after {@link #blockIndex}
     * moves on.
     */
    private int scoredBlockVectorOffset;

    /** Cursor into {@link #candidates}. */
    private int cursor = -1;

    /**
     * @param ordinals position-to-ordinal map for this posting; only the first {@code vectorCount} entries are
     *     this posting's, since it may be a buffer the scan reuses across postings
     * @param vectorCount entries in this posting
     */
    public BlockPostingScorer(final BlockVectorScorer scorer, final int[] ordinals, final int vectorCount, final Bits acceptedOrds) {
        this.scorer = scorer;
        this.reader = scorer.reader();
        this.ordinals = ordinals;
        this.vectorCount = vectorCount;
        this.acceptedOrds = acceptedOrds;
        this.numBlocks = reader.numBlocks();
        this.blockSize = reader.blockSize();
        this.validPos = new FixedBitSet(blockSize);
        candidates.growNoCopy(blockSize);
    }

    @Override
    public boolean advance(float minCompetitiveSimilarity) throws IOException {
        if (cursor != -1 && ++cursor < candidates.getSize()) {
            return true;
        }
        // Before reading `ordinals`, which may be a buffer the scan shares: if the reader has moved on, the walk
        // would silently run out of blocks against another posting's ordinals rather than fail.
        scorer.ensureCurrentSequence();

        while (reader.advance(blockIndex)) {
            int currentBlock = blockIndex;
            blockIndex = nextWantedBlock(currentBlock + 1);

            // Prefetch only the next block in anticipation
            if (blockIndex < numBlocks) {
                reader.prefetchBlock(blockIndex);
            }

            int postingVectorOffset = currentBlock * blockSize;
            int vectorCount = reader.blockVectorCount();
            if (validPos(postingVectorOffset, vectorCount) != 0) {
                reader.fetchBlock();
                reader.readBlockVectors();
                ClusterANNStat.BLOCKS_FETCHED.increment();

                float maxScore = scorer.scoreBlock(validPos, candidates);
                // Counted here rather than at the collector: every position in `candidates` cost a decode and a
                // distance, including the whole block's worth that the threshold check below may discard.
                ClusterANNStat.VECTORS_SCORED.record(candidates.getSize());
                scoredBlockVectorOffset = postingVectorOffset;
                cursor = 0;
                if (maxScore > minCompetitiveSimilarity) {
                    return true;
                }
            }
        }

        candidates.setSize(0);
        return false;
    }

    @Override
    public int ord() {
        return ordinals[scoredBlockVectorOffset + candidates.getPositions()[cursor]];
    }

    @Override
    public float score() {
        return candidates.getScores()[cursor];
    }

    /**
     * First block at or after {@code from} holding an ordinal the filter accepts, or {@link #numBlocks} when
     * none is left. Decided from {@link #ordinals} alone, so looking ahead costs no IO — which is what lets
     * the hint go to the block that will actually be fetched rather than the adjacent one.
     */
    private int nextWantedBlock(int from) {
        if (acceptedOrds == null) {
            return from;
        }

        for (int block = from; block < numBlocks; block++) {
            int startPos = block * blockSize;
            int endPos = Math.min(startPos + blockSize, vectorCount);
            for (int pos = startPos; pos < endPos; pos++) {
                if (acceptedOrds.get(ordinals[pos])) {
                    return block;
                }
            }
        }
        return numBlocks;
    }

    /** Marks the positions of the current block the filter accepts, returns the cardinality. */
    private int validPos(int postingVectorOffset, int vectorCount) {
        validPos.clear();

        if (acceptedOrds == null) {
            validPos.set(0, vectorCount);
            return vectorCount;
        }

        int cardinality = 0;
        for (int pos = 0; pos < vectorCount; pos++) {
            if (acceptedOrds.get(ordinals[postingVectorOffset + pos])) {
                validPos.set(pos);
                cardinality++;
            }
        }
        return cardinality;
    }
}
