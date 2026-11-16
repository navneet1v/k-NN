/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.ScoreComputer;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * {@link BlockVectorScorer} over data-blind int8 blocks that delegates the arithmetic to a {@link ScoreComputer}.
 *
 * <p>Per block it turns the accepted positions into a dense {@code int[]}, hands that list and the compute context
 * (the block's arrays plus the query encoded once for the scan) to the computer in one call, and returns the block
 * maximum the computer reports. Which computer runs is the field's choice at reader open: the format's
 * {@link JavaInt8ScoreComputer} unless a {@link org.opensearch.knn.clusterann.read.ScoreComputerFactory}
 * supplied another.
 *
 * <p>This is the absolute-scheme counterpart of
 * {@link org.opensearch.knn.clusterann.read.block.scalar.ScalarQuantizedBlockScorer}: the context carries
 * no centroid, because nothing is centred. One instance per query per cluster; not thread-safe.
 */
@NotThreadSafe
public class DataBlindInt8BlockScorer implements BlockVectorScorer {

    private final DataBlindInt8BlockReader reader;
    private final DataBlindInt8ComputeContext computeContext;
    private final ScoreComputer computer;
    private final int generation;

    /**
     * @param reader the scan's block reader, already reset onto the cluster this scorer walks
     * @param computeContext the reader's block buffers and the query encoded for this scan
     * @param computer scores each block's accepted positions
     * @throws IllegalArgumentException if the context holds another reader's buffers, or was built against an
     *     earlier cluster of this reader
     */
    public DataBlindInt8BlockScorer(
        final DataBlindInt8BlockReader reader,
        final DataBlindInt8ComputeContext computeContext,
        final ScoreComputer computer
    ) {
        if (computeContext.codes() != reader.codes()) {
            throw new IllegalArgumentException("The compute context holds another reader's block buffers");
        }
        this.reader = reader;
        this.computeContext = computeContext;
        this.computer = computer;
        this.generation = reader.generation();
    }

    @Override
    public BlockVectorFormat.Reader reader() {
        return reader;
    }

    @Override
    public void ensureCurrentSequence() {
        if (reader.generation() != generation) {
            throw new IllegalStateException(
                "The scan's block reader has moved to another cluster; this scorer must be drained before the next" + " cluster is scanned"
            );
        }
    }

    @Override
    public float scoreBlock(final FixedBitSet validPos, final BlockCandidates out) {
        if (validPos == null || out == null) {
            throw new IllegalArgumentException("validPos and out must not be null");
        }
        ensureCurrentSequence();
        final int[] positions = out.getPositions();

        int count = 0;
        int index = validPos.nextSetBit(0);
        while (index != DocIdSetIterator.NO_MORE_DOCS) {
            positions[count++] = index;
            // nextSetBit asserts its argument is inside the set, so the last position needs the explicit stop.
            final int next = index + 1;
            index = next < validPos.length() ? validPos.nextSetBit(next) : DocIdSetIterator.NO_MORE_DOCS;
        }
        out.setSize(count);
        // BlockPostingScorer skips a block whose accepted set is empty, so this is for other callers (tests included):
        // it keeps the contract of the interface without a call into the computer.
        if (count == 0) {
            return Float.NEGATIVE_INFINITY;
        }
        return computer.score(computeContext, positions, count, out.getScores());
    }
}
