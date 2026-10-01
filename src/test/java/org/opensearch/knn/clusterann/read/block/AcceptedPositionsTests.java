/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner.Decision;

import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class AcceptedPositionsTests {

    private static final int BLOCK_SIZE = 4;

    /** 10 positions over 3 blocks, the last one partial. */
    private static final int[] ORDINALS = { 30, 31, 12, 13, 40, 41, 22, 23, 50, 51 };

    /** Accepted at positions 0, 5, 6 and 9: one in block 0, two in block 1, one in the partial block 2. */
    @Test
    void testMask_thenMarksExactlyTheAcceptedPositionsOfTheBlock() {
        // given
        AcceptedPositions positions = filter(0, 5, 6, 9);

        // then
        assertEquals(List.of(0), maskOf(positions, 0, BLOCK_SIZE));
        assertEquals(List.of(1, 2), maskOf(positions, 1, BLOCK_SIZE));
        assertEquals(List.of(1), maskOf(positions, 2, 2), "the last block holds 2 of 4 positions");
    }

    @Test
    void testTest_whenABlockHasNothingAccepted_thenSkipsIt() {
        // given — nothing accepted in block 1
        BlockPostingsPruner pruner = filter(0, 9).pruner();

        // then
        assertEquals(Decision.SCORE, pruner.test(0, 0f));
        assertEquals(Decision.SKIP, pruner.test(1, 0f));
        assertEquals(Decision.SCORE, pruner.test(2, 0f));
    }

    /** A filter pruner never ends a posting: a later block may still hold something accepted. */
    @Test
    void testTest_thenNeverTerminates() {
        // given
        BlockPostingsPruner pruner = filter().pruner();

        // then
        for (int block = 0; block < 3; block++) {
            assertEquals(Decision.SKIP, pruner.test(block, 0f), "block " + block);
        }
    }

    /** The mask is the correctness mechanism, so a bit left by an earlier block must not survive into the next. */
    @Test
    void testMask_whenReusingTheBitSet_thenClearsWhatTheLastBlockLeft() {
        // given — everything accepted in block 0, nothing in block 1
        AcceptedPositions positions = filter(0, 1, 2, 3);
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);

        // when
        positions.mask(0, BLOCK_SIZE, validPos);
        positions.mask(1, BLOCK_SIZE, validPos);

        // then
        assertEquals(0, validPos.cardinality());
    }

    @Test
    void testOf_whenNothingIsFiltered_thenAcceptsEveryPositionAndPrunesNothing() {
        // given
        AcceptedPositions positions = AcceptedPositions.of(ORDINALS, null, BLOCK_SIZE);

        // then
        assertSame(BlockPostingsPruner.NONE, positions.pruner(), "with no filter there is nothing to prune on");
        assertEquals(List.of(0, 1, 2, 3), maskOf(positions, 1, BLOCK_SIZE));
        assertEquals(List.of(0, 1), maskOf(positions, 2, 2), "a partial block masks only the positions it holds");
    }

    // ---------------------------------------------------------------- helpers

    /** The filter accepting the ordinals that live at {@code acceptedPositions}, as a scan would see it. */
    private static AcceptedPositions filter(int... acceptedPositions) {
        FixedBitSet bits = new FixedBitSet(64);
        for (int pos : acceptedPositions) {
            bits.set(ORDINALS[pos]);
        }
        return AcceptedPositions.of(ORDINALS, (Bits) bits, BLOCK_SIZE);
    }

    /** The block-local positions {@code positions} marks, as a list so a failure reads as positions not words. */
    private static List<Integer> maskOf(AcceptedPositions positions, int block, int vectorCount) {
        FixedBitSet validPos = new FixedBitSet(BLOCK_SIZE);
        positions.mask(block, vectorCount, validPos);
        return IntStream.range(0, BLOCK_SIZE).filter(validPos::get).boxed().toList();
    }
}
