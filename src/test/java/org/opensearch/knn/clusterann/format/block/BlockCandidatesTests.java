/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer.BlockCandidates;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link BlockCandidates} (JUnit 5).
 *
 * <p>The buffer is reused across every block of a scan, so what matters is that growing it is a one-off: it
 * must be big enough after the first ask, and must not reallocate on later asks it already satisfies.
 */
class BlockCandidatesTests {

    @Test
    void testGrowNoCopy_whenBufferIsTooSmall_thenGrowsBothArrays() {
        // given
        BlockCandidates candidates = new BlockCandidates();

        // when
        candidates.growNoCopy(4);

        // then
        assertTrue(candidates.getPositions().length >= 4, "the buffer must hold at least the requested size");
        assertEquals(
            candidates.getPositions().length,
            candidates.getScores().length,
            "positions and scores are indexed together, so they must stay the same length"
        );
    }

    @Test
    void testGrowNoCopy_whenBufferAlreadyFits_thenKeepsTheSameArrays() {
        // given
        BlockCandidates candidates = new BlockCandidates();
        candidates.growNoCopy(8);
        int[] positions = candidates.getPositions();
        float[] scores = candidates.getScores();

        // when
        candidates.growNoCopy(8);

        // then
        assertSame(positions, candidates.getPositions(), "a buffer that already fits must not be reallocated");
        assertSame(scores, candidates.getScores(), "a buffer that already fits must not be reallocated");
    }
}
