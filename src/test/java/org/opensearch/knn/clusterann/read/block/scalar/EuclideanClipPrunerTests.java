/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner.Decision;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EuclideanClipPrunerTests {

    private static final int BLOCK_SIZE = 4;
    private static final int NUM_BLOCKS = 3;

    /**
     * Squared distances in member order, as the posting stores them. Each block's first entry is the one the
     * pruner reads: shells at 1, 2 and 4 from the centroid, the rest of each block trailing behind it.
     */
    private static final float[] SORTED_DISTANCES = { 1f, 1.2f, 1.4f, 1.6f, 4f, 4.2f, 4.4f, 4.6f, 16f, 16.2f };

    /** The query, half a unit from the centroid — nearer than every shell, so all three carry a bound. */
    private static final float QUERY_TO_CENTROID = 0.5f;

    /** {@code 1/(1+gap²)} for the gaps this fixture produces: 0.5, 1.5 and 3.5. */
    private static final float BLOCK_1_BOUND = 1f / (1f + 1.5f * 1.5f);   // 0.3077
    private static final float BLOCK_2_BOUND = 1f / (1f + 3.5f * 3.5f);   // 0.0755

    private static Stream<Arguments> thresholds() {
        return Stream.of(
            Arguments.of("below every bound", 0.01f, Decision.SCORE, Decision.SCORE),
            Arguments.of("above the far block's bound only", 0.1f, Decision.SCORE, Decision.TERMINATE),
            Arguments.of("above both bounds", 0.5f, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of("exactly a bound", BLOCK_1_BOUND, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of("just under a bound", Math.nextDown(BLOCK_1_BOUND), Decision.SCORE, Decision.TERMINATE)
        );
    }

    /**
     * A shell the query sits outside of bounds every block from there on, so the decision is TERMINATE and never
     * SKIP. The threshold equal to a bound terminates too: the scan keeps a block only on a strictly better score.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("thresholds")
    void testTest_whenAShellIsTooFarToCompete_thenTerminates(
        String description,
        float minCompetitiveScore,
        Decision expectedForBlock1,
        Decision expectedForBlock2
    ) {
        // given
        BlockPostingsPruner pruner = new EuclideanClipPruner(SORTED_DISTANCES, BLOCK_SIZE, QUERY_TO_CENTROID);

        // then
        assertEquals(expectedForBlock1, pruner.test(1, minCompetitiveScore));
        assertEquals(expectedForBlock2, pruner.test(2, minCompetitiveScore));
    }

    /** The nearest shell is bounded too, as soon as it has passed the query: at 1 against a query at 0.5, by 0.5. */
    @Test
    void testTest_whenTheNearestShellHasPassedTheQuery_thenBoundsItToo() {
        // given
        BlockPostingsPruner pruner = new EuclideanClipPruner(SORTED_DISTANCES, BLOCK_SIZE, QUERY_TO_CENTROID);
        float blockZeroBound = 1f / (1f + 0.5f * 0.5f);

        // then
        assertEquals(Decision.SCORE, pruner.test(0, Math.nextDown(blockZeroBound)));
        assertEquals(Decision.TERMINATE, pruner.test(0, blockZeroBound));
    }

    /**
     * The inner arm: a block wholly inside the query's shell is bounded by its <em>farthest</em> member, the one
     * closest to the query's shell. It skips rather than terminates, because the blocks after it come closer.
     */
    @Test
    void testTest_whenABlockSitsWhollyInsideTheQuerysShell_thenSkipsIt() {
        // given — a query at 3, beyond blocks 0 and 1 (farthest 1.265 and 2.145) but inside block 2 (nearest 4)
        BlockPostingsPruner pruner = new EuclideanClipPruner(SORTED_DISTANCES, BLOCK_SIZE, 3f);
        float blockZeroBound = bound(3f - (float) Math.sqrt(SORTED_DISTANCES[3]));
        float blockOneBound = bound(3f - (float) Math.sqrt(SORTED_DISTANCES[7]));

        // then — each inner block is pruned at its own bound, and the nearer one binds more loosely
        assertEquals(Decision.SKIP, pruner.test(0, blockZeroBound));
        assertEquals(Decision.SCORE, pruner.test(0, Math.nextDown(blockZeroBound)));
        assertEquals(Decision.SKIP, pruner.test(1, blockOneBound));
        assertEquals(Decision.SCORE, pruner.test(1, Math.nextDown(blockOneBound)));
        assertTrue(blockZeroBound < blockOneBound, "the block further inside must bound more tightly");
    }

    /**
     * A block whose members lie on both sides of the query's shell bounds nothing at all: one of them could be
     * touching the query however high the bar is set.
     */
    @Test
    void testTest_whenABlockStraddlesTheQuerysShell_thenAlwaysScores() {
        // given — a query at 2.1, between block 1's nearest (2) and its farthest (2.145)
        BlockPostingsPruner pruner = new EuclideanClipPruner(SORTED_DISTANCES, BLOCK_SIZE, 2.1f);

        // then
        assertEquals(Decision.SCORE, pruner.test(1, 0.9f));
        assertEquals(Decision.SCORE, pruner.test(1, 1.0f));
    }

    /** The two arms in one walk: the blocks inside the query's shell skip, and the first one past it terminates. */
    @Test
    void testTest_whenTheWalkCrossesTheQuerysShell_thenSkipsInsideAndTerminatesOutside() {
        // given — a query at 3 with a bar above every block's bound
        BlockPostingsPruner pruner = new EuclideanClipPruner(SORTED_DISTANCES, BLOCK_SIZE, 3f);

        // then
        assertEquals(Decision.SKIP, pruner.test(0, 0.9f));
        assertEquals(Decision.SKIP, pruner.test(1, 0.9f));
        assertEquals(Decision.TERMINATE, pruner.test(2, 0.9f));
    }

    /** The bound has to fall as the walk moves outwards, or TERMINATE would be unsound. */
    @Test
    void testTest_thenTerminatesNoLaterThanTheBlockBefore() {
        // given
        BlockPostingsPruner pruner = new EuclideanClipPruner(SORTED_DISTANCES, BLOCK_SIZE, QUERY_TO_CENTROID);

        // when — a threshold that clips block 1 must clip block 2, which sits further out
        Decision nearer = pruner.test(1, BLOCK_1_BOUND);
        Decision further = pruner.test(2, BLOCK_1_BOUND);

        // then
        assertEquals(Decision.TERMINATE, nearer);
        assertEquals(Decision.TERMINATE, further);
        assertEquals(Decision.SCORE, pruner.test(1, Math.nextDown(BLOCK_2_BOUND)), "the far bound must not clip the near block");
    }

    /** The best score a member exactly {@code gap} from the query could get, as the pruner computes it. */
    private static float bound(float gap) {
        return 1f / (1f + gap * gap);
    }
}
