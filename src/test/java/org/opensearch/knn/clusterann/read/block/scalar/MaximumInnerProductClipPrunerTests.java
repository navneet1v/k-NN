/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner.Decision;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MaximumInnerProductClipPrunerTests {

    private static final int BLOCK_SIZE = 4;
    private static final int NUM_BLOCKS = 3;

    /**
     * Squared distances in member order, as an inner-product posting stores them: <b>descending</b>, farthest
     * from the centroid first. Each block's first entry is the one the pruner reads — shells at 4, 2 and 1.
     */
    private static final float[] SORTED_DISTANCES = { 16f, 15.8f, 15.6f, 15.4f, 4f, 3.8f, 3.6f, 3.4f, 1f, 0.8f };

    private static final float QUERY_DOT_CENTROID = 0.25f;
    private static final float QUERY_NORM = 2f;

    /** {@code ⟨q,c⟩ + ‖q‖·r}, scaled the way the scorer scales a dot: radii here are 4, 2 and 1. */
    private static float bound(final float radius) {
        final float dot = QUERY_DOT_CENTROID + QUERY_NORM * radius;
        return dot >= 0 ? dot + 1 : 1f / (1f - dot);
    }

    private static final float BLOCK_0_BOUND = bound(4f);  // 0.25 + 8 -> 9.25
    private static final float BLOCK_1_BOUND = bound(2f);  // 0.25 + 4 -> 5.25
    private static final float BLOCK_2_BOUND = bound(1f);  // 0.25 + 2 -> 3.25

    private static MaximumInnerProductClipPruner pruner() {
        return new MaximumInnerProductClipPruner(SORTED_DISTANCES, BLOCK_SIZE, QUERY_DOT_CENTROID, QUERY_NORM);
    }

    private static Stream<Arguments> thresholds() {
        return Stream.of(
            Arguments.of("below every ceiling", 1f, Decision.SCORE, Decision.SCORE, Decision.SCORE),
            Arguments.of("above the nearest block's ceiling only", 4f, Decision.SCORE, Decision.SCORE, Decision.TERMINATE),
            Arguments.of("above the two nearer ceilings", 6f, Decision.SCORE, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of("above every ceiling", 20f, Decision.TERMINATE, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of("exactly a ceiling", BLOCK_1_BOUND, Decision.SCORE, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of("just under a ceiling", Math.nextDown(BLOCK_1_BOUND), Decision.SCORE, Decision.SCORE, Decision.TERMINATE)
        );
    }

    /**
     * The ceiling falls as the walk goes on, so a block that cannot compete ends the posting — the decision is
     * TERMINATE and never SKIP. A threshold exactly equal to a ceiling terminates too: the scan keeps a block
     * only on a score strictly above the bar.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("thresholds")
    void testTest_whenAShellCannotCompete_thenTerminates(
        final String description,
        final float threshold,
        final Decision block0,
        final Decision block1,
        final Decision block2
    ) {
        final MaximumInnerProductClipPruner pruner = pruner();
        assertEquals(block0, pruner.test(0, threshold), "block 0");
        assertEquals(block1, pruner.test(1, threshold), "block 1");
        assertEquals(block2, pruner.test(2, threshold), "block 2");
    }

    /** SKIP would be unsound here: the bound is monotone, so there is never a later block worth going on for. */
    @Test
    void testTest_thenNeverSkips() {
        final MaximumInnerProductClipPruner pruner = pruner();
        for (float threshold = -5f; threshold < 20f; threshold += 0.25f) {
            for (int block = 0; block < NUM_BLOCKS; block++) {
                assertTrue(pruner.test(block, threshold) != Decision.SKIP, "SKIP at block " + block + ", threshold " + threshold);
            }
        }
    }

    /**
     * The property the TERMINATE rests on: farthest-first storage makes the ceiling non-increasing in the block
     * index, so cutting the tail cannot discard a block that would have bounded higher than the one that failed.
     */
    @Test
    void testTest_thenTheCeilingNeverRisesAlongTheWalk() {
        assertTrue(BLOCK_0_BOUND > BLOCK_1_BOUND, "block 0 must bound above block 1");
        assertTrue(BLOCK_1_BOUND > BLOCK_2_BOUND, "block 1 must bound above block 2");

        // Read through the pruner itself: any threshold that terminates a block must terminate every later one.
        final MaximumInnerProductClipPruner pruner = pruner();
        for (float threshold = -5f; threshold < 20f; threshold += 0.1f) {
            boolean terminated = false;
            for (int block = 0; block < NUM_BLOCKS; block++) {
                final Decision decision = pruner.test(block, threshold);
                if (terminated) {
                    assertEquals(Decision.TERMINATE, decision, "block " + block + " after a terminate, threshold " + threshold);
                }
                terminated |= decision == Decision.TERMINATE;
            }
        }
    }

    /**
     * A query whose dot with the centroid is negative still bounds: the scorer's scaling folds negatives into
     * {@code (0,1]} rather than going negative, and the pruner has to land in the same space to be comparable.
     */
    @Test
    void testTest_whenTheCentroidDotIsNegative_thenBoundsInTheScorersSpace() {
        final MaximumInnerProductClipPruner negative = new MaximumInnerProductClipPruner(
            new float[] { 0.01f, 0.01f, 0.01f, 0.01f },
            BLOCK_SIZE,
            -10f,
            0.5f
        );
        // dot = -10 + 0.5*0.1 = -9.95 -> 1/(1+9.95) = 0.0913, which is positive and below any real threshold.
        assertEquals(Decision.TERMINATE, negative.test(0, 0.5f), "a far-from-query shell cannot compete");
        assertEquals(Decision.SCORE, negative.test(0, 0.01f), "but it still beats a bar below its ceiling");
    }
}
