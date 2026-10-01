/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner.Decision;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class BlockPostingsPrunerTests {

    private static final float ANY_THRESHOLD = 0.5f;

    /** Every pairing of two decisions, and what the two together have to mean. */
    private static Stream<Arguments> pairs() {
        return Stream.of(
            Arguments.of(Decision.SCORE, Decision.SCORE, Decision.SCORE),
            Arguments.of(Decision.SCORE, Decision.SKIP, Decision.SKIP),
            Arguments.of(Decision.SKIP, Decision.SCORE, Decision.SKIP),
            Arguments.of(Decision.SKIP, Decision.SKIP, Decision.SKIP),
            Arguments.of(Decision.SCORE, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of(Decision.TERMINATE, Decision.SCORE, Decision.TERMINATE),
            Arguments.of(Decision.SKIP, Decision.TERMINATE, Decision.TERMINATE),
            Arguments.of(Decision.TERMINATE, Decision.SKIP, Decision.TERMINATE)
        );
    }

    @ParameterizedTest(name = "{0} and {1} is {2}")
    @MethodSource("pairs")
    void testTest_whenPrunersDisagree_thenTheStrongestDecisionWins(Decision first, Decision second, Decision expected) {
        // given
        BlockPostingsPruner pruner = BlockPostingsPruner.of(always(first), always(second));

        // when
        Decision decision = pruner.test(0, ANY_THRESHOLD);

        // then
        assertEquals(expected, decision);
    }

    /**
     * A SKIP must not short-circuit: the pruners after it still get asked, because losing a TERMINATE would cost
     * the whole tail of the posting, one skipped block at a time.
     */
    @Test
    void testTest_whenAnEarlyPrunerSkips_thenTheLaterOnesAreStillAsked() {
        // given
        List<String> asked = new ArrayList<>();
        BlockPostingsPruner pruner = BlockPostingsPruner.of(
            recording(asked, "first", Decision.SKIP),
            recording(asked, "second", Decision.SCORE)
        );

        // when
        pruner.test(0, ANY_THRESHOLD);

        // then
        assertEquals(List.of("first", "second"), asked);
    }

    /** TERMINATE outranks everything, so there is nothing left to ask. */
    @Test
    void testTest_whenAnEarlyPrunerTerminates_thenTheLaterOnesAreNotAsked() {
        // given
        List<String> asked = new ArrayList<>();
        BlockPostingsPruner pruner = BlockPostingsPruner.of(
            recording(asked, "first", Decision.TERMINATE),
            recording(asked, "second", Decision.SCORE)
        );

        // when
        pruner.test(0, ANY_THRESHOLD);

        // then
        assertEquals(List.of("first"), asked);
    }

    /** Nothing to prune on means no composite and no array walk per block. */
    @Test
    void testOf_whenNothingPrunes_thenCollapsesToNone() {
        assertSame(BlockPostingsPruner.NONE, BlockPostingsPruner.of());
        assertSame(BlockPostingsPruner.NONE, BlockPostingsPruner.of(BlockPostingsPruner.NONE, null, BlockPostingsPruner.NONE));
    }

    @Test
    void testOf_whenOnlyOnePrunes_thenHandsBackThatPruner() {
        // given
        BlockPostingsPruner only = always(Decision.SKIP);

        // when
        BlockPostingsPruner pruner = BlockPostingsPruner.of(BlockPostingsPruner.NONE, only, null);

        // then
        assertSame(only, pruner);
    }

    private static BlockPostingsPruner always(Decision decision) {
        return (block, minCompetitiveScore) -> decision;
    }

    private static BlockPostingsPruner recording(List<String> asked, String name, Decision decision) {
        return (block, minCompetitiveScore) -> {
            asked.add(name);
            return decision;
        };
    }
}
