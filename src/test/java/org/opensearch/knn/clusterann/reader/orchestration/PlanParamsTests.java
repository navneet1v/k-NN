/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.orchestration;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link PlanParams} (JUnit 5).
 *
 * <p>The bounds are only two numbers, so what there is to cover is the arithmetic that derives them and the invariant
 * that keeps them usable: a range with the floor above the ceiling has no answer in it.
 */
class PlanParamsTests {

    // ---------------------------------------------------------------- derived bounds

    /**
     * Both bounds are built from the integer √centroids, so they track the field rather than a fixed guess. Expressed
     * in terms of the constants rather than in absolute numbers: these are tuning knobs, and retuning one should not
     * mean rewriting the test that says what it means. The roots are the literals, and they are truncated, not rounded.
     */
    @ParameterizedTest(name = "{0} clusters, root {1}")
    @CsvSource({
        "10000, 100",
        "1000, 31",     // sqrt(1000) is 31.6, truncated
        "100, 10",
        "36, 6" })
    void testOf_thenScalesBothBoundsToTheRootOfTheClusterCount(int numCentroids, int root) {
        // given / when
        PlanParams params = PlanParams.of(numCentroids);

        // then
        assertEquals(PlanParams.NPROBE_MULTIPLIER * root, params.maxProbes());
        assertEquals(Math.max(PlanParams.MIN_NPROBE, root), params.minProbes());
    }

    /** The ceiling cannot exceed the field: a multiple of the root overshoots once the field is small enough. */
    @Test
    void testOf_whenTheFieldIsSmallerThanItsShare_thenTheCeilingIsTheField() {
        // given / when — root 2, so the ceiling would be a multiple of 2 against only 4 clusters
        PlanParams params = PlanParams.of(4);

        // then
        assertEquals(4, params.maxProbes());
    }

    /** The floor gives way to the ceiling: on a small field there are not {@code MIN_NPROBE} clusters to insist on. */
    @Test
    void testOf_whenTheFieldHasFewerClustersThanTheFloor_thenTheFloorGivesWay() {
        // given / when
        PlanParams params = PlanParams.of(4);

        // then
        assertEquals(params.maxProbes(), params.minProbes());
        assertTrue(params.minProbes() < PlanParams.MIN_NPROBE, "the floor cannot ask for clusters that do not exist");
    }

    /** No clusters: the planner returns before it uses these, but they still have to describe a valid range. */
    @Test
    void testOf_whenTheFieldHasNoClusters_thenTheBoundsAreStillValid() {
        // given / when
        PlanParams params = PlanParams.of(0);

        // then
        assertEquals(1, params.minProbes());
        assertEquals(1, params.maxProbes());
    }

    /** The ceiling never exceeds the field, and the floor never exceeds the ceiling — at any cluster count. */
    @ParameterizedTest(name = "{0} clusters")
    @CsvSource({ "0", "1", "2", "9", "10", "99", "1000", "1048576", "2147483647" })
    void testOf_thenTheBoundsAreAlwaysUsable(int numCentroids) {
        // given / when
        PlanParams params = PlanParams.of(numCentroids);

        // then
        assertTrue(params.minProbes() >= 1, "minProbes " + params.minProbes());
        assertTrue(params.minProbes() <= params.maxProbes(), params.minProbes() + " > " + params.maxProbes());
        assertTrue(params.maxProbes() <= Math.max(1, numCentroids), "maxProbes " + params.maxProbes());
    }

    // ---------------------------------------------------------------- invariants

    /** A plan names at least one cluster, so a floor below one describes nothing. */
    @ParameterizedTest(name = "minProbes {0}")
    @CsvSource({ "0", "-1", "-100" })
    void testConstruction_whenTheFloorIsBelowOne_thenThrows(int minProbes) {
        // given / when
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new PlanParams(minProbes, 10));

        // then
        assertTrue(e.getMessage().contains("minProbes must be at least 1"), e.getMessage());
    }

    /** A ceiling under the floor leaves no count that satisfies both, so it is rejected rather than silently resolved. */
    @Test
    void testConstruction_whenTheCeilingIsBelowTheFloor_thenThrows() {
        // given / when
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new PlanParams(8, 4));

        // then
        assertTrue(e.getMessage().contains("maxProbes=4 is below minProbes=8"), e.getMessage());
    }

    /** A single permitted count is a legitimate range: it fixes the probe count outright. */
    @Test
    void testConstruction_whenTheBoundsAreEqual_thenAccepted() {
        // given / when
        PlanParams params = new PlanParams(6, 6);

        // then
        assertEquals(6, params.minProbes());
        assertEquals(6, params.maxProbes());
    }
}
