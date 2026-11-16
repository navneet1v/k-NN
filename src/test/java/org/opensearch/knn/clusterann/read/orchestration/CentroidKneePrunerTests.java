/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link CentroidKneePruner}.
 *
 * <p>Bar = factor x tail rise / tail gaps, the tail starting at rank {@code floor-1}. Unfiltered (factor 3) unless stated.
 */
class CentroidKneePrunerTests {

    private static final double UNFILTERED = 1.0;

    // ---------------------------------------------------------------- cutting

    /**
     * Rise 9999 over 5 gaps bars at 5999.4. The gaps are 3, 5, 7, 9 and 9975: the first four are the query's own
     * neighbourhood and the last is the rest of the index, which is where the plan ends.
     */
    @Test
    void testPrune_whenTheCurveBreaks_thenCutsAtTheBreak() {
        // given
        float[] distances = { 1f, 4f, 9f, 16f, 25f, 10000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(5, keep, "everything up to the cliff, and nothing past it");
    }

    /**
     * The cut keeps the clusters before the gap, not after it: a break between index 1 and 2 keeps two. Rise 300 over
     * 5 gaps bars at 180, and the 294 gap is the only one to clear it.
     */
    @Test
    void testPrune_thenKeepsTheClustersBeforeTheGap() {
        // given
        float[] distances = { 1f, 2f, 296f, 297f, 298f, 301f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(2, keep);
    }

    /** Two breaks: the plan ends at the first. Rise 207 over 9 gaps, bar 69. */
    @Test
    void testPrune_whenTheCurveBreaksTwice_thenCutsAtTheFirst() {
        // given
        float[] distances = { 0f, 1f, 101f, 102f, 103f, 104f, 204f, 205f, 206f, 207f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(2, keep, "the second break is past the end of the plan");
    }

    /**
     * The key is a negated similarity under MAXIMUM_INNER_PRODUCT and COSINE, so a curve can be entirely negative.
     * The cut reads gaps, which are differences, so it does not care where zero is.
     */
    @Test
    void testPrune_whenTheCurveIsNegative_thenCutsTheSameWay() {
        // given — the shape of the breaking curve above, shifted below zero
        float[] distances = { -10000f, -9997f, -9992f, -9985f, -9976f, -1f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(5, keep);
    }

    // ---------------------------------------------------------------- not cutting

    /**
     * A smooth curve has no knee and keeps everything. These gaps are 2, 4, 6, 8 and 10 — rising the whole way, and
     * the largest still under the 18 the rise of 30 bars at, because growing steadily is not the same as breaking.
     */
    @Test
    void testPrune_whenTheCurveIsSmooth_thenKeepsEverything() {
        // given
        float[] distances = { 1f, 3f, 7f, 13f, 21f, 31f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    /** Equidistant centroids give a mean gap of zero to compare against, so nothing stands out and nothing is cut. */
    @Test
    void testPrune_whenTheCurveIsFlat_thenKeepsEverything() {
        // given
        float[] distances = { 5f, 5f, 5f, 5f, 5f, 5f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    /** Three gaps bar at the whole rise, which no one gap exceeds. */
    @Test
    void testPrune_whenFewerThanFiveRanked_thenCannotCut() {
        // given — as sharp a cliff as a four-point curve can carry
        float[] distances = { 1f, 2f, 3f, 100000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    // ---------------------------------------------------------------- the floor

    /** Floor 3: tail rise 62 over 7 gaps, bar 26.6; the 56 gap cuts. The head gaps are not in the bar. */
    @Test
    void testPrune_whenTheTailBreaks_thenCutsBetweenTheFloorAndTheBudget() {
        // given
        float[] distances = { 0f, 100f, 200f, 201f, 202f, 203f, 204f, 260f, 261f, 262f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3, UNFILTERED);

        // then
        assertEquals(7, keep, "the tail's own knee, above the floor and below the ten ranked");
    }

    /** An even tail past a steep head does not cut. */
    @Test
    void testPrune_whenTheTailIsFlat_thenKeepsTheBudget() {
        // given: tail gaps all 1, rise 7 over 7 gaps, bar 3
        float[] distances = { 0f, 100f, 200f, 201f, 202f, 203f, 204f, 205f, 206f, 207f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    /** A break inside the floor is not a knee: floor 3, tail gaps all 1. */
    @Test
    void testPrune_whenTheOnlyBreakIsInsideTheFloor_thenIgnoresIt() {
        // given
        float[] distances = { 0f, 1f, 1000f, 1001f, 1002f, 1003f, 1004f, 1005f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    /** Floor 3: tail rise 105 over 8 gaps, bar 39.4; the 98 gap right after the floor keeps exactly the floor. */
    @Test
    void testPrune_whenTheKneeIsAtTheFloor_thenCutsToTheFloor() {
        // given
        float[] distances = { 0f, 1f, 2f, 100f, 101f, 102f, 103f, 104f, 105f, 106f, 107f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3, UNFILTERED);

        // then
        assertEquals(3, keep, "the cut lands on the floor, not above it");
    }

    /** Floor 2: tail rise 9996 over 6 gaps, bar 4998; only the 9951 gap clears it. */
    @Test
    void testPrune_whenTheKneeIsAboveTheFloor_thenCutsAtTheKnee() {
        // given
        float[] distances = { 1f, 4f, 9f, 16f, 25f, 36f, 49f, 10000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 2, UNFILTERED);

        // then
        assertEquals(7, keep);
    }

    /** A tail of three gaps cannot cut, however sharp. */
    @Test
    void testPrune_whenTheTailHasTooFewGaps_thenCannotCut() {
        // given: floor 3, tail gaps 1, 1, 99996
        float[] distances = { 1f, 2f, 3f, 4f, 5f, 100001f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    // ---------------------------------------------------------------- the factor

    /** The 5 gap is 3.57 mean gaps (rise 14 over 10): it cuts unfiltered (factor 3), not at s = 0.10 (factor 4). */
    @Test
    void testPrune_whenAGapIsBetweenThreeAndFourMeanGaps_thenOnlyFactorThreeCuts() {
        // given
        float[] distances = { 0f, 1f, 2f, 3f, 4f, 9f, 10f, 11f, 12f, 13f, 14f };

        // when
        int atThree = CentroidKneePruner.prune(distances, 1, UNFILTERED);
        int atFour = CentroidKneePruner.prune(distances, 1, 0.10);

        // then
        assertEquals(5, atThree);
        assertEquals(distances.length, atFour);
    }

    // ---------------------------------------------------------------- the factor curve

    @Test
    void testKneeFactor_whenAQuarterOrMore_thenThree() {
        assertEquals(3f, CentroidKneePruner.kneeFactor(1.0));
        assertEquals(3f, CentroidKneePruner.kneeFactor(0.5));
        assertEquals(3f, CentroidKneePruner.kneeFactor(0.25));
    }

    @Test
    void testKneeFactor_whenUnderAQuarter_thenRisesWithTheCurve() {
        assertEquals(4.00f, CentroidKneePruner.kneeFactor(0.10), 0.01f);
        assertEquals(4.97f, CentroidKneePruner.kneeFactor(0.05), 0.01f);
    }

    @Test
    void testKneeFactor_whenVerySelective_thenCapsAtEight() {
        assertEquals(8f, CentroidKneePruner.kneeFactor(0.01));
        assertEquals(8f, CentroidKneePruner.kneeFactor(0.001));
    }

    @Test
    void testKneeFactor_whenNotPositive_thenCapsAtEight() {
        assertEquals(8f, CentroidKneePruner.kneeFactor(0.0));
        assertEquals(8f, CentroidKneePruner.kneeFactor(-0.5));
        assertEquals(8f, CentroidKneePruner.kneeFactor(Double.NaN));
    }

    @Test
    void testKneeFactor_isMonotoneNonIncreasingInSelectivity() {
        float previous = CentroidKneePruner.kneeFactor(0.0);
        for (int i = 1; i <= 10_000; i++) {
            final double selectivity = i / 10_000.0;
            final float factor = CentroidKneePruner.kneeFactor(selectivity);
            assertTrue(factor <= previous, "rose at s = " + selectivity);
            previous = factor;
        }
    }

    /** Nothing ranked beyond the floor leaves no room to cut in, and the curve is not even looked at. */
    @Test
    void testPrune_whenTheFloorCoversEverythingRanked_thenKeepsEverything() {
        // given
        float[] distances = { 1f, 2f, 3f, 100000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 4, UNFILTERED);

        // then
        assertEquals(distances.length, keep);
    }

    /** A field with one occupied cluster is below any floor, so it is returned whole. */
    @Test
    void testPrune_whenOneRanked_thenKeepsIt() {
        // when
        int keep = CentroidKneePruner.prune(new float[] { 7f }, 1, UNFILTERED);

        // then
        assertEquals(1, keep);
    }

    /** A field with no occupied cluster has no curve and no plan. */
    @Test
    void testPrune_whenNoneRanked_thenKeepsNothing() {
        // when
        int keep = CentroidKneePruner.prune(new float[0], 1, UNFILTERED);

        // then
        assertEquals(0, keep);
    }
}
