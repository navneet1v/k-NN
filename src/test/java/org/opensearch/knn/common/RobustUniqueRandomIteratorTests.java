/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.knn.KNNTestCase;

import java.util.HashSet;
import java.util.Set;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class RobustUniqueRandomIteratorTests extends KNNTestCase {

    // --- Constructor validation edge cases ---

    public void testConstructorThrowsWhenMaxValExclusiveIsZero() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(0, 0, 1L));
    }

    public void testConstructorThrowsWhenMaxValExclusiveIsNegative() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(-1, 0, 1L));
    }

    public void testConstructorThrowsWhenNumPopulateIsNegative() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(10, -1, 1L));
    }

    public void testConstructorThrowsWhenNumPopulateExceedsMaxVal() {
        expectThrows(IllegalArgumentException.class, () -> new RobustUniqueRandomIterator(5, 6, 1L));
    }

    // --- Zero populate ---

    public void testZeroPopulateReturnsNoMoreDocsImmediately() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(10, 0, 1L);
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- maxValExclusive = 1 (single element) ---

    public void testSingleElementRange() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(1, 1, 1L);
        assertEquals(0, iter.next());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Power-of-two range (no rejection needed) ---

    public void testPowerOfTwoRange() {
        int maxVal = 8;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, maxVal, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < maxVal; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Non-power-of-two range (rejection sampling exercised) ---

    public void testNonPowerOfTwoRange() {
        int maxVal = 77_777;
        int numPopulate = 1000;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < numPopulate; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Full exhaustion: numPopulate == maxValExclusive ---

    public void testFullExhaustionSmallRange() {
        int maxVal = 100;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, maxVal, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < maxVal; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals("Should have all values", maxVal, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Partial sampling: numPopulate < maxValExclusive ---

    public void testPartialSampling() {
        int maxVal = 500;
        int numPopulate = 50;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < numPopulate; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(numPopulate, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- maxValExclusive = 2 (smallest non-trivial range) ---

    public void testRangeOfTwo() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(2, 2, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 2; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < 2);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- numPopulate == maxValExclusive boundary ---

    public void testNumPopulateEqualsMaxVal() {
        int maxVal = 10;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, maxVal, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < maxVal; i++) {
            int val = iter.next();
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(maxVal, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Multiple calls past exhaustion still return NO_MORE_DOCS ---

    public void testRepeatedCallsAfterExhaustion() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(5, 2, 1L);
        iter.next();
        iter.next();
        assertEquals(NO_MORE_DOCS, iter.next());
        assertEquals(NO_MORE_DOCS, iter.next());
        assertEquals(NO_MORE_DOCS, iter.next());
    }

    // --- Large range uniqueness stress test ---

    public void testLargeRangeUniqueness() {
        int maxVal = 100_000;
        int numPopulate = 10_000;
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < numPopulate; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < maxVal);
            assertTrue("Duplicate at iteration " + i + ": " + val, seen.add(val));
        }
        assertEquals(numPopulate, seen.size());
    }

    // --- Determinism: same seed always produces same sequence ---

    public void testSameSeedProducesSameSequence() {
        int maxVal = 10_000;
        int numPopulate = 100;
        long seed = 123456789L;

        int[] firstRun = new int[numPopulate];
        RobustUniqueRandomIterator iter1 = new RobustUniqueRandomIterator(maxVal, numPopulate, seed);
        for (int i = 0; i < numPopulate; i++) {
            firstRun[i] = iter1.next();
        }

        for (int run = 0; run < 100; run++) {
            RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(maxVal, numPopulate, seed);
            for (int i = 0; i < numPopulate; i++) {
                assertEquals("Mismatch at index " + i + " on run " + run, firstRun[i], iter.next());
            }
        }
    }

    // --- Different seeds produce different sequences ---

    public void testDifferentSeedsProduceDifferentSequences() {
        int maxVal = 10_000;
        int numPopulate = 50;

        RobustUniqueRandomIterator iter1 = new RobustUniqueRandomIterator(maxVal, numPopulate, 42L);
        RobustUniqueRandomIterator iter2 = new RobustUniqueRandomIterator(maxVal, numPopulate, 43L);

        boolean anyDifferent = false;
        for (int i = 0; i < numPopulate; i++) {
            if (iter1.next() != iter2.next()) {
                anyDifferent = true;
                break;
            }
        }
        assertTrue("Different seeds should produce different sequences", anyDifferent);
    }

    // --- Boundary: maxValExclusive = 3 (non-power-of-two, small) ---

    public void testSmallNonPowerOfTwo() {
        RobustUniqueRandomIterator iter = new RobustUniqueRandomIterator(3, 3, 1L);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 3; i++) {
            int val = iter.next();
            assertTrue("Value out of range: " + val, val >= 0 && val < 3);
            assertTrue("Duplicate value: " + val, seen.add(val));
        }
        assertEquals(3, seen.size());
        assertEquals(NO_MORE_DOCS, iter.next());
    }
}
