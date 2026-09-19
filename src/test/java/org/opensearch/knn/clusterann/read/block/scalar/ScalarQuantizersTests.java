/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Fast, isolated unit tests for {@link ScalarQuantizers} (JUnit 5).
 *
 * <p>What matters here is that the sharing is real and that it is per similarity — a single instance handed to
 * everything would quantize against the wrong metric, and a fresh instance per call would defeat the point.
 */
class ScalarQuantizersTests {

    /** Every member of the enum is served, so no field can name a similarity the read path cannot quantize for. */
    @ParameterizedTest(name = "{0}")
    @EnumSource(VectorSimilarityFunction.class)
    void testForSimilarity_thenServesEverySimilarity(VectorSimilarityFunction similarity) {
        assertNotNull(ScalarQuantizers.forSimilarity(similarity));
    }

    /**
     * The point of the class: one instance per similarity, not one per caller. A quantizer holds only its similarity
     * and two tuning constants, all final, so every field and query can share it.
     */
    @ParameterizedTest(name = "{0}")
    @EnumSource(VectorSimilarityFunction.class)
    void testForSimilarity_whenAskedTwice_thenHandsOutTheSameInstance(VectorSimilarityFunction similarity) {
        assertSame(ScalarQuantizers.forSimilarity(similarity), ScalarQuantizers.forSimilarity(similarity));
    }

    /**
     * Per similarity rather than one for all: a quantizer is bound to the metric it was built with, so serving one
     * similarity's quantizer for another would score against the wrong space and never say so.
     */
    @Test
    void testForSimilarity_thenEachSimilarityGetsItsOwn() {
        // given / when
        VectorSimilarityFunction[] similarities = VectorSimilarityFunction.values();

        // then — every pair differs, which also catches a map that collapsed to a single entry
        for (int i = 0; i < similarities.length; i++) {
            for (int j = i + 1; j < similarities.length; j++) {
                assertNotSame(
                    ScalarQuantizers.forSimilarity(similarities[i]),
                    ScalarQuantizers.forSimilarity(similarities[j]),
                    similarities[i] + " and " + similarities[j] + " must not share a quantizer"
                );
            }
        }
    }

    /**
     * A null similarity is refused rather than answered: there is no quantizer that stands in for an unknown metric,
     * and handing back any of them would be silently wrong instead of loudly broken.
     */
    @Test
    void testForSimilarity_whenTheSimilarityIsNull_thenThrows() {
        assertThrows(NullPointerException.class, () -> ScalarQuantizers.forSimilarity(null));
    }

    /** The quantizer served is the one Lucene builds for that similarity, at Lucene's own defaults. */
    @Test
    void testForSimilarity_thenMatchesLucenesDefaultConstruction() {
        // given
        OptimizedScalarQuantizer shared = ScalarQuantizers.forSimilarity(VectorSimilarityFunction.EUCLIDEAN);

        // when — quantize the same vector both ways
        float[] viaShared = { 1f, 2f, 3f, 4f };
        float[] viaFresh = { 1f, 2f, 3f, 4f };
        float[] centroid = { 0.5f, 0.5f, 0.5f, 0.5f };
        byte[] sharedCodes = new byte[4];
        byte[] freshCodes = new byte[4];

        OptimizedScalarQuantizer.QuantizationResult sharedResult = shared.scalarQuantize(viaShared, sharedCodes, (byte) 4, centroid);
        OptimizedScalarQuantizer.QuantizationResult freshResult = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN)
            .scalarQuantize(viaFresh, freshCodes, (byte) 4, centroid);

        // then
        assertArrayEqualsExactly(freshCodes, sharedCodes);
        assertEquals(freshResult.lowerInterval(), sharedResult.lowerInterval());
        assertEquals(freshResult.upperInterval(), sharedResult.upperInterval());
        assertEquals(freshResult.quantizedComponentSum(), sharedResult.quantizedComponentSum());
        assertEquals(freshResult.additionalCorrection(), sharedResult.additionalCorrection());
    }

    private static void assertArrayEqualsExactly(byte[] expected, byte[] actual) {
        assertEquals(expected.length, actual.length, "code length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], "code byte " + i);
        }
    }
}
