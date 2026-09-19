/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * The one {@link OptimizedScalarQuantizer} per similarity, shared by every field, cluster and query in the JVM.
 *
 * <p>A quantizer carries nothing but its similarity and two tuning constants, all final, so one instance answers every
 * caller and sharing needs no coordination. Building them per field — as each reader and writer would otherwise do —
 * allocates one object per similarity per segment for no gain, and leaves each construction site free to disagree about
 * lambda and the iteration count, which would quantize the same vector two ways.
 *
 * <p>Lives outside the read and write packages because both sides quantize against the same parameters: a query
 * quantized differently from the codes it is scored against is silently wrong rather than broken.
 *
 * <p>Every similarity is populated eagerly, so {@link #forSimilarity} is a map lookup that cannot fail for any member
 * of the enum and never has to decide whether to construct.
 */
public final class ScalarQuantizers {

    /**
     * One per similarity, built once at class initialisation. Unmodifiable, and the values are immutable, so the map is
     * safe to publish as a constant.
     */
    private static final Map<VectorSimilarityFunction, OptimizedScalarQuantizer> BY_SIMILARITY = buildQuantizers();

    private ScalarQuantizers() {}

    /**
     * The quantizer for one similarity.
     *
     * @param similarity the field's similarity function
     * @return the shared quantizer; never {@code null} for any {@link VectorSimilarityFunction}
     * @throws NullPointerException if {@code similarity} is {@code null}, since there is no quantizer that could
     *     stand in for an unknown similarity and returning one for a different metric would score silently wrong
     */
    public static OptimizedScalarQuantizer forSimilarity(final VectorSimilarityFunction similarity) {
        final OptimizedScalarQuantizer quantizer = BY_SIMILARITY.get(similarity);
        if (quantizer == null) {
            throw new NullPointerException("No quantizer for similarity: " + similarity);
        }
        return quantizer;
    }

    private static Map<VectorSimilarityFunction, OptimizedScalarQuantizer> buildQuantizers() {
        final EnumMap<VectorSimilarityFunction, OptimizedScalarQuantizer> quantizers = new EnumMap<>(VectorSimilarityFunction.class);
        for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
            quantizers.put(similarity, new OptimizedScalarQuantizer(similarity));
        }
        return Collections.unmodifiableMap(quantizers);
    }
}
