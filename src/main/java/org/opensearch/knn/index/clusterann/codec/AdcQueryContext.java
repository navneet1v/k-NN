/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Per-posting query state: the query quantized <em>relative to this posting's centroid</em> at the
 * width the query asked for and transposed once, plus the scalar factors and centroid terms the
 * corrective math needs.
 *
 * <p>Quantizing the query against the centroid is the expensive per-posting step, so it is done once
 * here and shared by everything that scores or bounds this posting — the {@link ADCBlockScorer}
 * (per-vector similarity) and the {@link AdcCorrectionsPruner} (per-block upper bound). Both derive
 * their numbers from these same factors, so neither re-quantizes.
 *
 * <p>The query width is a per-query knob ({@link ScanParams#queryBits()}), not a property of the index:
 * wider than the doc codes is asymmetric (ADC), equal is symmetric (SDC). Only 4-bit query
 * quantization has a dot kernel today; the rest are rejected rather than silently mis-scored.
 */
final class AdcQueryContext {

    private static final int SUPPORTED_QUERY_BITS = 4;

    final byte[] transposedQuery;        // 4-bit query, half-byte transposed (for the bit-dot)
    final float queryLower;
    final float queryScale;
    final float queryComponentSum;
    final float queryAdditionalCorrection;
    final float centroidNormSq;          // ‖c‖²
    final float centroidDp;              // ⟨q,c⟩ for inner-product metrics; 0 for L2

    private AdcQueryContext(
        byte[] transposedQuery,
        float queryLower,
        float queryScale,
        float queryComponentSum,
        float queryAdditionalCorrection,
        float centroidNormSq,
        float centroidDp
    ) {
        this.transposedQuery = transposedQuery;
        this.queryLower = queryLower;
        this.queryScale = queryScale;
        this.queryComponentSum = queryComponentSum;
        this.queryAdditionalCorrection = queryAdditionalCorrection;
        this.centroidNormSq = centroidNormSq;
        this.centroidDp = centroidDp;
    }

    /**
     * Quantize {@code query} at {@code queryBits} relative to {@code centroid} and precompute the
     * corrective factors.
     *
     * @param centroidNormSq {@code ‖c‖²} for this centroid — read from {@code .clac} (stored per
     *     centroid) rather than recomputed here.
     * @param queryBits query-side quantization width (see {@link ScanParams#queryBits()}).
     */
    static AdcQueryContext quantize(
        float[] query,
        float[] centroid,
        float centroidNormSq,
        OptimizedScalarQuantizer quantizer,
        int dimension,
        VectorSimilarityFunction sim,
        int queryBits
    ) {
        if (queryBits != SUPPORTED_QUERY_BITS) {
            // The transposition and dot kernels below are 4-bit-query specific. A narrower query (e.g.
            // 1-bit against 1-bit docs, i.e. symmetric popcount) needs its own packing + kernel; reject
            // rather than score against a mismatched layout.
            throw new UnsupportedOperationException(
                "Only " + SUPPORTED_QUERY_BITS + "-bit query quantization is implemented, got: " + queryBits);
        }
        byte[] scratch = new byte[dimension];
        byte[][] destinations = new byte[][] { scratch };
        byte[] bitsArray = new byte[] { (byte) queryBits };
        float[] queryCopy = query.clone(); // multiScalarQuantize centers/normalizes in place
        OptimizedScalarQuantizer.QuantizationResult q =
            quantizer.multiScalarQuantize(queryCopy, destinations, bitsArray, centroid)[0];

        byte[] transposed = new byte[((dimension + 7) / 8) * 4];
        OptimizedScalarQuantizer.transposeHalfByte(scratch, transposed);

        float queryLower = q.lowerInterval();
        float queryScale = (q.upperInterval() - queryLower) / ((1 << queryBits) - 1);
        float queryComponentSum = (float) q.quantizedComponentSum();
        float queryAdditionalCorrection = q.additionalCorrection();
        float centroidDp = sim == VectorSimilarityFunction.EUCLIDEAN ? 0f : VectorUtil.dotProduct(query, centroid);

        return new AdcQueryContext(
            transposed, queryLower, queryScale, queryComponentSum, queryAdditionalCorrection, centroidNormSq, centroidDp);
    }
}
