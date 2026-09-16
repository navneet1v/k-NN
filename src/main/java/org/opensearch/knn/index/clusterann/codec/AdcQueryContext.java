/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * The query, quantized against one reference centroid, plus every term the corrective math derives from
 * it. Built once per <em>reference</em> — not once per posting — and shared by every posting whose codes
 * were written against that reference.
 *
 * <p>That scoping is the point. Quantizing the query is the expensive per-reference step
 * ({@code multiScalarQuantize} optimizes the interval, then the result is transposed into bit planes), and
 * the terms below are all constants once it is done. Today one cluster is one reference, so this is built
 * per posting; when several clusters share a quantization centroid it is built once for the group and
 * reused, and nothing here changes.
 *
 * <p>The loop constants are precomputed here rather than in the scorer because they are constant over the
 * whole reference, not just over a block.
 */
final class AdcQueryContext {

    /** 4-bit query, half-byte transposed into bit planes for the dot kernel. */
    final byte[] transposedQuery;

    final int dimension;
    final float queryLower;        // l_q
    final float queryScale;        // s_q
    final float queryComponentSum; // Σᵢbᵢ

    /**
     * The query's metric-dependent corrective term, mirroring the per-vector {@code add} column: it is
     * {@code ‖q−c‖²} for {@code EUCLIDEAN} and {@code ⟨q,c⟩} otherwise. One value serves both because the
     * quantizer switches on its own similarity function, which is the field's.
     */
    final float queryCorrection;

    /** {@code ‖c‖²} of the reference. Inner-product metrics only — L2's centroid terms cancel. */
    final float centroidNormSq;

    // Constant over this reference; hoisted out of the per-vector loop.
    final float qLowerDim;      // l_q · dim
    final float qScaleCompSum;  // s_q · Σᵢbᵢ
    final float dpMinusNorm;    // ⟨q,c⟩ − ‖c‖², inner product only

    private AdcQueryContext(
        byte[] transposedQuery,
        int dimension,
        float queryLower,
        float queryScale,
        float queryComponentSum,
        float queryCorrection,
        float centroidNormSq
    ) {
        this.transposedQuery = transposedQuery;
        this.dimension = dimension;
        this.queryLower = queryLower;
        this.queryScale = queryScale;
        this.queryComponentSum = queryComponentSum;
        this.queryCorrection = queryCorrection;
        this.centroidNormSq = centroidNormSq;
        this.qLowerDim = queryLower * dimension;
        this.qScaleCompSum = queryScale * queryComponentSum;
        this.dpMinusNorm = queryCorrection - centroidNormSq;
    }

    /**
     * Quantize {@code query} relative to {@code reference} at {@code queryBits} and transpose it for the
     * dot kernel. The corrective term comes straight from the quantizer, so its meaning follows the
     * quantizer's own similarity function — which must be the field's, or it would be the wrong quantity.
     */
    static AdcQueryContext quantize(
        float[] query,
        Centroid reference,
        OptimizedScalarQuantizer quantizer,
        int queryBits
    ) {
        if (queryBits != ScanParams.DEFAULT_QUERY_BITS) {
            // The transposition and dot kernels are 4-bit-query specific. A narrower query (e.g. 1-bit
            // against 1-bit docs, i.e. symmetric popcount) needs its own packing and kernel; reject rather
            // than score against a layout it doesn't match.
            throw new UnsupportedOperationException(
                "Only " + ScanParams.DEFAULT_QUERY_BITS + "-bit query quantization is implemented, got: " + queryBits);
        }
        int dimension = query.length;
        byte[] scratch = new byte[dimension];
        float[] queryCopy = query.clone(); // multiScalarQuantize centers in place
        OptimizedScalarQuantizer.QuantizationResult q = quantizer.multiScalarQuantize(
            queryCopy, new byte[][] { scratch }, new byte[] { (byte) queryBits }, reference.vector())[0];

        byte[] transposed = new byte[((dimension + 7) / 8) * 4];
        OptimizedScalarQuantizer.transposeHalfByte(scratch, transposed);

        float lower = q.lowerInterval();
        float scale = (q.upperInterval() - lower) / ((1 << queryBits) - 1);
        return new AdcQueryContext(
            transposed, dimension, lower, scale, q.quantizedComponentSum(), q.additionalCorrection(), reference.normSq());
    }
}
