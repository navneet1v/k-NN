/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * Flow C (ScaNN residual PQ) per-query scan state. Holds the global codebook + raw query and, per
 * probed cell, builds the metric-appropriate lookup table so a doc's PQ code reconstructs its score.
 *
 * <p>The doc stores {@code Quantize(v − c)} = residual code r. Reconstruction depends on the metric:
 *
 * <ul>
 *   <li><b>Inner product</b>: ⟨q, d⟩ = ⟨q, d−c⟩ + ⟨q, c⟩ = ⟨q, r⟩ + ⟨q, c⟩. LUT from the RAW query q;
 *       add per-cell bias ⟨q, c⟩.
 *   <li><b>Euclidean (L2)</b>: ‖q−d‖² = ‖q−c−r‖² = ‖q−c‖² − 2⟨q−c, r⟩ + ‖r‖². LUT from the RESIDUALIZED
 *       query (q−c); add per-cell ‖q−c‖² and per-code ‖r‖². Returned as a similarity 1/(1+d²).
 * </ul>
 */
public final class PQScanState {

    private final PQCodebook codebook;
    private final float[] query;
    private final boolean l2;
    private float[] lut;
    private float bias;        // IP: ⟨q,c⟩   L2: ‖q−c‖²

    public PQScanState(PQCodebook codebook, float[] query, VectorSimilarityFunction sim) {
        this.codebook = codebook;
        this.query = query;
        this.l2 = sim == VectorSimilarityFunction.EUCLIDEAN;
    }

    public PQCodebook codebook() {
        return codebook;
    }

    public void prepareCell(float[] centroid) {
        int dim = query.length;
        if (l2) {
            // LUT from residualized query (q−c); bias = ‖q−c‖².
            float[] rq = new float[dim];
            float nq = 0f;
            for (int d = 0; d < dim; d++) { float t = query[d] - centroid[d]; rq[d] = t; nq += t * t; }
            this.lut = codebook.buildDotLUT(rq);
            this.bias = nq;
        } else {
            // LUT from raw query q; bias = ⟨q,c⟩.
            this.lut = codebook.buildDotLUT(query);
            float b = 0f;
            for (int d = 0; d < dim; d++) b += query[d] * centroid[d];
            this.bias = b;
        }
    }

    /** Similarity for the doc whose PQ code starts at {@code code[off]} (higher = better). */
    public float score(byte[] code, int off) {
        if (l2) {
            // ‖q−d‖² = ‖q−c‖² − 2⟨q−c, r⟩ + ‖r‖²
            float d2 = bias - 2f * codebook.scoreDot(lut, code, off) + codebook.codeNormSq(code, off);
            if (d2 < 0f) d2 = 0f;
            return 1f / (1f + d2);
        }
        // IP: ⟨q, d⟩ = ⟨q, r⟩ + ⟨q, c⟩
        float dot = codebook.scoreDot(lut, code, off) + bias;
        return dot >= 0f ? dot + 1f : 1f / (1f - dot);
    }
}
