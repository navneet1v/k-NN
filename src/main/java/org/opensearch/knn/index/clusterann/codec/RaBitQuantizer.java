/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * Extended RaBitQ quantizer (Gao et al., VLDB 2024, arXiv:2409.09913) — B-bit codes with an
 * unbiased inner-product estimator and a theoretical error bound. Flow D in ClusterANN.
 *
 * <p><b>What it does.</b> For a data vector {@code v} in an IVF cell with centroid {@code c}:
 * <ol>
 *   <li>Residualize and normalize: {@code o = (v − c) / ‖v − c‖} (a unit direction).</li>
 *   <li>Rotate into a random orthonormal basis: {@code o' = P⁻¹ o} (rotation shared by all vectors;
 *       here supplied externally so the codec can reuse ClusterANN's stored rotation).</li>
 *   <li>Quantize {@code o'} to unsigned B-bit codes {@code u ∈ [0, 2^B−1]^D} that maximize the cosine
 *       with {@code o'} over the grid {@code g = u − (2^B−1)/2}. This is RaBitQ's rescale-and-round
 *       (Algorithm 1): sweep the scale {@code t} at the values where {@code round(t·o'_i)} changes and
 *       keep the grid point with the highest cosine.</li>
 * </ol>
 *
 * <p><b>Per-vector we store:</b> the code {@code u}, the residual norm {@code ‖v − c‖}, and the
 * estimator denominator {@code odob = ⟨ō, o⟩} where {@code ō = g/‖g‖} is the reconstructed unit
 * direction. At query time the unbiased estimate of {@code ⟨o, q'⟩} is {@code ⟨ō,q'⟩ / odob}, with
 * {@code ⟨ō,q'⟩ = (1/‖g‖)(⟨u, q'⟩ − half·Σq')}. The full-space {@code ⟨v,q⟩} or {@code ‖v−q‖²} is then
 * reconstructed from the residual norms and the query↔centroid terms (see {@link RaBitQScanState}).
 *
 * <p>Faithful to the paper's estimator; the codebook here is the rotated integer grid (Extended
 * RaBitQ), which subsumes the 1-bit hypercube of the original RaBitQ at B=1.
 */
public final class RaBitQuantizer {

    private RaBitQuantizer() {}

    /** Grid half-width for B bits: values are u − half, u ∈ [0, 2^B−1]. */
    public static float half(int bits) {
        return ((1 << bits) - 1) / 2.0f;
    }

    /**
     * Quantize a rotated unit residual {@code oprime} to B-bit unsigned codes.
     *
     * @param oprime rotated normalized residual P⁻¹o (length D)
     * @param bits   B in [1, 8]
     * @param outCode destination for the D unsigned codes (0..2^B−1), length ≥ D
     * @return {@code ‖g‖} (norm of the signed grid vector g = code − half), used by the estimator
     */
    public static double quantize(float[] oprime, int bits, byte[] outCode) {
        final int dim = oprime.length;
        final double h = half(bits);
        final double gmax = h;

        // Collect the critical scales t where round(t·o'_i) crosses a half-integer, per dimension:
        // t = (k + 0.5) / |o'_i| for k = 0 .. floor(h)-1. Sweeping these enumerates every distinct
        // rounding outcome; we keep the grid point with the highest cosine to o'.
        int maxK = (int) Math.floor(h);
        // Upper bound on count; we cap the sweep for very high dim/bits to stay O(D·2^B) bounded.
        double[] crit = new double[dim * Math.max(1, maxK)];
        int ci = 0;
        for (int i = 0; i < dim; i++) {
            double oi = Math.abs(oprime[i]);
            if (oi < 1e-12) continue;
            for (int k = 0; k < maxK; k++) {
                crit[ci++] = (k + 0.5) / oi;
            }
        }
        if (ci == 0) {
            // Degenerate (all-zero direction): put every code at the grid center.
            int center = Math.round((float) h);
            for (int i = 0; i < dim; i++) outCode[i] = (byte) center;
            // g is all zeros → ‖g‖ = 0; caller guards against this.
            return 0.0;
        }
        double[] sorted = java.util.Arrays.copyOf(crit, ci);
        java.util.Arrays.sort(sorted);

        // Cap the sweep: evaluating the O(dim) cosine at EVERY critical t is O(dim²·maxK) per vector,
        // which is prohibitive at high dim (768² · 7 ≈ 4M ops/vec). Subsampling the sorted critical
        // values to a bounded count keeps encode ~O(dim · CAP) with negligible recall impact (the
        // offline prototype used the same cap and still hit B=4 ≈ 0.96). CAP scales mildly with bits.
        final int CAP = Math.min(sorted.length, 96 + 32 * bits);
        int stride = Math.max(1, sorted.length / CAP);

        double bestCos = Double.NEGATIVE_INFINITY;
        float[] bestG = new float[dim];
        float[] g = new float[dim];
        for (int s = 0; s < sorted.length; s += stride) {
            double t = sorted[s];
            double gn = 0, dot = 0;
            for (int i = 0; i < dim; i++) {
                double gi = Math.rint(t * oprime[i]);
                if (gi > gmax) gi = gmax;
                else if (gi < -gmax) gi = -gmax;
                g[i] = (float) gi;
                gn += gi * gi;
                dot += gi * oprime[i];
            }
            if (gn == 0) continue;
            double cos = dot / Math.sqrt(gn);
            if (cos > bestCos) {
                bestCos = cos;
                System.arraycopy(g, 0, bestG, 0, dim);
            }
        }

        double gnorm = 0;
        for (int i = 0; i < dim; i++) {
            int u = (int) (bestG[i] + h);
            outCode[i] = (byte) u;
            gnorm += (double) bestG[i] * bestG[i];
        }
        return Math.sqrt(gnorm);
    }

    /**
     * The estimator denominator {@code odob = ⟨ō, o'⟩} for a quantized vector, where
     * {@code ō = g/‖g‖}. Equal to {@code (1/‖g‖) Σ (code_i − half) · o'_i}.
     */
    public static double odob(byte[] code, float[] oprime, int bits, double gnorm) {
        if (gnorm <= 0) return 1.0; // guarded degenerate
        final double h = half(bits);
        double dot = 0;
        for (int i = 0; i < oprime.length; i++) {
            dot += ((code[i] & 0xFF) - h) * (double) oprime[i];
        }
        return dot / gnorm;
    }
}
