/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * Flow D (Extended RaBitQ) per-query scan state. For each probed cell it prepares the rotated query
 * residual, then scores each doc from its stored RaBitQ code + corrections via the unbiased estimator.
 *
 * <p><b>Doc record</b> (written by the Flow D posting writer, one per vector):
 * <pre>code[dim] (unsigned B-bit, one byte each) | gnorm (float) | odob (float) | resNorm (float)</pre>
 * where {@code gnorm = ‖g‖}, {@code odob = ⟨ō,o'⟩}, {@code resNorm = ‖v − c‖}, {@code o = (v−c)/resNorm}.
 *
 * <p><b>Estimator.</b> With the rotated query residual {@code q' = P⁻¹(q − c)} (NOT normalized — we keep
 * its magnitude so the estimate is in the query's units), the unbiased estimate of {@code ⟨o, (q−c)⟩} is
 * <pre>estOQ = ⟨ō, q'⟩ / odob,  where ⟨ō,q'⟩ = (1/gnorm)(Σ code_i·q'_i − half·Σ q'_i)</pre>
 * Rotation is orthonormal so {@code ⟨ō, q'⟩ = ⟨P ō, q−c⟩}; dividing by {@code odob = ⟨ō,o'⟩ = ⟨Pō, o⟩}
 * yields an unbiased estimate of {@code ⟨o, q−c⟩}. Then {@code v = c + resNorm·o} gives:
 * <ul>
 *   <li><b>IP</b>: {@code ⟨q,v⟩ = ⟨q,c⟩ + resNorm·⟨o,q⟩}, and {@code ⟨o,q⟩ = ⟨o,q−c⟩ + ⟨o,c⟩}. We
 *       estimate {@code ⟨o,q−c⟩} directly; {@code ⟨o,c⟩} is unknown per-doc, so we fold it in by
 *       estimating {@code ⟨o, q⟩} against the RAW rotated query instead (see {@link #prepareCell}).</li>
 *   <li><b>L2</b>: {@code ‖q−v‖² = ‖q−c‖² − 2·resNorm·⟨o,q−c⟩ + resNorm²}.</li>
 * </ul>
 *
 * <p>To keep the estimator self-consistent we score against the rotated query residual {@code q' = P⁻¹(q−c)}
 * and the stored direction {@code o}; the estimate is {@code ⟨o, q−c⟩}. For IP we then add {@code ⟨q,c⟩}
 * (cell-constant) and use {@code ⟨q,v⟩ = ⟨q,c⟩ + resNorm·estOQ + } (the {@code ⟨o,c⟩} term is absorbed
 * because {@code o ⟂}-ish to nothing; empirically the (q−c) formulation matches the offline prototype).
 */
public final class RaBitQScanState {

    private final int dim;
    private final int bits;
    private final float half;
    private final boolean l2;
    private final float[] query;              // raw query
    private final RotationFn rotation;        // applies the same rotation the writer used to o

    /** Rotation applied to a residual vector (writer used the identical map on o). */
    @FunctionalInterface
    public interface RotationFn {
        void transform(float[] in, float[] out) throws java.io.IOException;
    }

    // Per-cell prepared state:
    private float[] qprimeRes;   // P·(q − c)  — used for L2 (estimates ⟨o, q−c⟩)
    private float sumQPrimeRes;  // Σ qprimeRes
    private float[] qprimeRaw;   // P·q        — used for IP (estimates ⟨o, q⟩ directly)
    private float sumQPrimeRaw;  // Σ qprimeRaw
    private float qcResNormSq;   // ‖q − c‖² (L2)
    private float qcDot;         // ⟨q, c⟩   (IP)

    public RaBitQScanState(float[] query, int dim, int bits, VectorSimilarityFunction sim, RotationFn rotation) {
        this.query = query;
        this.dim = dim;
        this.bits = bits;
        this.half = RaBitQuantizer.half(bits);
        this.l2 = sim == VectorSimilarityFunction.EUCLIDEAN;
        this.rotation = rotation;
    }

    /** Prepare the rotated query (raw for IP, residual for L2) and cell-constant terms for centroid {@code c}. */
    public void prepareCell(float[] c) throws java.io.IOException {
        double rn = 0;
        if (l2) {
            float[] resid = new float[dim];
            for (int d = 0; d < dim; d++) { float r = query[d] - c[d]; resid[d] = r; rn += (double) r * r; }
            float[] rotated = new float[dim];
            rotation.transform(resid, rotated);   // P·(q − c) — same map the writer applied to o
            this.qprimeRes = rotated;
            double s = 0;
            for (int d = 0; d < dim; d++) s += rotated[d];
            this.sumQPrimeRes = (float) s;
            this.qcResNormSq = (float) rn;
        } else {
            // IP: rotate the RAW query so the estimator yields ⟨o, q⟩ directly (no missing ⟨o,c⟩ term).
            float[] rotated = new float[dim];
            rotation.transform(query, rotated);    // P·q
            this.qprimeRaw = rotated;
            double s = 0;
            for (int d = 0; d < dim; d++) s += rotated[d];
            this.sumQPrimeRaw = (float) s;
            double qc = 0;
            for (int d = 0; d < dim; d++) qc += (double) query[d] * c[d];
            this.qcDot = (float) qc;
        }
    }

    /**
     * Similarity for a doc whose record starts at byte {@code off} in {@code buf}
     * (layout: code[dim] | gnorm | odob | resNorm). Higher = better.
     */
    public float score(byte[] buf, int off) {
        int cp = off + dim;
        float gnorm = readFloat(buf, cp);
        float odob = readFloat(buf, cp + 4);
        float resNorm = readFloat(buf, cp + 8);
        double invG = gnorm <= 0 ? 1.0 : 1.0 / gnorm;
        double invOdob = odob == 0 ? 1.0 : 1.0 / odob;

        if (l2) {
            double codeDot = 0;
            for (int d = 0; d < dim; d++) codeDot += (buf[off + d] & 0xFF) * (double) qprimeRes[d];
            double estOQ = ((codeDot - half * sumQPrimeRes) * invG) * invOdob; // ≈ ⟨o, q−c⟩
            // ‖q − v‖² = ‖q − c‖² − 2·resNorm·⟨o,q−c⟩ + resNorm²
            double d2 = qcResNormSq - 2.0 * resNorm * estOQ + (double) resNorm * resNorm;
            if (d2 < 0) d2 = 0;
            return (float) (1.0 / (1.0 + d2));
        }
        // IP: rotate RAW query → estOQ ≈ ⟨o, q⟩ directly; ⟨q,v⟩ = ⟨q,c⟩ + resNorm·⟨o,q⟩.
        double codeDot = 0;
        for (int d = 0; d < dim; d++) codeDot += (buf[off + d] & 0xFF) * (double) qprimeRaw[d];
        double estOQ = ((codeDot - half * sumQPrimeRaw) * invG) * invOdob;
        double dot = qcDot + resNorm * estOQ;
        return dot >= 0 ? (float) (dot + 1.0) : (float) (1.0 / (1.0 - dot));
    }

    private static float readFloat(byte[] b, int o) {
        int bits = ((b[o] & 0xFF)) | ((b[o + 1] & 0xFF) << 8) | ((b[o + 2] & 0xFF) << 16) | ((b[o + 3] & 0xFF) << 24);
        return Float.intBitsToFloat(bits);
    }

    /** Per-doc record size in bytes for a given dim: code + 3 float corrections. */
    public static int recordBytes(int dim) {
        return dim + 12;
    }
}
