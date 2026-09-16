/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import java.util.Arrays;
import java.util.Random;

/**
 * Product Quantization / Asymmetric Hashing codebook — a faithful reproduction of Google ScaNN's
 * asymmetric_hashing2 PRODUCT scheme (see google-research/scann hashes/asymmetric_hashing2).
 *
 * <p>Flow C ("ScaNN residual AH"): the vector is split into {@code numSubspaces} contiguous chunks
 * ("blocks"). Each chunk is quantized to one of {@code K} learned centers (a single GLOBAL codebook,
 * trained once on the RESIDUALS {@code v − centroid} across all cells — which is how ScaNN avoids a
 * per-leaf codebook: subtracting the centroid makes residuals from every partition statistically
 * uniform, so one codebook suffices). A vector's code is {@code numSubspaces} byte indices.
 *
 * <p>Scoring is asymmetric (AH): for a (residualized) query we precompute a lookup table
 * {@code lut[block][c]} = partial dot of the query's chunk with center {@code c}; a doc's score is
 * {@code Σ_block lut[block][code[block]]} — {@code numSubspaces} table lookups, no per-dim math.
 *
 * <p>UNVERIFIED research reproduction. PRODUCT scheme only (no STACKED / PRODUCT_AND_BIAS).
 */
public final class PQCodebook {

    /**
     * Centers per subspace ("num_clusters_per_block"). ScaNN's max-accuracy default is 256 (8-bit),
     * but that makes single-threaded Java encode (K·subDim per block, ×numSubspaces, ×N vectors)
     * too slow at 1M. We use 16 (4-bit codes) — a standard ScaNN config — which cuts encode 16×
     * and remains real residual product quantization / AH. 16 still fits one byte per block.
     */
    public static final int K = 16;

    /** ScaNN default dimensions_per_block = 2. numSubspaces = ceil(dim / 2). */
    public static final int DIMS_PER_BLOCK = 2;

    private final int dim;
    private final int numSubspaces;
    private final int subDim;             // dim / numSubspaces (dim must be divisible)
    /** centers[block][c][subDim] — the global codebook. */
    private final float[][][] centers;

    private PQCodebook(int dim, int numSubspaces, int subDim, float[][][] centers) {
        this.dim = dim;
        this.numSubspaces = numSubspaces;
        this.subDim = subDim;
        this.centers = centers;
    }

    public int dim() { return dim; }
    public int numSubspaces() { return numSubspaces; }
    public int codeBytes() { return numSubspaces; } // 1 byte per block at K<=256

    /**
     * Train the global codebook on residuals with ScaNN defaults (K=256, dims_per_block=2).
     * Trains on a SAMPLE (ScaNN samples too) to keep write-time bounded at scale.
     */
    public static PQCodebook trainDefault(float[][] residuals, int dim, long seed) {
        int numSubspaces = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
        if (dim % DIMS_PER_BLOCK != 0) {
            throw new IllegalArgumentException("dim " + dim + " not divisible by DIMS_PER_BLOCK " + DIMS_PER_BLOCK);
        }
        return train(residuals, dim, numSubspaces, seed);
    }

    /** Max residuals used to train each subspace's k-means (ScaNN samples; keeps training bounded). */
    private static final int TRAIN_SAMPLE = Integer.getInteger("clusterann.pq.trainSample", 100_000);

    /**
     * Train the global codebook on residuals. {@code residuals[i]} is one {@code dim}-length
     * residual (v − its cell centroid). Per subspace: k-means (Lloyd) to K centers on a sample.
     */
    public static PQCodebook train(float[][] residuals, int dim, int numSubspaces, long seed) {
        if (dim % numSubspaces != 0) {
            throw new IllegalArgumentException("dim " + dim + " not divisible by numSubspaces " + numSubspaces);
        }
        int subDim = dim / numSubspaces;
        Random rng = new Random(seed);

        // Sample rows once, shared across subspaces (ScaNN trains AH on a sample of the residuals).
        int n = residuals.length;
        int m = Math.min(n, TRAIN_SAMPLE);
        int[] sample = new int[m];
        if (m == n) {
            for (int i = 0; i < m; i++) sample[i] = i;
        } else {
            for (int i = 0; i < m; i++) sample[i] = rng.nextInt(n);
        }

        float[][][] centers = new float[numSubspaces][K][subDim];
        for (int b = 0; b < numSubspaces; b++) {
            int off = b * subDim;
            // init: K distinct-ish sampled chunks
            for (int c = 0; c < K; c++) {
                int r = sample[rng.nextInt(m)];
                System.arraycopy(residuals[r], off, centers[b][c], 0, subDim);
            }
            int[] assign = new int[m];
            for (int iter = 0; iter < 12; iter++) {
                for (int s = 0; s < m; s++) {
                    int i = sample[s];
                    float best = Float.MAX_VALUE; int bc = 0;
                    for (int c = 0; c < K; c++) {
                        float d = 0f;
                        for (int j = 0; j < subDim; j++) {
                            float diff = residuals[i][off + j] - centers[b][c][j];
                            d += diff * diff;
                        }
                        if (d < best) { best = d; bc = c; }
                    }
                    assign[s] = bc;
                }
                float[][] sum = new float[K][subDim];
                int[] cnt = new int[K];
                for (int s = 0; s < m; s++) {
                    int i = sample[s]; int c = assign[s];
                    cnt[c]++;
                    for (int j = 0; j < subDim; j++) sum[c][j] += residuals[i][off + j];
                }
                for (int c = 0; c < K; c++) {
                    if (cnt[c] > 0) {
                        for (int j = 0; j < subDim; j++) centers[b][c][j] = sum[c][j] / cnt[c];
                    } else {
                        int r = sample[rng.nextInt(m)];
                        System.arraycopy(residuals[r], off, centers[b][c], 0, subDim);
                    }
                }
            }
        }
        return new PQCodebook(dim, numSubspaces, subDim, centers);
    }

    /** Encode one residual vector to {@code numSubspaces} center-index bytes into {@code dest[off..]}. */
    public void encode(float[] residual, byte[] dest, int off) {
        for (int b = 0; b < numSubspaces; b++) {
            int so = b * subDim;
            float best = Float.MAX_VALUE; int bc = 0;
            for (int c = 0; c < K; c++) {
                float d = 0f;
                for (int j = 0; j < subDim; j++) {
                    float diff = residual[so + j] - centers[b][c][j];
                    d += diff * diff;
                }
                if (d < best) { best = d; bc = c; }
            }
            dest[off + b] = (byte) bc;
        }
    }

    /**
     * Build the AH lookup table for a residualized query. {@code lut[b*K + c]} is the partial DOT of
     * the query's chunk {@code b} with center {@code c}. Doc score (dot) = Σ_b lut[b*K + code[b]].
     * (Dot, because ClusterANN's IP metric wants ⟨q,d⟩; residual dot approximates ⟨q−ctr, d−ctr⟩.)
     */
    public float[] buildDotLUT(float[] residualQuery) {
        float[] lut = new float[numSubspaces * K];
        for (int b = 0; b < numSubspaces; b++) {
            int so = b * subDim;
            for (int c = 0; c < K; c++) {
                float d = 0f;
                for (int j = 0; j < subDim; j++) d += residualQuery[so + j] * centers[b][c][j];
                lut[b * K + c] = d;
            }
        }
        return lut;
    }

    /** Score a doc code against a prebuilt LUT: Σ_b lut[b*K + code[b]] = approx dot of residuals. */
    public float scoreDot(float[] lut, byte[] code, int off) {
        float s = 0f;
        for (int b = 0; b < numSubspaces; b++) {
            s += lut[b * K + (code[off + b] & 0xFF)];
        }
        return s;
    }

    /** ‖reconstructed residual‖² for a doc code = Σ_b ‖center[b][code[b]]‖² (for the L2 expansion). */
    public float codeNormSq(byte[] code, int off) {
        float s = 0f;
        for (int b = 0; b < numSubspaces; b++) {
            float[] ctr = centers[b][code[off + b] & 0xFF];
            for (int j = 0; j < subDim; j++) s += ctr[j] * ctr[j];
        }
        return s;
    }

    // ---- serialization ----
    public void write(org.apache.lucene.store.IndexOutput out) throws java.io.IOException {
        out.writeInt(dim);
        out.writeInt(numSubspaces);
        for (int b = 0; b < numSubspaces; b++)
            for (int c = 0; c < K; c++)
                for (int j = 0; j < subDim; j++)
                    out.writeInt(Float.floatToIntBits(centers[b][c][j]));
    }

    public static PQCodebook read(org.apache.lucene.store.IndexInput in) throws java.io.IOException {
        int dim = in.readInt();
        int ns = in.readInt();
        int sd = dim / ns;
        float[][][] c = new float[ns][K][sd];
        for (int b = 0; b < ns; b++)
            for (int k = 0; k < K; k++)
                for (int j = 0; j < sd; j++)
                    c[b][k][j] = Float.intBitsToFloat(in.readInt());
        return new PQCodebook(dim, ns, sd, c);
    }
}
