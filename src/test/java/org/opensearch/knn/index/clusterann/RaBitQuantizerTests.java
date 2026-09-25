/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.clusterann.codec.RaBitQuantizer;

import java.util.Random;

/**
 * Validates the RaBitQ B-bit encode + unbiased estimator in isolation (no codec/IVF), the way
 * rabitq_proto.py does offline: rank neighbors by the estimated inner product and compare recall to
 * brute-force. Confirms the estimator math before it is wired into the Flow D codec.
 */
public class RaBitQuantizerTests extends KNNTestCase {

    /** A simple deterministic orthonormal rotation via Gram–Schmidt on a seeded Gaussian matrix. */
    private static float[][] randomRotation(int dim, long seed) {
        Random rng = new Random(seed);
        double[][] a = new double[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) a[i][j] = rng.nextGaussian();
        // Gram–Schmidt
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < i; j++) {
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += a[i][d] * a[j][d];
                for (int d = 0; d < dim; d++) a[i][d] -= dot * a[j][d];
            }
            double n = 0;
            for (int d = 0; d < dim; d++) n += a[i][d] * a[i][d];
            n = Math.sqrt(n);
            for (int d = 0; d < dim; d++) a[i][d] /= n;
        }
        float[][] r = new float[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) r[i][j] = (float) a[i][j];
        return r;
    }

    private static float[] matVec(float[][] m, float[] v) {
        int dim = v.length;
        float[] out = new float[dim];
        for (int i = 0; i < dim; i++) {
            double s = 0;
            for (int j = 0; j < dim; j++) s += m[i][j] * v[j];
            out[i] = (float) s;
        }
        return out;
    }

    private static float[] normalize(float[] v) {
        double n = 0;
        for (float x : v) n += (double) x * x;
        n = Math.sqrt(n);
        float[] out = new float[v.length];
        if (n > 0) for (int i = 0; i < v.length; i++) out[i] = (float) (v[i] / n);
        return out;
    }

    private double recallAtK(int bits) {
        final int dim = 64, n = 2000, nq = 100, k = 50;
        Random rng = new Random(123);
        // Random data + a single global centroid proxy (as in the offline prototype).
        float[][] data = new float[n][dim];
        float[] centroid = new float[dim];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) data[i][d] = (float) rng.nextGaussian();
            for (int d = 0; d < dim; d++) centroid[d] += data[i][d] / n;
        }
        // Rotation (P). We rotate the normalized residual by P^{-1} = P^T (orthonormal).
        float[][] P = randomRotation(dim, 7L);
        float[][] Pt = new float[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) Pt[i][j] = P[j][i];

        // Encode all data vectors.
        byte[][] codes = new byte[n][dim];
        double[] gnorm = new double[n];
        double[] odob = new double[n];
        float[][] on = new float[n][]; // normalized residual (for GT)
        for (int i = 0; i < n; i++) {
            float[] res = new float[dim];
            for (int d = 0; d < dim; d++) res[d] = data[i][d] - centroid[d];
            float[] o = normalize(res);
            on[i] = o;
            float[] oprime = matVec(Pt, o);
            gnorm[i] = RaBitQuantizer.quantize(oprime, bits, codes[i]);
            odob[i] = RaBitQuantizer.odob(codes[i], oprime, bits, gnorm[i]);
        }

        double half = RaBitQuantizer.half(bits);
        int hits = 0;
        for (int qi = 0; qi < nq; qi++) {
            float[] q = new float[dim];
            for (int d = 0; d < dim; d++) q[d] = (float) rng.nextGaussian();
            float[] qres = new float[dim];
            for (int d = 0; d < dim; d++) qres[d] = q[d] - centroid[d];
            float[] qn = normalize(qres);
            float[] qprime = matVec(Pt, qn);
            double sumq = 0;
            for (int d = 0; d < dim; d++) sumq += qprime[d];

            // Estimated ⟨o,q'⟩ per doc.
            double[] est = new double[n];
            for (int i = 0; i < n; i++) {
                double obq = 0;
                for (int d = 0; d < dim; d++) obq += (codes[i][d] & 0xFF) * (double) qprime[d];
                obq = (obq - half * sumq) / (gnorm[i] <= 0 ? 1 : gnorm[i]);
                est[i] = obq / odob[i];
            }
            // Ground truth: cosine of normalized residuals (⟨o, qn⟩).
            double[] gt = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int d = 0; d < dim; d++) s += on[i][d] * qn[d];
                gt[i] = s;
            }
            hits += overlapTopK(est, gt, k);
        }
        return hits / (double) (nq * k);
    }

    private static int overlapTopK(double[] a, double[] b, int k) {
        Integer[] ia = topKIdx(a, k), ib = topKIdx(b, k);
        java.util.Set<Integer> sa = new java.util.HashSet<>(java.util.Arrays.asList(ia));
        int c = 0;
        for (Integer x : ib) if (sa.contains(x)) c++;
        return c;
    }

    private static Integer[] topKIdx(double[] v, int k) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (x, y) -> Double.compare(v[y], v[x]));
        return java.util.Arrays.copyOf(idx, k);
    }

    public void testEstimatorRecall_increasesWithBits() {
        double r3 = recallAtK(3);
        double r4 = recallAtK(4);
        double r5 = recallAtK(5);
        double r7 = recallAtK(7);
        // Recall must climb monotonically-ish with bits and reach high values, matching the offline
        // prototype (which saw ~0.85→0.99 across B on 64-128 dim data).
        assertTrue("B=3 recall too low: " + r3, r3 > 0.55);
        assertTrue("B=4 recall too low: " + r4, r4 > 0.70);
        assertTrue("B=5 recall too low: " + r5, r5 > 0.82);
        assertTrue("B=7 recall too low: " + r7, r7 > 0.92);
        assertTrue("recall should increase with bits: " + r3 + "," + r4 + "," + r5 + "," + r7,
            r4 >= r3 - 0.02 && r5 >= r4 - 0.02 && r7 >= r5 - 0.02);
    }
}
