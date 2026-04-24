/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.algorithm;

/**
 * Default scalar implementation of {@link BulkVectorOps}.
 * Uses {@code Math.fma} for fused multiply-add (hardware FMA on x86/ARM when available).
 */
final class DefaultBulkVectorOps implements BulkVectorOps {

    @Override
    public void squareDistanceBulk(float[] q, float[] v0, float[] v1, float[] v2, float[] v3, float[] distances) {
        float d0 = 0f, d1 = 0f, d2 = 0f, d3 = 0f;
        for (int i = 0; i < q.length; i++) {
            float qi = q[i];
            float diff0 = qi - v0[i];
            d0 = Math.fma(diff0, diff0, d0);
            float diff1 = qi - v1[i];
            d1 = Math.fma(diff1, diff1, d1);
            float diff2 = qi - v2[i];
            d2 = Math.fma(diff2, diff2, d2);
            float diff3 = qi - v3[i];
            d3 = Math.fma(diff3, diff3, d3);
        }
        distances[0] = d0;
        distances[1] = d1;
        distances[2] = d2;
        distances[3] = d3;
    }

    @Override
    public void soarDistanceBulk(
        float[] vec,
        float[] c0,
        float[] c1,
        float[] c2,
        float[] c3,
        float[] residual,
        float soarLambda,
        float residualNormSq,
        float[] distances
    ) {
        float dsq0 = 0f, dsq1 = 0f, dsq2 = 0f, dsq3 = 0f;
        float proj0 = 0f, proj1 = 0f, proj2 = 0f, proj3 = 0f;
        for (int i = 0; i < vec.length; i++) {
            float vi = vec[i];
            float ri = residual[i];
            float diff0 = vi - c0[i];
            dsq0 = Math.fma(diff0, diff0, dsq0);
            proj0 = Math.fma(diff0, ri, proj0);
            float diff1 = vi - c1[i];
            dsq1 = Math.fma(diff1, diff1, dsq1);
            proj1 = Math.fma(diff1, ri, proj1);
            float diff2 = vi - c2[i];
            dsq2 = Math.fma(diff2, diff2, dsq2);
            proj2 = Math.fma(diff2, ri, proj2);
            float diff3 = vi - c3[i];
            dsq3 = Math.fma(diff3, diff3, dsq3);
            proj3 = Math.fma(diff3, ri, proj3);
        }
        float invNorm = soarLambda / residualNormSq;
        distances[0] = dsq0 + invNorm * proj0 * proj0;
        distances[1] = dsq1 + invNorm * proj1 * proj1;
        distances[2] = dsq2 + invNorm * proj2 * proj2;
        distances[3] = dsq3 + invNorm * proj3 * proj3;
    }
}
