/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * POC query-side int8 scorer for shard-level rerank. Quantizes the rotated query once, then scores
 * a stashed doc int8 code with the exact IVFaster offset correction. Metric fixed to MAX_INNER_PRODUCT
 * for the current benchmark (normalized/innerproduct data); extend if other metrics are needed.
 */
public final class ShardInt8Query {

    private static final int OFFSET = 128;

    private final int dim;
    private final byte[] q;
    private final int qSum;
    private final float qScale;

    public ShardInt8Query(float[] rotatedQuery, int dim) {
        this.dim = dim;
        this.q = new byte[dim];
        float maxAbs = 0f;
        for (int d = 0; d < dim; d++) {
            float a = Math.abs(rotatedQuery[d]);
            if (a > maxAbs) maxAbs = a;
        }
        if (maxAbs == 0f) {
            java.util.Arrays.fill(q, (byte) OFFSET);
            this.qScale = 1f;
            this.qSum = 0;
            return;
        }
        this.qScale = maxAbs / 127f;
        float inv = 127f / maxAbs;
        int sum = 0;
        for (int d = 0; d < dim; d++) {
            int v = Math.round(rotatedQuery[d] * inv);
            if (v > 127) v = 127;
            else if (v < -127) v = -127;
            sum += v;
            q[d] = (byte) (v + OFFSET);
        }
        this.qSum = sum;
    }

    /** MAX_INNER_PRODUCT score for a doc int8 code + corrections (exact offset correction). */
    public float score(byte[] code, float dScale, int dSum, float dNorm) {
        long unsignedDot = 0;
        for (int d = 0; d < dim; d++) {
            unsignedDot += (long) (q[d] & 0xFF) * (code[d] & 0xFF);
        }
        long offsetConst = (long) OFFSET * qSum + 16384L * dim;
        long signedDot = unsignedDot - offsetConst - (long) OFFSET * (long) dSum;
        double dot = (double) signedDot * qScale * dScale;
        return dot >= 0 ? (float) (dot + 1.0) : (float) (1.0 / (1.0 - dot));
    }
}
