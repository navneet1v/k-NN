/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.clusterann.codec.Int4DotProduct;
import org.opensearch.knn.index.clusterann.codec.ScalarBitEncoding;

import java.util.Random;

/**
 * Isolated correctness check of the ADC L2 reconstruction — no clustering, no search. Quantizes a
 * doc {@code v} and query {@code q} relative to a centroid {@code c} (exactly as the writer/reader
 * do), runs the {@code ADCBlockScorer} L2 math, and asserts the reconstructed distance matches the
 * true {@code ‖q−v‖²}. Pins whether the L2 recall problem is in the ADC math or in how the scanner
 * feeds per-block values.
 */
public class ClusterANNADCReconstructionTests extends KNNTestCase {

    private static final int DIM = 32;
    private static final int QUERY_BITS = 4;
    private static final float FOUR_BIT_SCALE = 1f / ((1 << QUERY_BITS) - 1);

    public void testAdcL2Reconstruction_4bit() {
        Random rng = new Random(7);
        final int docBits = 4;
        ScalarBitEncoding enc = ScalarBitEncoding.fromDocBits(docBits);
        int packedBytes = enc.docPackedBytes(DIM);
        float docBitScale = enc.docBitScale();
        OptimizedScalarQuantizer osq = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);

        double sumRelErr = 0;
        double maxRelErr = 0;
        int n = 300;
        for (int t = 0; t < n; t++) {
            float[] c = randVec(rng);
            float[] v = randVec(rng);      // doc FAR from c (large residual — e.g. a SOAR spill)
            float[] q = randVec(rng);      // query FAR from c too

            // Doc side (mirrors QuantizedVectorWriter.quantizeOne for docBits=4).
            byte[] dScratch = new byte[DIM];
            OptimizedScalarQuantizer.QuantizationResult dr =
                osq.multiScalarQuantize(v.clone(), new byte[][] { dScratch }, new byte[] { (byte) docBits }, c)[0];
            byte[] dPacked = new byte[packedBytes];
            OptimizedScalarQuantizer.transposeHalfByte(dScratch, dPacked);

            // Query side (mirrors ADCBlockScorer's constructor-time quantization).
            byte[] qScratch = new byte[DIM];
            OptimizedScalarQuantizer.QuantizationResult qr =
                osq.multiScalarQuantize(q.clone(), new byte[][] { qScratch }, new byte[] { (byte) QUERY_BITS }, c)[0];
            byte[] qT = new byte[((DIM + 7) / 8) * 4];
            OptimizedScalarQuantizer.transposeHalfByte(qScratch, qT);

            float docLower = dr.lowerInterval();
            float docScale = (dr.upperInterval() - docLower) * docBitScale;
            int docSum = dr.quantizedComponentSum();
            float docAdd = dr.additionalCorrection();

            float qLower = qr.lowerInterval();
            float queryScale = (qr.upperInterval() - qLower) * FOUR_BIT_SCALE;
            int qSum = qr.quantizedComponentSum();
            float qAdd = qr.additionalCorrection();

            float rawDot = Int4DotProduct.int4NibbleDotProduct(qT, dPacked);

            // ADCBlockScorer L2 math.
            float score = docLower * (qLower * DIM)
                + qLower * docScale * docSum
                + docLower * (queryScale * qSum)
                + docScale * queryScale * rawDot;
            float distAdc = qAdd + docAdd - 2f * score;

            float distTrue = sqDist(q, v);
            double relErr = Math.abs(distAdc - distTrue) / Math.max(distTrue, 1e-3);
            sumRelErr += relErr;
            maxRelErr = Math.max(maxRelErr, relErr);
        }
        double avgRelErr = sumRelErr / n;
        logger.info("[ADC L2 recon 4-bit] avgRelErr={} maxRelErr={} over {} pairs", avgRelErr, maxRelErr, n);
        assertTrue("ADC L2 reconstruction relative error too high (avg=" + avgRelErr + ")", avgRelErr < 0.15);
    }

    private float[] randVec(Random rng) {
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) v[i] = rng.nextFloat() * 10f;
        return v;
    }

    private float[] nearVec(float[] c, Random rng) {
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) v[i] = c[i] + (float) rng.nextGaussian() * 0.15f;
        return v;
    }

    private float sqDist(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
}
