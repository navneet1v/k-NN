package org.opensearch.knn.quantization;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link DataBlindInt8Quantizer}: encode then score, checked against the full-precision
 * similarity it estimates. The quantizer is data-blind and absolute (no centroid), so a query and a doc are
 * encoded the same way and the signed-dot reconstruction must recover ⟨v_query, v_doc⟩ within int8 error.
 */
class DataBlindInt8QuantizerTests {

    @ParameterizedTest(name = "dim {0}")
    @ValueSource(ints = { 8, 16, 128, 768, 1024 })
    void encodeThenScore_mip_estimatesTrueInnerProduct(final int dim) {
        final Random random = new Random(dim * 7L);
        final DataBlindInt8Quantizer q = new DataBlindInt8Quantizer(dim);

        final float[] query = gaussian(random, dim);
        final float[] doc = gaussian(random, dim);

        final byte[] qCode = new byte[dim];
        final byte[] dCode = new byte[dim];
        final DataBlindInt8Quantizer.Int8Code qc = q.encode(query, qCode);
        final DataBlindInt8Quantizer.Int8Code dc = q.encode(doc, dCode);

        final long udot = Int8DotProduct.unsignedDot(qCode, dCode, 0, dim);
        final float est = DataBlindInt8Quantizer.score(
            udot,
            q.queryTerms(qc),
            dc.scale(),
            dc.sum(),
            dc.sqNorm(),
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
        );

        // Invert the MIP transform: est = dot+1 (dot≥0) or 1/(1-dot) (dot<0) => recover the estimated dot.
        final double recoveredDot = est >= 1f ? (est - 1.0) : (1.0 - 1.0 / est);
        final double trueDot = dot(query, doc);
        // int8 over ~unit-scaled gaussians: error grows with dim; bound generously but meaningfully.
        final double tol = 0.03 * Math.max(1.0, Math.abs(trueDot)) + 0.5 * Math.sqrt(dim) * 0.02;
        assertTrue(
            Math.abs(recoveredDot - trueDot) < tol,
            "dim=" + dim + " estDot=" + recoveredDot + " trueDot=" + trueDot + " tol=" + tol
        );
    }

    @ParameterizedTest(name = "dim {0}")
    @ValueSource(ints = { 8, 128, 768 })
    void encodeThenScore_euclidean_estimatesTrueSquaredDistance(final int dim) {
        final Random random = new Random(dim * 11L);
        final DataBlindInt8Quantizer q = new DataBlindInt8Quantizer(dim);
        final float[] query = gaussian(random, dim);
        final float[] doc = gaussian(random, dim);

        final byte[] qCode = new byte[dim];
        final byte[] dCode = new byte[dim];
        final DataBlindInt8Quantizer.Int8Code qc = q.encode(query, qCode);
        final DataBlindInt8Quantizer.Int8Code dc = q.encode(doc, dCode);

        final long udot = Int8DotProduct.unsignedDot(qCode, dCode, 0, dim);
        final float est = DataBlindInt8Quantizer.score(
            udot,
            q.queryTerms(qc),
            dc.scale(),
            dc.sum(),
            dc.sqNorm(),
            VectorSimilarityFunction.EUCLIDEAN
        );

        // est = 1/(1+d²) => d² = 1/est - 1
        final double recoveredD2 = 1.0 / est - 1.0;
        final double trueD2 = squaredDistance(query, doc);
        final double tol = 0.05 * Math.max(1.0, trueD2) + Math.sqrt(dim) * 0.05;
        assertTrue(Math.abs(recoveredD2 - trueD2) < tol, "dim=" + dim + " estD2=" + recoveredD2 + " trueD2=" + trueD2 + " tol=" + tol);
    }

    @Test
    void encode_zeroVector_isOffsetWithUnitScaleAndZeroSum() {
        final int dim = 32;
        final DataBlindInt8Quantizer q = new DataBlindInt8Quantizer(dim);
        final byte[] code = new byte[dim];
        final DataBlindInt8Quantizer.Int8Code c = q.encode(new float[dim], code);

        assertEquals(1f, c.scale());
        assertEquals(0, c.sum());
        for (int d = 0; d < dim; d++) {
            assertEquals((byte) DataBlindInt8Quantizer.OFFSET, code[d]);
        }
    }

    @Test
    void encode_identicalVectors_scoreIsSelfSimilarityNearMax() {
        final int dim = 256;
        final Random random = new Random(1L);
        final DataBlindInt8Quantizer q = new DataBlindInt8Quantizer(dim);
        final float[] v = gaussian(random, dim);
        final byte[] a = new byte[dim];
        final byte[] b = new byte[dim];
        final DataBlindInt8Quantizer.Int8Code ca = q.encode(v, a);
        final DataBlindInt8Quantizer.Int8Code cb = q.encode(v.clone(), b);

        final long udot = Int8DotProduct.unsignedDot(a, b, 0, dim);
        final float est = DataBlindInt8Quantizer.score(
            udot,
            q.queryTerms(ca),
            cb.scale(),
            cb.sum(),
            cb.sqNorm(),
            VectorSimilarityFunction.EUCLIDEAN
        );
        // self-distance ≈ 0 => similarity ≈ 1
        assertTrue(est > 0.95f, "self-similarity should be near 1, was " + est);
    }

    private static float[] gaussian(final Random r, final int dim) {
        final float[] v = new float[dim];
        for (int d = 0; d < dim; d++) {
            v[d] = (float) r.nextGaussian();
        }
        return v;
    }

    private static double dot(final float[] a, final float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
        }
        return s;
    }

    private static double squaredDistance(final float[] a, final float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            final double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
}
