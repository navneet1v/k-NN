/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link HadamardRotation}: it must be orthonormal (preserve norms and dot products, so the
 * ranking survives) across power-of-two and non-power-of-two dimensions, and must reproduce exactly from its
 * seed so the write and read sides land in the same space.
 */
class HadamardRotationTests {

    @ParameterizedTest(name = "dim {0}")
    @ValueSource(ints = { 2, 8, 128, 384, 768, 1024 })
    void preservesDotProduct(final int dim) {
        final Random r = new Random(dim * 13L);
        final HadamardRotation rot = HadamardRotation.create(dim, 42L);
        final float[] a = gaussian(r, dim);
        final float[] b = gaussian(r, dim);
        final float[] ra = new float[dim];
        final float[] rb = new float[dim];
        rot.rotate(a, ra);
        rot.rotate(b, rb);

        final double before = dot(a, b);
        final double after = dot(ra, rb);
        assertTrue(
            Math.abs(before - after) < 1e-3 * Math.max(1.0, Math.abs(before)),
            "dim=" + dim + " dot before=" + before + " after=" + after
        );
    }

    @ParameterizedTest(name = "dim {0}")
    @ValueSource(ints = { 8, 768, 1024 })
    void preservesNorm(final int dim) {
        final Random r = new Random(dim * 17L);
        final HadamardRotation rot = HadamardRotation.create(dim, 7L);
        final float[] a = gaussian(r, dim);
        final float[] ra = new float[dim];
        rot.rotate(a, ra);
        assertTrue(Math.abs(dot(a, a) - dot(ra, ra)) < 1e-3 * dot(a, a), "norm not preserved at dim=" + dim);
    }

    @Test
    void sameSeedReproducesSameRotation() {
        final int dim = 768;
        final HadamardRotation a = HadamardRotation.create(dim, 123L);
        final HadamardRotation b = HadamardRotation.create(dim, 123L);
        final Random r = new Random(1L);
        final float[] v = gaussian(r, dim);
        final float[] ra = new float[dim];
        final float[] rb = new float[dim];
        a.rotate(v, ra);
        b.rotate(v, rb);
        for (int i = 0; i < dim; i++) {
            assertEquals(ra[i], rb[i], 0f, "seed must reproduce the rotation exactly at i=" + i);
        }
    }

    @Test
    void flattensVariance() {
        // A spiky input (energy in one dim) should spread across dims after rotation: per-dim variance evens out.
        final int dim = 256;
        final HadamardRotation rot = HadamardRotation.create(dim, 5L);
        final float[] spike = new float[dim];
        spike[0] = 10f;
        final float[] out = new float[dim];
        rot.rotate(spike, out);
        double max = 0;
        for (float x : out) {
            max = Math.max(max, Math.abs(x));
        }
        // Energy 100 over 256 dims normalized: no single dim should hold more than a small fraction.
        assertTrue(max < 2.0, "variance not flattened; max coord " + max);
    }

    @Test
    void rotate_whenSrcAndDestAreTheSameArray_thenThrows() {
        final HadamardRotation rot = HadamardRotation.create(8, 1L);
        final float[] shared = new float[8];
        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> rot.rotate(shared, shared));
        assertTrue(e.getMessage().contains("distinct"), e.getMessage());
    }

    @Test
    void rotate_whenAnArrayIsTheWrongLength_thenThrows() {
        final HadamardRotation rot = HadamardRotation.create(8, 1L);
        assertThrows(IllegalArgumentException.class, () -> rot.rotate(new float[7], new float[8]));
        assertThrows(IllegalArgumentException.class, () -> rot.rotate(new float[8], new float[9]));
    }

    @Test
    void create_whenDimensionIsNotPositive_thenThrows() {
        assertThrows(IllegalArgumentException.class, () -> HadamardRotation.create(0, 1L));
        assertThrows(IllegalArgumentException.class, () -> HadamardRotation.create(-4, 1L));
    }

    @Test
    void ramBytesUsed_isPositive() {
        assertTrue(HadamardRotation.create(768, 1L).ramBytesUsed() > 0);
    }

    private static float[] gaussian(final Random r, final int dim) {
        final float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = (float) r.nextGaussian();
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
}
