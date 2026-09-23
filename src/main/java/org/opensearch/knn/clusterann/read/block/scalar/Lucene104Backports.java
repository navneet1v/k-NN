/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.util.VectorUtil;

/**
 * The {@code OptimizedScalarQuantizer} and {@code VectorUtil} statics the ClusterANN code needs that Lucene 10.4 has
 * and the Lucene 10.3 OpenSearch 3.5 ships does not. SearchServicesKnnVectorFormats builds against a patched 10.3 that
 * carries them and calls {@code OptimizedScalarQuantizer.transposeDibit} and {@code VectorUtil.int4DotProductSinglePacked}
 * directly; the sync rewrite redirects those calls here. Delete this class, {@link ScalarEncoding}, and the rewrite
 * rules once the bundled Lucene has them.
 *
 * <p>{@link #transposeDibit} is copied from Lucene's {@code OptimizedScalarQuantizer} so the on-disk layout is Lucene's,
 * byte for byte: two stripes of {@code packed.length / 2}, the low bit of every dimension then the high bit, most
 * significant bit first within a byte. {@link Int4DotProduct#dibit} reads exactly this.
 */
public final class Lucene104Backports {

    private Lucene104Backports() {}

    /**
     * Dot product of an unpacked int4 vector against a nibble-packed one; Lucene 10.4 renamed 10.3's
     * {@code VectorUtil.int4DotProductPacked} to this (the same scalar and Panama implementations, with a
     * {@code BothPacked} sibling), so the sync rewrite redirects the new name here and this delegates to the old one.
     *
     * @param unpacked one int4 value (0-15) per byte, {@code dimension} bytes
     * @param packed two int4 values per byte, {@code (dimension + 1) / 2} bytes
     */
    public static int int4DotProductSinglePacked(byte[] unpacked, byte[] packed) {
        return VectorUtil.int4DotProductPacked(unpacked, packed);
    }

    /**
     * Transpose a 2-bit (dibit) quantized vector into a byte array for efficient bitwise operations. The result has 2
     * stripes: similar to {@code OptimizedScalarQuantizer#transposeHalfByte}, but only for 2 bits.
     *
     * @param vector the 2-bit quantized vector (values 0-3)
     * @param packed the byte array to store the transposed vector
     */
    public static void transposeDibit(byte[] vector, byte[] packed) {
        int limit = vector.length - 7;
        int i = 0;
        int index = 0;
        for (; i < limit; i += 8, index++) {
            int lowerByte = (vector[i] & 1) << 7 | (vector[i + 1] & 1) << 6 | (vector[i + 2] & 1) << 5 | (vector[i + 3] & 1) << 4
                | (vector[i + 4] & 1) << 3 | (vector[i + 5] & 1) << 2 | (vector[i + 6] & 1) << 1 | (vector[i + 7] & 1);
            int upperByte = ((vector[i] >> 1) & 1) << 7 | ((vector[i + 1] >> 1) & 1) << 6 | ((vector[i + 2] >> 1) & 1) << 5 | ((vector[i
                + 3] >> 1) & 1) << 4 | ((vector[i + 4] >> 1) & 1) << 3 | ((vector[i + 5] >> 1) & 1) << 2 | ((vector[i + 6] >> 1) & 1) << 1
                | ((vector[i + 7] >> 1) & 1);
            packed[index] = (byte) lowerByte;
            packed[index + packed.length / 2] = (byte) upperByte;
        }
        if (i == vector.length) {
            return;
        }
        int lowerByte = 0;
        int upperByte = 0;
        for (int j = 7; i < vector.length; j--, i++) {
            assert vector[i] >= 0 && vector[i] <= 3;
            lowerByte |= (vector[i] & 1) << j;
            upperByte |= ((vector[i] >> 1) & 1) << j;
        }
        packed[index] = (byte) lowerByte;
        packed[index + packed.length / 2] = (byte) upperByte;
    }
}
