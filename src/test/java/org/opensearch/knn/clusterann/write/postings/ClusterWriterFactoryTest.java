/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8ClusterWriter;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies {@link ClusterWriterFactory} maps a quantization backend to its cluster writer and to its stable
 * {@code .clam} {@code quantizerId}. Unsupported backends can't be constructed — {@link ClusterANNEncoding} is a
 * closed enum and the factory's switches are exhaustive — so adding one is a compile error, not a runtime check.
 */
class ClusterWriterFactoryTest {

    private static final int DIMENSION = 4;
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;
    private static final int ONE_KB = 1024;
    private static final int STORAGE_BLOCK_BYTES = 32 * ONE_KB;

    private static QuantizationParams quantization(final int docBits) {
        return QuantizationParams.of(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, docBits);
    }

    @Test
    void buildsScalarQuantizedWriterForOptimizedSqBackend() {
        final ClusterWriter writer = new ClusterWriterFactory(quantization(1), DIMENSION, METRIC, STORAGE_BLOCK_BYTES).newWriter(
            new float[DIMENSION]
        );
        assertInstanceOf(OptimizedScalarQuantizedClusterWriter.class, writer);
    }

    @Test
    void buildsDataBlindInt8WriterAtItsFixedWidth() {
        final QuantizationParams int8 = QuantizationParams.of(ClusterANNEncoding.DATA_BLIND_INT8, DataBlindInt8ClusterWriter.DOC_BITS);
        final ClusterWriter writer = new ClusterWriterFactory(int8, DIMENSION, METRIC, STORAGE_BLOCK_BYTES).newWriter(new float[DIMENSION]);
        assertInstanceOf(DataBlindInt8ClusterWriter.class, writer);
    }

    @Test
    void rejectsDataBlindInt8AtAnyWidthOtherThanEight() {
        // The int8 grid is fixed at 8-bit codes, but writeField records docBits into .clam verbatim; a width the
        // writer cannot honour must be rejected at this seam so the recorded width can't drift from the bytes
        // written. 4 is the OSQ default and the most likely value to be passed by mistake.
        final QuantizationParams drifting = QuantizationParams.of(ClusterANNEncoding.DATA_BLIND_INT8, 4);
        assertThrows(IllegalArgumentException.class, () -> new ClusterWriterFactory(drifting, DIMENSION, METRIC, STORAGE_BLOCK_BYTES));
    }

    @Test
    void validateDocBitsRejectsDataBlindInt8AtAnyWidthOtherThanEight() {
        // The static guard is called on both the non-empty path (via the constructor) and the empty-field path
        // (before any .clam is written), so an empty DATA_BLIND_INT8 field with a mismatched width fails the same
        // way a non-empty one does, and the recorded width can never drift from the bytes written.
        assertThrows(
            IllegalArgumentException.class,
            () -> ClusterWriterFactory.validateDocBits(QuantizationParams.of(ClusterANNEncoding.DATA_BLIND_INT8, 4))
        );
    }

    @Test
    void validateDocBitsAcceptsDataBlindInt8AtItsFixedWidth() {
        // 8 is the one width the int8 backend encodes; it must pass.
        ClusterWriterFactory.validateDocBits(
            QuantizationParams.of(ClusterANNEncoding.DATA_BLIND_INT8, DataBlindInt8ClusterWriter.DOC_BITS)
        );
    }

    @Test
    void validateDocBitsLeavesOptimizedScalarWidthsToItsOwnQuantizer() {
        // The optimized-scalar backend validates its width where it builds its quantizer, so this seam accepts any
        // width for it rather than second-guessing.
        for (final int bits : new int[] { 1, 2, 4 }) {
            ClusterWriterFactory.validateDocBits(QuantizationParams.of(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, bits));
        }
    }

    @Test
    void vectorsPerBlockFillTheStorageBlock() {
        // dim 768: 16 B of corrective terms + 96 B of 1-bit codes = 112 B, or + 192 B of 2-bit codes = 208 B
        assertEquals(292, ClusterWriterFactory.vectorsPerBlock(quantization(1), 768, STORAGE_BLOCK_BYTES));
        assertEquals(157, ClusterWriterFactory.vectorsPerBlock(quantization(2), 768, STORAGE_BLOCK_BYTES));
        // 4-bit: 16 B + 384 B of nibble codes = 400 B
        assertEquals(81, ClusterWriterFactory.vectorsPerBlock(quantization(4), 768, STORAGE_BLOCK_BYTES));
    }

    @Test
    void vectorsPerBlockCountsOnlyWholeVectors() {
        final int bytesPerVector = 112; // dim 768 at 1 bit
        assertEquals(3, ClusterWriterFactory.vectorsPerBlock(quantization(1), 768, 4 * bytesPerVector - 1));
    }

    @Test
    void vectorsPerBlockIsAtLeastOneWhenAVectorExceedsTheStorageBlock() {
        assertEquals(1, ClusterWriterFactory.vectorsPerBlock(quantization(2), 4096, 64));
    }

    @Test
    void mapsOptimizedSqBackendToItsOnDiskQuantizerId() {
        assertEquals(
            ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ,
            ClusterWriterFactory.quantizerId(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION)
        );
    }
}
