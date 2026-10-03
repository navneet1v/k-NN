/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

/**
 * What a vector costs in a scalar-quantized code block, and so how many of them a block holds.
 *
 * <p>Sits beside {@link ScalarEncoding}, which both sides already depend on, because the writer that lays a
 * block down and the reader that walks it have to agree to the byte. {@code .clam} records the field's {@code ioFetchBytes} budget rather than the vector count it
 * works out to, so the count is derived here on write and derived again here on read — one implementation, so
 * the two cannot drift.
 *
 * <p>That choice keeps block geometry out of the orchestrators: nothing between the format and the block
 * implementation has to carry a vector count, which is what lets a different block layout be plugged in without
 * touching the writers and readers above it. The price is that this arithmetic is part of the on-disk
 * contract — changing it reinterprets every segment already written — so it moves with the codec version and is
 * not a tuning knob. {@code ioFetchBytes} is the knob.
 */
public final class ScalarBlockLayout {

    /** Corrective-term bytes each vector carries in a block: lower, upper, add and sum, one int each. */
    private static final int CORRECTIVE_BYTES = 4 * Integer.BYTES;

    private static final String UNSUPPORTED_ENCODING = "unsupported scalar encoding: ";

    private ScalarBlockLayout() {}

    /**
     * Vectors per code block for a field: as many whole vectors as {@code ioFetchBytes} holds, and never fewer
     * than one, so a vector wider than the budget still gets a block of its own rather than no block at all.
     *
     * @param ioFetchBytes the field's byte budget for one block
     * @param encoding     the scalar encoding the codes are packed with
     * @param dimension    the field's vector dimension
     */
    public static int blockSize(final int ioFetchBytes, final ScalarEncoding encoding, final int dimension) {
        return Math.max(1, ioFetchBytes / bytesPerVector(encoding, dimension));
    }

    /**
     * Bytes one vector occupies in a block: its four corrective terms plus its packed codes.
     *
     * @throws UnsupportedOperationException if the block layout does not support {@code encoding}
     */
    public static int bytesPerVector(final ScalarEncoding encoding, final int dimension) {
        return CORRECTIVE_BYTES + packedLength(encoding, dimension);
    }

    /**
     * Bytes one vector's packed codes occupy.
     *
     * @throws UnsupportedOperationException if the block layout does not support {@code encoding}
     */
    public static int packedLength(final ScalarEncoding encoding, final int dimension) {
        return switch (encoding) {
            case SINGLE_BIT_QUERY_NIBBLE, DIBIT_QUERY_NIBBLE, PACKED_NIBBLE -> encoding.getDocPackedLength(
                encoding.getDiscreteDimensions(dimension)
            );
            default -> throw new UnsupportedOperationException(UNSUPPORTED_ENCODING + encoding);
        };
    }
}
