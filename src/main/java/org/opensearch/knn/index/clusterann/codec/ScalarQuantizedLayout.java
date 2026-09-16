/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * How one field's scalar-quantized vectors are laid out and scored — the facts {@code .clam} recorded when
 * the field was written, plus what is derived from them.
 *
 * <p>These travel together everywhere in this family: the reader needs the strides, the scorer needs the
 * widths and the metric, the corrections bound needs both. Bundling them keeps that from being five
 * parameters threaded through four constructors, and makes it obvious they are fixed for the segment's life
 * rather than anything a query varies.
 *
 * <p>Immutable and shared: one instance per field, built at reader open.
 */
final class ScalarQuantizedLayout {

    private static final int CORRECTION_BYTES = 4 * Integer.BYTES; // lower, upper, add, sum

    private final int dimension;
    private final ScalarBitEncoding encoding;
    private final VectorSimilarityFunction sim;
    private final OptimizedScalarQuantizer quantizer;
    private final int packedBytes;
    private final long recordBytes;

    ScalarQuantizedLayout(int dimension, ScalarBitEncoding encoding, VectorSimilarityFunction sim) {
        this.dimension = dimension;
        this.encoding = encoding;
        this.sim = sim;
        this.quantizer = new OptimizedScalarQuantizer(sim);
        this.packedBytes = encoding.docPackedBytes(dimension);
        this.recordBytes = CORRECTION_BYTES + this.packedBytes;
    }

    int dimension() {
        return dimension;
    }

    VectorSimilarityFunction sim() {
        return sim;
    }

    OptimizedScalarQuantizer quantizer() {
        return quantizer;
    }

    /** Bytes of packed codes per vector. */
    int packedBytes() {
        return packedBytes;
    }

    /** Bytes one vector occupies in the payload: its corrective terms plus its packed codes. */
    long recordBytes() {
        return recordBytes;
    }

    /** Stored quantization width, which with the query's width selects the dot kernel. */
    int docBits() {
        return encoding.docBits();
    }

    /** {@code 1/(2^docBits − 1)} — turns a stored interval into a per-step scale. */
    float docBitScale() {
        return encoding.docBitScale();
    }

    /**
     * Bytes of a posting a scan is (near-)certain to read: the metadata header — always parsed in full —
     * plus the first block. Everything after that is streamed by the block reader's own prefetch as the scan
     * advances, so a posting that terminates early or is mostly pruned is never over-fetched.
     */
    long guaranteedBytes(int count) {
        return PostingHeader.byteLength(count)
            + (long) Math.min(ClusterANNFormatConstants.BLOCK_SIZE, count) * recordBytes;
    }
}
