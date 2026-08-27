/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * Identifies which quantization family a field's postings were written with — the key (together with
 * {@code docBits}) for choosing the {@link Cluster} implementation and its scorer at read time.
 *
 * <p>Serialized as a single byte in the {@code .clam} per-field header. Today only {@link #SCALAR}
 * exists (every field is written with {@code OptimizedScalarQuantizer}); the type is stored anyway so
 * the read path can dispatch without guessing, and so a future family (e.g. product/binary
 * quantization) is a purely additive change.
 */
public enum QuantizerType {
    SCALAR((byte) 0);

    private final byte id;

    QuantizerType(byte id) {
        this.id = id;
    }

    /** Stable on-disk id (do not renumber existing values). */
    public byte id() {
        return id;
    }

    public static QuantizerType fromId(byte id) {
        for (QuantizerType t : values()) {
            if (t.id == id) {
                return t;
            }
        }
        throw new IllegalArgumentException("Unknown quantizer id: " + id);
    }
}
