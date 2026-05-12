/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.search.DocValueFormat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;

/**
 * DocValueFormat for knn_vector fields. Supports two modes:
 * <ul>
 *   <li>{@code array} — vectors are returned as JSON numeric arrays</li>
 *   <li>{@code binary} (default) — vectors are returned as base64-encoded little-endian byte strings (with padding)</li>
 * </ul>
 */
@Getter
public class KNNVectorDocValueFormat implements DocValueFormat {

    public static final String NAME = "knn_vector";
    public static final String FORMAT_ARRAY = "array";
    public static final String FORMAT_BINARY = "binary";

    public static final KNNVectorDocValueFormat ARRAY_FORMAT = new KNNVectorDocValueFormat(false);
    public static final KNNVectorDocValueFormat BINARY_FORMAT = new KNNVectorDocValueFormat(true);

    private final boolean binary;
    private static final Base64.Encoder BASE64_ENCODER = Base64.getEncoder();

    private KNNVectorDocValueFormat(boolean binary) {
        this.binary = binary;
    }

    public KNNVectorDocValueFormat(final StreamInput in) throws IOException {
        this.binary = in.readBoolean();
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public void writeTo(final StreamOutput out) throws IOException {
        out.writeBoolean(binary);
    }

    /**
     * Encodes a float[] vector as a base64 string with little-endian byte order.
     * Each float is written as 4 bytes in little-endian format (native byte order on x86/ARM),
     * then the resulting byte array is base64-encoded.
     */
    public static String encodeToBinary(final float[] vector) {
        final ByteBuffer buffer = ByteBuffer.allocate(vector.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(vector); // Bulk operation optimized by the JVM
        final byte[] bytes = buffer.array();
        return BASE64_ENCODER.encodeToString(bytes);
    }

    @Override
    public String toString() {
        return binary ? "knn_vector(binary)" : "knn_vector(array)";
    }
}
