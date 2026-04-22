/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.HashMap;

public class ClusterANNSQEncoderTests extends KNNTestCase {

    private final ClusterANNSQEncoder encoder = new ClusterANNSQEncoder();

    private MethodComponentContext encoderContext(int bits) {
        return new MethodComponentContext(ClusterANNSQEncoder.NAME, new HashMap<>() {
            {
                put(ClusterANNSQEncoder.BITS_PARAM, bits);
            }
        });
    }

    public void testCalculateCompressionLevel_1bit() {
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(encoderContext(1), KNNMethodConfigContext.EMPTY));
    }

    public void testCalculateCompressionLevel_2bit() {
        assertEquals(CompressionLevel.x16, encoder.calculateCompressionLevel(encoderContext(2), KNNMethodConfigContext.EMPTY));
    }

    public void testCalculateCompressionLevel_4bit() {
        assertEquals(CompressionLevel.x8, encoder.calculateCompressionLevel(encoderContext(4), KNNMethodConfigContext.EMPTY));
    }

    public void testCalculateCompressionLevel_noBitsParam() {
        MethodComponentContext ctx = new MethodComponentContext(ClusterANNSQEncoder.NAME, new HashMap<>());
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(ctx, KNNMethodConfigContext.EMPTY));
    }

    public void testCalculateCompressionLevel_nullContext() {
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(null, KNNMethodConfigContext.EMPTY));
    }

    public void testGetName() {
        assertEquals("sq", encoder.getName());
    }
}
