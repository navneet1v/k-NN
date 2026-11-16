/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ClusterANNEncodingTests {

    @Test
    void testFromQuantizerId_whenOptimizedSq_thenNamesThatBackend() {
        assertEquals(
            ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION,
            ClusterANNEncoding.fromQuantizerId(ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ)
        );
    }

    @Test
    void testFromQuantizerId_whenUnknown_thenThrowsNamingTheId() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> ClusterANNEncoding.fromQuantizerId(7));
        assertTrue(e.getMessage().contains("7"), e.getMessage());
    }
}
