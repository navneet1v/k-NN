/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Fast, isolated unit tests for {@link ScanParams} (JUnit 5).
 *
 * <p>The record constrains neither of its components, so what is left to pin is that it hands both back
 * untouched and that the factory applies the default width.
 */
class ScanParamsTests {

    private static final float[] QUERY = { 1.0f, 2.0f };

    @Test
    void testQuery_thenCarriedThroughWithoutCopying() {
        // given
        ScanParams params = new ScanParams(QUERY, ScanParams.DEFAULT_QUERY_BITS);

        // when
        float[] query = params.query();

        // then
        assertSame(QUERY, query, "the query is shared across every cluster a scan visits, so it must not be copied");
    }

    @Test
    void testOf_thenUsesTheDefaultQueryWidth() {
        // given / when
        ScanParams params = ScanParams.of(QUERY);

        // then
        assertSame(QUERY, params.query());
        assertEquals(ScanParams.DEFAULT_QUERY_BITS, params.queryBits());
        assertEquals(4, ScanParams.DEFAULT_QUERY_BITS);
    }
}
