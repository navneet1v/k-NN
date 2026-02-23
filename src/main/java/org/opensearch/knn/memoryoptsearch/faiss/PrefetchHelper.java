/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Helper class to prefetch vector data from disk.
 * Used by Faiss flat index implementations to prefetch vectors before scoring.
 */
public class PrefetchHelper {

    private PrefetchHelper() {}

    public static void prefetch(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        if (ordsToPrefetch == null) {
            return;
        }
        for (int i = 0; i < numOrds; i++) {
            long offset = baseOffset + (long) ordsToPrefetch[i] * oneVectorByteSize;
            indexInput.prefetch(offset, oneVectorByteSize);
        }
    }
}
