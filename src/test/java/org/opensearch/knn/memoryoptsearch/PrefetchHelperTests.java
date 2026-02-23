/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.PrefetchHelper;

import java.io.IOException;

public class PrefetchHelperTests extends KNNTestCase {

    private TrackingIndexInput createTrackingInput() throws IOException {
        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        try (IndexOutput out = dir.createOutput("test", IOContext.DEFAULT)) {
            out.writeBytes(new byte[1024], 1024);
        }
        IndexInput delegate = dir.openInput("test", IOContext.DEFAULT);
        return new TrackingIndexInput(delegate);
    }

    public void testPrefetch_whenValidOrds_thenPrefetchesCorrectOffsets() throws IOException {
        long baseOffset = 100L;
        long vectorByteSize = 16L;
        TrackingIndexInput trackingInput = createTrackingInput();

        int[] ords = { 0, 2, 5 };
        PrefetchHelper.prefetch(trackingInput, baseOffset, vectorByteSize, ords, ords.length);

        assertEquals(3, trackingInput.prefetchCalls.size());
        // ord 0: offset = 100 + 0*16 = 100
        assertEquals(100L, trackingInput.prefetchCalls.get(0).offset());
        assertEquals(16L, trackingInput.prefetchCalls.get(0).length());
        // ord 2: offset = 100 + 2*16 = 132
        assertEquals(132L, trackingInput.prefetchCalls.get(1).offset());
        assertEquals(16L, trackingInput.prefetchCalls.get(1).length());
        // ord 5: offset = 100 + 5*16 = 180
        assertEquals(180L, trackingInput.prefetchCalls.get(2).offset());
        assertEquals(16L, trackingInput.prefetchCalls.get(2).length());
    }

    public void testPrefetch_whenNullOrds_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput();

        PrefetchHelper.prefetch(trackingInput, 0L, 16L, null, 0);

        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenNumOrdsLessThanArrayLength_thenOnlyPrefetchesNumOrds() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput();

        int[] ords = { 0, 1, 2, 3, 4 };
        PrefetchHelper.prefetch(trackingInput, 0L, 8L, ords, 2);

        assertEquals(2, trackingInput.prefetchCalls.size());
    }
}
