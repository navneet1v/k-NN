/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;
import java.util.Arrays;

/**
 * Helper class to prefetch vector data from disk to improve query performance.
 * <p>
 * This class provides two prefetch strategies:
 * <ul>
 *   <li>128KB prefetch: Prefetches fixed 128KB chunks starting from vector offsets</li>
 *   <li>Exact prefetch: Prefetches only the exact byte range needed for vectors</li>
 * </ul>
 * <p>
 * Vectors within 128KB of each other are grouped together to minimize prefetch operations.
 * The strategy is controlled by {@link KNNFeatureFlags#isExactVectorSizePrefetch()}.
 * <p>
 * <b>Example - 128KB Prefetch:</b><br>
 * Given vectors at offsets [100KB, 120KB, 300KB] with 10KB vector size:
 * <ul>
 *   <li>Group 1: Prefetch 128KB starting at 100KB (covers vectors at 100KB and 120KB)</li>
 *   <li>Group 2: Prefetch 128KB starting at 300KB (covers vector at 300KB)</li>
 * </ul>
 * Result: 2 prefetch operations, 256KB total prefetched
 * <p>
 * <b>Example - Exact Prefetch:</b><br>
 * Given vectors at offsets [100KB, 120KB, 300KB] with 10KB vector size:
 * <ul>
 *   <li>Group 1: Prefetch 40KB (100KB to 130KB, covers vectors at 100KB and 120KB)</li>
 *   <li>Group 2: Prefetch 10KB (300KB to 310KB, covers vector at 300KB)</li>
 * </ul>
 * Result: 2 prefetch operations, 50KB total prefetched
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class PrefetchHelper {

    // TODO: If needed we can get this value via Cluster Settings
    private static final long BYTES_128 = 128 * 1024;

    /**
     * Prefetches vector data from disk using the configured strategy.
     * <p>
     * Vectors are grouped by proximity (within 128KB) to minimize prefetch calls.
     * The prefetch strategy is determined by the feature flag.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    public static void prefetch(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        if (ordsToPrefetch == null || numOrds <= 1) {
            return;
        }
        if (KNNFeatureFlags.isExactVectorSizePrefetch()) {
            log.debug("Using exact vector size prefetch for {} vectors ids", numOrds);
            prefetchExactVectorSize(indexInput, baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        } else {
            log.debug("Using 128KB prefetch for {} vectors ids", numOrds);
            prefetch128KB(indexInput, baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        }
    }

    /**
     * Prefetches vectors using fixed 128KB chunks.
     * <p>
     * Groups vectors within 128KB ranges and prefetches 128KB starting from each group.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    private static void prefetch128KB(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        long[] offsets = calculateAndSortOffsets(baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        int groupCount = prefetchWithGrouping(indexInput, offsets, numOrds, oneVectorByteSize, false);
        log.trace(
            "Prefetching compressed [{}] vectors where num of ords was [{}] using {} bytes prefetch size",
            groupCount,
            numOrds,
            BYTES_128
        );
    }

    /**
     * Prefetches vectors using exact byte ranges.
     * <p>
     * Groups vectors within 128KB ranges and prefetches only the exact bytes needed for each group.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    private static void prefetchExactVectorSize(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        long[] offsets = calculateAndSortOffsets(baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        int groupCount = prefetchWithGrouping(indexInput, offsets, numOrds, oneVectorByteSize, true);
        log.trace("Prefetching compressed [{}] vectors where num of ords was [{}] using exact prefetch size", groupCount, numOrds);
    }

    /**
     * Groups vectors within 128KB ranges and prefetches them.
     * <p>
     * Vectors are sorted by offset and grouped if they fall within 128KB of the group start.
     * Each group is prefetched as a single operation.
     *
     * @param indexInput the index input to prefetch from
     * @param offsets sorted array of vector offsets
     * @param numOrds number of vectors to prefetch
     * @param oneVectorByteSize the size of one vector in bytes
     * @param exactSize if true, prefetch exact byte range; if false, prefetch 128KB chunks
     * @return the number of prefetch groups created
     * @throws IOException if an I/O error occurs during prefetch
     */
    private static int prefetchWithGrouping(
        final IndexInput indexInput,
        final long[] offsets,
        final int numOrds,
        final long oneVectorByteSize,
        final boolean exactSize
    ) throws IOException {
        int groupCount = 1;
        long groupStartOffset = offsets[0];

        for (int i = 1; i < numOrds; i++) {
            if (offsets[i] - groupStartOffset > BYTES_128) {
                long length = exactSize ? offsets[i - 1] + oneVectorByteSize - groupStartOffset : BYTES_128;
                indexInput.prefetch(groupStartOffset, length);
                groupCount++;
                groupStartOffset = offsets[i];
            }
        }
        // Prefetch final group
        long finalLength = exactSize
            ? offsets[numOrds - 1] + oneVectorByteSize - groupStartOffset
            : Math.min(BYTES_128, indexInput.length() - groupStartOffset);
        indexInput.prefetch(groupStartOffset, finalLength);
        return groupCount;
    }

    /**
     * Calculates file offsets for vector ordinals and sorts them.
     *
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals
     * @param numOrds number of valid ordinals in the array
     * @return sorted array of file offsets
     */
    private static long[] calculateAndSortOffsets(long baseOffset, long oneVectorByteSize, int[] ordsToPrefetch, int numOrds) {
        long[] offsets = new long[numOrds];
        for (int i = 0; i < numOrds; i++) {
            offsets[i] = baseOffset + ((long) ordsToPrefetch[i] * oneVectorByteSize);
        }
        Arrays.sort(offsets);
        return offsets;
    }
}
