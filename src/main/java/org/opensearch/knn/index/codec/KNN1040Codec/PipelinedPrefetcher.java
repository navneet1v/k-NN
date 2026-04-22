/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Pipelined prefetch scheduler for IVF search. Prefetches posting list data and
 * quantized vector data for upcoming centroids while the current centroid is being scored.
 *
 * <p>Unlike a simple ring-buffer approach that only prefetches one level ahead,
 * this scheduler:
 * <ul>
 *   <li>Prefetches both .clap (posting lists) and .claq (quantized vectors) in one pass</li>
 *   <li>Uses adaptive lookahead: prefetches up to {@code depth} centroids ahead</li>
 *   <li>Skips .claq prefetch when native SIMD scoring uses mmap (OS handles page faults)</li>
 *   <li>Coalesces nearby byte ranges to minimize prefetch syscalls</li>
 * </ul>
 *
 * <p>Usage in the search loop:
 * <pre>{@code
 * PipelinedPrefetcher prefetcher = new PipelinedPrefetcher(postingsInput, quantizedInput, ...);
 * for (int i = 0; i < nprobe; i++) {
 *     prefetcher.advanceTo(i);  // prefetches ahead while we score centroid i
 *     scanPostingList(nearestCentroids[i], ...);
 * }
 * }</pre>
 */
final class PipelinedPrefetcher {

    private static final int DEFAULT_DEPTH = 2;

    private final IndexInput postingsInput;
    private final IndexInput quantizedInput;
    private final int[] centroidOrder;
    private final long[] primaryOffsets;
    private final long[] soarOffsets;
    private final long quantizedBaseOffset;
    private final int recordSize;
    private final int avgPostingBytes;
    private final int depth;
    private final boolean skipQuantizedPrefetch;

    private int prefetchedUpTo = -1;

    /**
     * @param postingsInput       .clap IndexInput
     * @param quantizedInput      .claq IndexInput (may be null if no ADC)
     * @param centroidOrder       ordered centroid IDs to probe
     * @param primaryOffsets      per-centroid primary posting list offsets
     * @param soarOffsets         per-centroid SOAR posting list offsets
     * @param quantizedBaseOffset base offset in .claq
     * @param recordSize          per-vector record size in .claq
     * @param numVectors          total vectors in segment
     * @param numCentroids        total centroids
     * @param nativeScoring       true if C++ SIMD uses mmap (skip .claq prefetch)
     */
    PipelinedPrefetcher(
        IndexInput postingsInput,
        IndexInput quantizedInput,
        int[] centroidOrder,
        long[] primaryOffsets,
        long[] soarOffsets,
        long quantizedBaseOffset,
        int recordSize,
        int numVectors,
        int numCentroids,
        boolean nativeScoring
    ) {
        this.postingsInput = postingsInput;
        this.quantizedInput = quantizedInput;
        this.centroidOrder = centroidOrder;
        this.primaryOffsets = primaryOffsets;
        this.soarOffsets = soarOffsets;
        this.quantizedBaseOffset = quantizedBaseOffset;
        this.recordSize = recordSize;
        this.avgPostingBytes = Math.max(64, (numVectors / Math.max(1, numCentroids)) * 5);
        this.depth = DEFAULT_DEPTH;
        this.skipQuantizedPrefetch = nativeScoring || quantizedInput == null;
    }

    /**
     * Called before scoring centroid at position {@code currentIdx} in the probe order.
     * Prefetches posting lists (and optionally quantized data) for centroids
     * {@code currentIdx+1} through {@code currentIdx+depth}.
     */
    void advanceTo(int currentIdx) throws IOException {
        int target = Math.min(currentIdx + depth, centroidOrder.length - 1);
        for (int i = prefetchedUpTo + 1; i <= target; i++) {
            prefetchCentroid(centroidOrder[i]);
        }
        prefetchedUpTo = target;
    }

    /**
     * Prefetch quantized data for specific ordinals after reading a posting list.
     * More precise than the estimated region in advanceTo().
     */
    void prefetchQuantized(int[] ordinals, int count) throws IOException {
        if (skipQuantizedPrefetch || count == 0) return;
        for (int i = 0; i < count; i++) {
            long offset = quantizedBaseOffset + (long) ordinals[i] * recordSize;
            quantizedInput.prefetch(offset, recordSize);
        }
    }

    private void prefetchCentroid(int centId) throws IOException {
        // Posting lists (.clap) — clamp to avoid out-of-bounds
        long fileLength = postingsInput.length();
        long primaryLen = Math.min(avgPostingBytes, fileLength - primaryOffsets[centId]);
        if (primaryLen > 0) postingsInput.prefetch(primaryOffsets[centId], primaryLen);
        long soarLen = Math.min(avgPostingBytes, fileLength - soarOffsets[centId]);
        if (soarLen > 0) postingsInput.prefetch(soarOffsets[centId], soarLen);

        // Quantized vectors (.claq) — skip if native SIMD uses mmap
        if (!skipQuantizedPrefetch) {
            long qFileLength = quantizedInput.length();
            long estimatedStart = quantizedBaseOffset + (primaryOffsets[centId] / Math.max(1, avgPostingBytes)) * recordSize;
            long windowSize = Math.max((long) avgPostingBytes / 5 * recordSize, 4096);
            long clampedSize = Math.min(windowSize, qFileLength - Math.min(estimatedStart, qFileLength));
            if (clampedSize > 0 && estimatedStart < qFileLength) {
                quantizedInput.prefetch(estimatedStart, clampedSize);
            }
        }
    }
}
