/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.clusterann.prefetch;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Prefetches upcoming centroid posting data using {@link IndexInput#prefetch(long, long)}.
 * Ring buffer of configurable depth — each buffered probe has its posting bytes prefetched
 * before it's consumed.
 *
 * <p>Uses Lucene's async prefetch API which triggers OS-level readahead (madvise/fadvise).
 */
public final class PrefetchingProbeIterator implements ProbeIterator {

    private static final int DEFAULT_DEPTH = 2;

    private final ProbeIterator delegate;
    private final IndexInput postingsInput;
    private final ProbedCentroid[] buffer;
    private int readIdx;
    private int writeIdx;
    private int count;

    public PrefetchingProbeIterator(ProbeIterator delegate, IndexInput postingsInput) throws IOException {
        this(delegate, postingsInput, DEFAULT_DEPTH);
    }

    public PrefetchingProbeIterator(ProbeIterator delegate, IndexInput postingsInput, int depth) throws IOException {
        this.delegate = delegate;
        this.postingsInput = postingsInput;
        this.buffer = new ProbedCentroid[depth];
        this.readIdx = 0;
        this.writeIdx = 0;
        this.count = 0;
        fillBuffer();
    }

    @Override
    public boolean hasNext() {
        return count > 0;
    }

    @Override
    public ProbedCentroid next() throws IOException {
        ProbedCentroid result = buffer[readIdx];
        readIdx = (readIdx + 1) % buffer.length;
        count--;

        // Refill one from delegate
        if (delegate.hasNext()) {
            ProbedCentroid probe = delegate.next();
            buffer[writeIdx] = probe;
            writeIdx = (writeIdx + 1) % buffer.length;
            count++;
            prefetch(probe);
        }
        return result;
    }

    private void fillBuffer() throws IOException {
        while (count < buffer.length && delegate.hasNext()) {
            ProbedCentroid probe = delegate.next();
            buffer[writeIdx] = probe;
            writeIdx = (writeIdx + 1) % buffer.length;
            count++;
            prefetch(probe);
        }
    }

    private void prefetch(ProbedCentroid probe) throws IOException {
        long offset = probe.fileOffset();
        long len = probe.postingBytes();
        if (len > 0 && offset >= 0 && offset + len <= postingsInput.length()) {
            postingsInput.prefetch(offset, len);
        }
    }
}
