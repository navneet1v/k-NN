/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Sliding-window read-ahead that prefetches upcoming centroid data from .clap.
 *
 * <p>Maintains a lookahead window of N probes. When a probe is consumed,
 * the window slides forward and the newly visible probe is prefetched.
 * Unlike a ring buffer, the window is a simple array with a cursor —
 * no wrap-around arithmetic needed.
 */
public final class ReadAheadProbeScheduler implements ProbeScheduler {

    private static final int DEFAULT_LOOKAHEAD = 2;

    private final ProbeScheduler delegate;
    private final IndexInput postingsInput;
    private final int lookahead;

    // Sliding window: probes[0..filled-1] are buffered, cursor points to next to yield
    private ProbeTarget[] window;
    private int cursor;
    private int filled;

    public ReadAheadProbeScheduler(ProbeScheduler delegate, IndexInput postingsInput) throws IOException {
        this(delegate, postingsInput, DEFAULT_LOOKAHEAD);
    }

    public ReadAheadProbeScheduler(ProbeScheduler delegate, IndexInput postingsInput, int lookahead) throws IOException {
        this.delegate = delegate;
        this.postingsInput = postingsInput;
        this.lookahead = lookahead;
        this.window = new ProbeTarget[lookahead + 1]; // +1 for the current probe
        this.cursor = 0;
        this.filled = 0;
        warmUp();
    }

    @Override
    public boolean hasNext() {
        return cursor < filled;
    }

    @Override
    public ProbeTarget next() throws IOException {
        ProbeTarget current = window[cursor++];

        // Slide: try to append one more from delegate
        if (delegate.hasNext()) {
            ProbeTarget upcoming = delegate.next();
            // Shift window down if cursor has advanced past half
            if (cursor > lookahead) {
                compact();
            }
            window[filled++] = upcoming;
            issueReadAhead(upcoming);
        }

        return current;
    }

    /** Fill the initial window and issue read-ahead for all buffered probes. */
    private void warmUp() throws IOException {
        while (filled < window.length && delegate.hasNext()) {
            ProbeTarget probe = delegate.next();
            window[filled++] = probe;
            issueReadAhead(probe);
        }
    }

    /** Shift remaining probes to the front of the window. */
    private void compact() {
        int remaining = filled - cursor;
        System.arraycopy(window, cursor, window, 0, remaining);
        cursor = 0;
        filled = remaining;
    }

    private void issueReadAhead(ProbeTarget probe) throws IOException {
        long offset = probe.fileOffset();
        long len = probe.postingBytes();
        if (len > 0 && offset >= 0 && offset + len <= postingsInput.length()) {
            postingsInput.prefetch(offset, len);
        }
    }
}
