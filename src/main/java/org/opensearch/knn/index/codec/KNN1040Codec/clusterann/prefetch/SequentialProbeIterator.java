/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.clusterann.prefetch;

import java.io.IOException;
import java.util.Arrays;

/**
 * Reorders probes by file offset within fixed-size windows for sequential I/O.
 *
 * <p>Global offset sorting hurts recall because early termination needs nearest-first.
 * Windows of W centroids give ~80% of sequential benefit with minimal recall impact.
 * Within each window, centroids are sorted by file offset.
 */
public final class SequentialProbeIterator implements ProbeIterator {

    private static final int DEFAULT_WINDOW = 4;

    private final ProbeIterator delegate;
    private final ProbedCentroid[] window;
    private int windowSize;
    private int windowCursor;

    public SequentialProbeIterator(ProbeIterator delegate) {
        this(delegate, DEFAULT_WINDOW);
    }

    public SequentialProbeIterator(ProbeIterator delegate, int windowSize) {
        this.delegate = delegate;
        this.window = new ProbedCentroid[windowSize];
        this.windowSize = 0;
        this.windowCursor = 0;
    }

    @Override
    public boolean hasNext() {
        return windowCursor < windowSize || delegate.hasNext();
    }

    @Override
    public ProbedCentroid next() throws IOException {
        if (windowCursor >= windowSize) {
            fillWindow();
        }
        return window[windowCursor++];
    }

    private void fillWindow() throws IOException {
        windowSize = 0;
        windowCursor = 0;
        while (windowSize < window.length && delegate.hasNext()) {
            window[windowSize++] = delegate.next();
        }
        if (windowSize > 1) {
            Arrays.sort(window, 0, windowSize, (a, b) -> Long.compare(a.fileOffset(), b.fileOffset()));
        }
    }
}
