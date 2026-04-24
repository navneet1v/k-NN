/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import java.io.IOException;

/**
 * Iterator over centroids to probe during IVF search.
 * Yields {@link ProbeTarget} in visit order. Decorators can reorder
 * or prefetch without changing the interface contract.
 */
public interface ProbeScheduler {
    boolean hasNext();

    ProbeTarget next() throws IOException;
}
