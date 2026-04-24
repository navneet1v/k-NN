/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import java.io.IOException;

/**
 * Iterator over centroids to probe during IVF search.
 * Yields {@link ProbedCentroid} in visit order. Decorators can reorder
 * or prefetch without changing the interface contract.
 */
interface ProbeIterator {
    boolean hasNext();

    ProbedCentroid next() throws IOException;
}
