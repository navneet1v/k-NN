/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

/**
 * Which documents of a segment an exact pass should score.
 *
 * <p>The only thing that differs between the two exact passes a nested query makes. Rescoring scores the candidates the
 * scan surfaced; expanding scores the siblings of whatever survived rescoring. Same reads, same batching, same threshold
 * feedback — different set of doc ids, so that is all this abstracts.
 */
interface CandidateSource {

    /**
     * How many documents this segment contributes, or {@code 0} when it contributes none.
     *
     * <p>Zero is the load-bearing answer: it means the segment is never opened. An exact number is welcome but not
     * required — anything non-zero is treated as an estimate for the scorer's cost.
     */
    int size(LeafReaderContext context) throws IOException;

    /**
     * A fresh iterator over those documents, ascending.
     *
     * <p>Fresh per call because a scorer consumes it, while the underlying set can be walked again. Ascending because the
     * scorer reads the segment's vectors forward — an out-of-order list would turn every read into a backwards seek
     * through a file laid out by document.
     */
    DocIdSetIterator iterator(LeafReaderContext context) throws IOException;
}
