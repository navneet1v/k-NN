/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * A fixed set of doc ids per segment — the candidates a completed pass left behind.
 *
 * <p>Segment-local and ascending, because that is the order the exact pass reads vectors in. Sorting happens where the
 * lists are built, since a globally-merged result arrives in score order.
 */
final class PerLeafDocIds implements CandidateSource {

    private final int[][] byLeaf;

    PerLeafDocIds(final int[][] byLeaf) {
        this.byLeaf = byLeaf;
    }

    @Override
    public int size(final LeafReaderContext context) {
        return byLeaf[context.ord].length;
    }

    @Override
    public DocIdSetIterator iterator(final LeafReaderContext context) {
        return new SortedDocIdSetIterator(byLeaf[context.ord]);
    }

    /** The list as an iterator. One per call, since a scorer consumes it while the array can be walked again. */
    private static final class SortedDocIdSetIterator extends DocIdSetIterator {

        private final int[] docs;
        private int index = -1;

        SortedDocIdSetIterator(final int[] docs) {
            this.docs = docs;
        }

        @Override
        public int docID() {
            if (index < 0) {
                return -1;
            }
            return index >= docs.length ? NO_MORE_DOCS : docs[index];
        }

        @Override
        public int nextDoc() {
            index++;
            return docID();
        }

        @Override
        public int advance(final int target) {
            while (nextDoc() < target && docID() != NO_MORE_DOCS) {
                // Linear, and deliberately so: the list is at most the first-pass k, and it is always the lead iterator,
                // so nothing ever seeks into it from a longer one.
            }
            return docID();
        }

        @Override
        public long cost() {
            return docs.length;
        }
    }
}
