/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.Getter;
import lombok.Setter;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;

/**
 * A per-document kNN numeric value.
 */
public class KNN80BinaryDocValues extends BinaryDocValues {

    private DocIDMerger<BinaryDocValuesSub> docIDMerger;

    @Setter
    private long cost;

    @Getter
    @Setter
    private long liveDocs;

    KNN80BinaryDocValues(DocIDMerger<BinaryDocValuesSub> docIdMerger) {
        this.docIDMerger = docIdMerger;
    }

    private BinaryDocValuesSub current;
    private int docID = -1;

    @Override
    public int docID() {
        return docID;
    }

    @Override
    public int nextDoc() throws IOException {
        current = docIDMerger.next();
        if (current == null) {
            docID = NO_MORE_DOCS;
        } else {
            docID = current.mappedDocID;
        }
        return docID;
    }

    @Override
    public int advance(int target) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean advanceExact(int target) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
        return cost;
    }

    @Override
    public BytesRef binaryValue() throws IOException {
        return current.getValues().binaryValue();
    }
};
