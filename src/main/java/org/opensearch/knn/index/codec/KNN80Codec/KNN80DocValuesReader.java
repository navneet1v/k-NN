/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.EmptyDocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Reader for KNNDocValues from the segments
 */
@Getter
@Log4j2
class KNN80DocValuesReader extends EmptyDocValuesProducer {

    private final MergeState mergeState;

    KNN80DocValuesReader(MergeState mergeState) {
        this.mergeState = mergeState;
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) {
        long cost = 0;
        long liveDocsCount = 0;
        try {
            List<BinaryDocValuesSub> subs = new ArrayList<>(this.mergeState.docValuesProducers.length);
            for (int i = 0; i < this.mergeState.docValuesProducers.length; i++) {
                BinaryDocValues values = null;
                DocValuesProducer docValuesProducer = mergeState.docValuesProducers[i];
                if (docValuesProducer != null) {
                    FieldInfo readerFieldInfo = mergeState.fieldInfos[i].fieldInfo(field.name);
                    if (readerFieldInfo != null && readerFieldInfo.getDocValuesType() == DocValuesType.BINARY) {
                        values = docValuesProducer.getBinary(readerFieldInfo);
                    }
                    if (values != null) {
                        cost += values.cost();
                        Bits liveDocs = this.mergeState.liveDocs[i];
                        if (liveDocs != null) {
                            log.info("There are some deleted docs present");
                            // so we counted all the live docs here
                            int docId;
                            for(docId = values.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId =
                                    values.nextDoc()) {
                                if (liveDocs.get(docId)) {
                                    liveDocsCount++;
                                }
                            }
                            // again setting this value as we have already used the older doc values.
                            values = docValuesProducer.getBinary(readerFieldInfo);
                        } else {
                            // no live docs are present so lets use all the docs.
                            liveDocsCount += values.cost();
                        }
                        subs.add(new BinaryDocValuesSub(mergeState.docMaps[i], values));
                    }
                }
            }
            KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(DocIDMerger.of(subs, mergeState.needsIndexSort));
            knn80BinaryDocValues.setCost(cost);
            knn80BinaryDocValues.setLiveDocs(liveDocsCount);
            log.info("There are {} live docs, {} cost", liveDocsCount, cost);
            return knn80BinaryDocValues;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
