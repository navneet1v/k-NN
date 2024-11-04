/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.service.OSLuceneDocId;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNNWeightV2 extends Weight {

    private final KNNQuery knnQuery;
    private final List<OSLuceneDocId> osLuceneDocIds;

    public KNNWeightV2(KNNQuery knnQuery, List<OSLuceneDocId> osLuceneDocIds) {
        super(knnQuery);
        this.knnQuery = knnQuery;
        this.osLuceneDocIds = osLuceneDocIds;
    }

    /**
     * An explanation of the score computation for the named document.
     *
     * @param context the readers context to create the {@link Explanation} for.
     * @param doc     the document's id relative to the given context's reader
     * @return an Explanation for the score
     * @throws IOException if an {@link IOException} occurs
     */
    @Override
    public Explanation explain(LeafReaderContext context, int doc) throws IOException {
        return null;
    }

    /**
     * Returns a {@link Scorer} which can iterate in order over all matching documents and assign them
     * a score.
     *
     * <p><b>NOTE:</b> null can be returned if no documents will be scored by this query.
     *
     * <p><b>NOTE</b>: The returned {@link Scorer} does not have {@link LeafReader#getLiveDocs()}
     * applied, they need to be checked on top.
     *
     * @param context the {@link LeafReaderContext} for which to return the
     *                {@link Scorer}.
     * @return a {@link Scorer} which scores documents in/out-of order.
     * @throws IOException if there is a low-level I/O error
     */
    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
        SegmentReader segmentReader = Lucene.segmentReader(context.reader());
        byte[] segmentId = segmentReader.getSegmentInfo().info.getId();
        Map<Integer, Float> docIdsToScoreMap = new HashMap<>();
        for (OSLuceneDocId osLuceneDocId : osLuceneDocIds) {
            if (Arrays.equals(osLuceneDocId.getSegmentId(), segmentId)) {
                docIdsToScoreMap.put(osLuceneDocId.getSegmentDocId(), osLuceneDocId.getScore());
            }
        }
        if (docIdsToScoreMap.isEmpty()) {
            return KNNScorer.emptyScorer(this);
        }
        final int maxDoc = Collections.max(docIdsToScoreMap.keySet()) + 1;
        return new KNNScorer(this, ResultUtil.resultMapToDocIds(docIdsToScoreMap, maxDoc), docIdsToScoreMap, 1);
    }

    /**
     * @param ctx
     * @return {@code true} if the object can be cached against a given leaf
     */
    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
        return true;
    }
}
