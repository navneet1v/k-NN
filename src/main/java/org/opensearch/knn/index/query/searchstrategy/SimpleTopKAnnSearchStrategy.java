/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.searchstrategy;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/**
 * A search strategy that is used to do pure topK search. Nothing else. This is just added so that we can make things
 * work. We need to revisit this.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class SimpleTopKAnnSearchStrategy extends KnnSearchStrategy {

    public static final SimpleTopKAnnSearchStrategy INSTANCE = new SimpleTopKAnnSearchStrategy();

    @Override
    public boolean equals(Object obj) {
        return obj instanceof SimpleTopKAnnSearchStrategy;
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }

    @Override
    public void nextVectorsBlock() {

    }
}
