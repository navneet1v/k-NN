/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingNearestChildrenKnnCollectorManager;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.MultiLeafKnnCollector;
import org.apache.lucene.util.hnsw.BlockingFloatHeap;
import org.opensearch.common.Nullable;

import java.io.IOException;

/**
 * Builds one collector per segment: all of them sharing a competitive threshold, and each keeping only a parent's best
 * child when the field is nested.
 *
 * <p>The two compose rather than compete, because parent dedup is a collector and so can sit inside the one that shares
 * the threshold. Without that, a nested query would have to choose between correct semantics and useful pruning.
 *
 * <p><b>Why the threshold is shared.</b> A segment that starts late learns the score it has to beat from the segments
 * already finished, so it prunes from its first cluster rather than rediscovering the same bar. Segment-local heaps leave
 * every bound uselessly loose until each one has filled its own.
 *
 * <p><b>Why parent dedup belongs here.</b> A parent holds many child vectors, and plain top-k over children can return k
 * children of one parent. Nested asks for k <em>parents</em>, each represented by its nearest child — which is a decision
 * about what to keep, so it is the collector's.
 *
 * <p>{@link KnnCollectorManager#isOptimistic()} is left at its default of {@code false}, deliberately. Lucene's
 * optimistic mode runs a first pass with a reduced per-leaf k and then re-enters the segments whose results reached the
 * global bar. That is a good trade for a graph, where the cost is nodes visited; a cluster scan's cost is the postings it
 * reads, and re-entering a segment means reading them again. So the second pass is not switched off here — it is never
 * asked for.
 */
final class ClusterANNCollectorManager implements KnnCollectorManager {

    private final int k;

    /** Shared across every segment of one query, which is the whole point. */
    private final BlockingFloatHeap globalThreshold;

    /** Builds the per-parent-best-child collectors, or {@code null} when the field is not nested. */
    @Nullable
    private final DiversifyingNearestChildrenKnnCollectorManager diversifying;

    ClusterANNCollectorManager(final int k, @Nullable final BitSetProducer parentsFilter, final IndexSearcher searcher) {
        this.k = k;
        this.globalThreshold = new BlockingFloatHeap(k);
        this.diversifying = parentsFilter == null ? null : new DiversifyingNearestChildrenKnnCollectorManager(k, parentsFilter, searcher);
    }

    /**
     * @return {@code null} when this segment holds no parent documents, which the caller must treat as "nothing to scan
     *     here" rather than as an empty result — a nested field's segment without parents has no children either.
     */
    @Override
    @Nullable
    public KnnCollector newCollector(final int visitedLimit, final KnnSearchStrategy searchStrategy, final LeafReaderContext context)
        throws IOException {
        final AbstractKnnCollector perSegment = perSegmentCollector(visitedLimit, searchStrategy, context);
        if (perSegment == null) {
            return null;
        }
        return new MultiLeafKnnCollector(k, globalThreshold, perSegment);
    }

    /**
     * The collector that decides what one segment keeps.
     *
     * <p>The cast is against a type Lucene keeps package-private and hands out only through its manager, so it cannot be
     * named here. It holds because that manager documents what it returns, and it is worth taking: the alternative is
     * giving up either the shared threshold or parent dedup, since only an {@link AbstractKnnCollector} can sit inside
     * {@link MultiLeafKnnCollector}.
     */
    @Nullable
    private AbstractKnnCollector perSegmentCollector(
        final int visitedLimit,
        final KnnSearchStrategy searchStrategy,
        final LeafReaderContext context
    ) throws IOException {
        if (diversifying == null) {
            return new TopKnnCollector(k, visitedLimit, searchStrategy);
        }
        return (AbstractKnnCollector) diversifying.newCollector(visitedLimit, searchStrategy, context);
    }
}
