/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Runs one stage of the pipeline to completion, and splits its result back up per segment.
 *
 * <p>Every stage but the last has to do this, because each one caps globally: how many candidates the <em>index</em>
 * contributes, not how many each segment does. A global cap can only be applied once every segment has reported, so a
 * stage that feeds another has to collect, and a stage that feeds the caller hands back a scorer instead.
 *
 * <p>Both halves are here because they are always used together: collect into global doc ids so they can be ordered
 * against each other, then split back into segment-local ones so the next stage can read vectors.
 */
final class LeafHits {

    private LeafHits() {}

    /**
     * Runs {@code weight} over every segment in parallel and merges the best {@code topN}.
     *
     * <p>Asking a segment for its scorer is what runs that segment's work, so this fan-out <em>is</em> the stage.
     *
     * <p>Each segment keeps at most {@code topN}, since that is the most it could contribute to a global top-{@code topN}.
     */
    static TopDocs collect(final IndexSearcher searcher, final List<LeafReaderContext> leaves, final Weight weight, final int topN)
        throws IOException {
        final List<Callable<TopDocs>> tasks = new ArrayList<>(leaves.size());
        for (final LeafReaderContext leaf : leaves) {
            tasks.add(() -> {
                final Scorer scorer = weight.scorer(leaf);
                return scorer == null ? TopDocsCollector.EMPTY_TOPDOCS : drain(scorer, leaf, topN);
            });
        }
        final TopDocs[] perLeaf = searcher.getTaskExecutor().invokeAll(tasks).toArray(TopDocs[]::new);
        return TopDocs.merge(topN, perLeaf);
    }

    /**
     * One segment's best {@code topN} hits, in index-wide doc ids so the merge can order them against other segments.
     *
     * <p>Bounded, because a segment cannot contribute more than {@code topN} to a global top-{@code topN} and keeping
     * more is waste. The heap's floor is pushed back into the scorer once it is full: today the approximate scan's scorer
     * ignores it — it is replaying an already-collected result — but a scorer that scores as it reads would use it to skip
     * whole batches, and this is where it would learn the bar.
     */
    private static TopDocs drain(final Scorer scorer, final LeafReaderContext leaf, final int topN) throws IOException {
        final HitQueue queue = new HitQueue(Math.max(topN, 1), true);
        ScoreDoc weakest = queue.top();
        final DocIdSetIterator iterator = scorer.iterator();
        int kept = 0;

        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            final float score = scorer.score();
            if (score > weakest.score) {
                weakest.score = score;
                weakest.doc = doc + leaf.docBase;
                weakest = queue.updateTop();
                // Only once the heap is full is its floor a real bound; before that it is a sentinel and pushing it
                // would prune against a score no document has to beat.
                if (++kept >= topN) {
                    scorer.setMinCompetitiveScore(weakest.score);
                }
            }
        }

        return collected(queue);
    }

    /**
     * Empties the heap into descending order, dropping the negative sentinels the queue was pre-filled with — a segment
     * with fewer than {@code topN} hits leaves the rest of them in place.
     */
    private static TopDocs collected(final HitQueue queue) {
        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }
        final ScoreDoc[] hits = new ScoreDoc[queue.size()];
        for (int i = hits.length - 1; i >= 0; i--) {
            hits[i] = queue.pop();
        }
        return new TopDocs(new TotalHits(hits.length, TotalHits.Relation.EQUAL_TO), hits);
    }

    /**
     * Splits merged hits into per-segment, segment-local, ascending doc ids.
     *
     * <p>Ascending because the next stage reads vectors forward: a list still in score order would turn every read into a
     * backwards seek through a file laid out by document.
     */
    static int[][] groupByLeaf(final TopDocs hits, final List<LeafReaderContext> leaves) {
        final List<List<Integer>> collected = new ArrayList<>(leaves.size());
        for (int i = 0; i < leaves.size(); i++) {
            collected.add(new ArrayList<>());
        }
        for (final ScoreDoc scoreDoc : hits.scoreDocs) {
            final int ord = ReaderUtil.subIndex(scoreDoc.doc, leaves);
            collected.get(ord).add(scoreDoc.doc - leaves.get(ord).docBase);
        }

        final int[][] byLeaf = new int[leaves.size()][];
        for (int ord = 0; ord < leaves.size(); ord++) {
            byLeaf[ord] = collected.get(ord).stream().mapToInt(Integer::intValue).toArray();
            Arrays.sort(byLeaf[ord]);
        }
        return byLeaf;
    }
}
