/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TwoPhaseIterator;

import java.io.IOException;

/**
 * Applies a query boost to another scorer's scores.
 *
 * <p>A wrapper rather than a boost parameter on the scorer itself, so the scorer being wrapped stays exactly as it is
 * shared with everything else that uses it.
 *
 * <p><b>The threshold has to be converted, not just forwarded.</b> A collector's competitive score arrives in boosted
 * space, while the wrapped scorer compares candidates against its own unboosted ones — so it is divided on the way in.
 * Forwarding it untouched would, at a boost above one, hand the scorer a bar higher than anything it can produce and
 * quietly prune away real hits; below one it would merely prune less than it could.
 *
 * <p>Nothing overrides the bulk scoring path: {@code Scorer#nextDocsAndScores} is built from {@link #iterator()},
 * {@link #docID()} and {@link #score()}, all of which are overridden here, so batches come out boosted without this
 * class knowing they exist.
 */
final class BoostedScorer extends Scorer {

    private final Scorer in;
    private final float boost;

    private BoostedScorer(final Scorer in, final float boost) {
        this.in = in;
        this.boost = boost;
    }

    /**
     * {@code scorer} boosted, or {@code scorer} itself when the boost cannot change anything.
     *
     * <p>A boost of exactly one is the common case and wrapping it would cost an indirection per score for nothing. A
     * boost of zero is passed through as well: every score becomes zero, so ordering is meaningless either way, and
     * dividing the threshold by it would ask the wrapped scorer to beat infinity.
     */
    static Scorer boosted(final Scorer scorer, final float boost) {
        if (scorer == null || boost == 1.0f || boost <= 0.0f) {
            return scorer;
        }
        return new BoostedScorer(scorer, boost);
    }

    @Override
    public int docID() {
        return in.docID();
    }

    @Override
    public DocIdSetIterator iterator() {
        return in.iterator();
    }

    @Override
    public TwoPhaseIterator twoPhaseIterator() {
        return in.twoPhaseIterator();
    }

    @Override
    public int advanceShallow(final int target) throws IOException {
        return in.advanceShallow(target);
    }

    @Override
    public float score() throws IOException {
        return in.score() * boost;
    }

    /**
     * The wrapped bound, boosted — and left alone when it is already the unbounded sentinel, since multiplying
     * {@link Float#MAX_VALUE} would turn a finite "no bound" into an infinity that reads as a different thing.
     */
    @Override
    public float getMaxScore(final int upTo) throws IOException {
        final float max = in.getMaxScore(upTo);
        return max == Float.MAX_VALUE ? Float.MAX_VALUE : max * boost;
    }

    @Override
    public void setMinCompetitiveScore(final float minScore) throws IOException {
        in.setMinCompetitiveScore(minScore / boost);
    }
}
