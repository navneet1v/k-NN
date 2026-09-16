/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.Bits;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.common.QueryUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Every child of the parents a previous stage settled on.
 *
 * <p>The stage before produced one child per parent — the nearest one. This turns each of those back into its whole nest,
 * which is what {@code expandNested} asks for: the parents were chosen by their best child, but the answer is all of
 * their children, each on its own score.
 *
 * <p>The parents are never named. A surviving child plus the parent bitset is enough to find its siblings, so the nest is
 * derived rather than carried.
 */
final class SiblingCandidates implements CandidateSource {

    /** Per segment: the children that survived, segment-local. One per parent, so also a count of parents. */
    private final int[][] survivorsByLeaf;

    private final BitSetProducer parentsFilter;

    /** The query's own filter, or {@code null}. See {@link #iterator} for what it does to a nest. */
    @Nullable
    private final Weight filterWeight;

    private final QueryUtils queryUtils;

    SiblingCandidates(
        final int[][] survivorsByLeaf,
        final BitSetProducer parentsFilter,
        @Nullable final Weight filterWeight,
        final QueryUtils queryUtils
    ) {
        this.survivorsByLeaf = survivorsByLeaf;
        this.parentsFilter = parentsFilter;
        this.filterWeight = filterWeight;
        this.queryUtils = queryUtils;
    }

    /**
     * The number of surviving parents, not of siblings — a lower bound, since every parent has at least the child that
     * got it here. The true count is not knowable without walking the nests, and the contract only requires that zero
     * means zero.
     */
    @Override
    public int size(final LeafReaderContext context) {
        return survivorsByLeaf[context.ord].length;
    }

    /**
     * The siblings, in doc order across all of this segment's surviving parents — so nests interleave, and a caller
     * cannot tell where one ends and the next begins. Nothing downstream needs to.
     *
     * <p><b>The filter is applied to the siblings.</b> A child the filter excludes is left out, so the nest that comes
     * back can be partial: filter on {@code color = red} and a parent with one red child and two blue ones yields one
     * child, not three. That is the existing behaviour of nested expansion, and it is a decision rather than an oversight
     * — the filter is taken to mean something about which children are wanted, not only about which parents qualify.
     */
    @Override
    public DocIdSetIterator iterator(final LeafReaderContext context) throws IOException {
        final int[] survivors = survivorsByLeaf[context.ord];
        if (survivors.length == 0) {
            return DocIdSetIterator.empty();
        }
        final Bits queryFilter = queryUtils.createBits(context, filterWeight);
        return queryUtils.getAllSiblings(context, asSet(survivors), parentsFilter, queryFilter);
    }

    /** Insertion-ordered, so the doc ids stay ascending for anything that iterates the set rather than the nests. */
    private static Set<Integer> asSet(final int[] docs) {
        final Set<Integer> set = new LinkedHashSet<>(Math.max(4, docs.length * 2));
        Arrays.stream(docs).forEach(set::add);
        return set;
    }
}
