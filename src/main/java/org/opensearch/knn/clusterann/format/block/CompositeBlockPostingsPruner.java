/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.block;

/** Several pruners as one. Built through {@link BlockPostingsPruner#of}. */
final class CompositeBlockPostingsPruner implements BlockPostingsPruner {

    private final BlockPostingsPruner[] pruners;

    CompositeBlockPostingsPruner(final BlockPostingsPruner[] pruners) {
        this.pruners = pruners;
    }

    @Override
    public Decision test(final int block, final float minCompetitiveScore) {
        Decision strongest = Decision.SCORE;
        for (BlockPostingsPruner pruner : pruners) {
            Decision decision = pruner.test(block, minCompetitiveScore);
            if (decision == Decision.TERMINATE) {
                return Decision.TERMINATE;
            }
            if (decision == Decision.SKIP) {
                // Deliberately not returned early: a later pruner may still terminate, and losing that would
                // cost the whole tail of the posting, one skipped block at a time.
                strongest = Decision.SKIP;
            }
        }
        return strongest;
    }
}
