/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter.ClusterMembers;

import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies the {@link ClusterWriter} contract that is independent of any storage family: the
 * {@link ClusterMembers} parallel-array invariant. Per-implementation layout is covered by each
 * {@link ClusterWriter} implementation's own test (e.g. {@code OptimizedScalarQuantizedClusterWriterTest}).
 */
class ClusterWriterTest {

    @Test
    void clusterMembers_rejectsParallelArraysOfDifferentLength() {
        assertThrows(
            IllegalArgumentException.class,
            () -> new ClusterMembers(new int[] { 0, 1 }, new float[] { 1f, 2f }, new boolean[] { false })
        );
        assertThrows(
            IllegalArgumentException.class,
            () -> new ClusterMembers(new int[] { 0, 1 }, new float[] { 1f }, new boolean[] { false, true })
        );
    }
}
