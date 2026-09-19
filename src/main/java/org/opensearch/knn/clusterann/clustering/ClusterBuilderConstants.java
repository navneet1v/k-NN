/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.clustering;

/**
 * Default tuning parameters for {@link ClusterBuilder}.
 *
 * <p>These are the build-level knobs that were previously passed as method
 * arguments. Centralizing them keeps {@link ClusterBuilder#build} to the minimal
 * caller-facing surface (vectors, metric, executor) while documenting the
 * rationale for each default in one place.
 */
final class ClusterBuilderConstants {

    private ClusterBuilderConstants() {}

    /** Target number of vectors per cluster (posting-list size objective). */
    static final int TARGET_CLUSTER_SIZE = 512;

    /**
     * SOAR (Spilled Optimized Assignment Routing) lambda. Controls the
     * residual-orthogonality penalty when choosing a secondary centroid.
     * {@code > 0} enables SOAR; {@code 0} disables secondary assignments.
     */
    static final float SOAR_LAMBDA = 1.0f;

    /** Fixed RNG seed for reproducible k-means++ initialization and sampling. */
    static final long SEED = 42L;

    /** Max secondary-centroid candidates considered per vector during SOAR. */
    static final int SOAR_CANDIDATE_LIMIT = 10;

    /**
     * Maximum number of SOAR secondary assignments (spills) a single vector may receive.
     * {@code 1} (default) reproduces classic single-secondary SOAR. Values {@code > 1} let a
     * boundary vector spill into up to this many neighbor centroids; interior vectors still
     * collapse to primary-only via {@link #SOAR_SPILL_MARGIN}. Must be {@code >= 1}.
     */
    static final int MAX_SOAR_ASSIGNMENTS = 1;

    /**
     * Relative-score margin gating spills beyond the first secondary. A ranked candidate is kept
     * only while its SOAR score is within {@code (1 + SOAR_SPILL_MARGIN)} times the best (first)
     * secondary's score; the first ranked candidate below {@link #MAX_SOAR_ASSIGNMENTS} beyond it
     * that fails the test stops the spill for that vector. This is the "margin test" that keeps
     * interior vectors primary-only and only spills genuine boundary vectors. Ignored when
     * {@code MAX_SOAR_ASSIGNMENTS == 1}. {@code 0.0} means a spill is kept only on an exact score
     * tie with the best secondary.
     */
    static final float SOAR_SPILL_MARGIN = 0.0f;

    /**
     * Upper bound on the reservoir sample used to seed initial centroids.
     * A single-pass reservoir sample of up to this many vectors is drawn from the
     * input; the first {@code numCentroids} of them seed clustering, skipping the
     * cost of k-means++ initialization over the full dataset on large merges.
     */
    static final int RESERVOIR_SIZE = 4096;
}
