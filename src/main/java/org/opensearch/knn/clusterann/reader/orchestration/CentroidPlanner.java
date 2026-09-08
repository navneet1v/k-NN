/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.orchestration;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.opensearch.knn.clusterann.reader.CentroidVectorValues;
import org.opensearch.knn.clusterann.reader.Clusters;

import java.io.IOException;

/**
 * Chooses which clusters a query should visit.
 *
 * <p>Sweeps every centroid, keeps the {@code maxProbes} nearest non-empty clusters, and returns their ordinals
 * closest-first. The scan relies on that order and does not restore it.
 *
 * <p>Ranks on a distance-like key (smaller is closer)
 *
 * <p>Stateless and thread-safe: the only cursor it uses is a private copy taken per call.
 */
public final class CentroidPlanner {

    private static final int[] NO_PROBES = new int[0];

    private CentroidPlanner() {}

    /**
     * Rank this field's centroids against the query and name the clusters to visit.
     *
     * <p>Costs one pass over the centroid region ({@code numClusters × dimension} floats), before any posting is
     * touched. Empty clusters are dropped during the sweep so they never take up a probe slot.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within
     * @return centroid ordinals to probe, ordered closest-first; empty if the field has no non-empty cluster
     */
    public static int[] plan(Clusters clusters, float[] query, PlanParams params) throws IOException {
        int numClusters = clusters.numClusters();
        if (numClusters == 0) {
            return NO_PROBES;
        }

        final VectorSimilarityFunction similarity = clusters.clusterMeta().similarityFunction();
        final CentroidVectorValues centroids = clusters.centroids();
        float queryNormSq = similarity == VectorSimilarityFunction.EUCLIDEAN ? VectorUtil.dotProduct(query, query) : 0f;

        final NeighborQueue nearest = new NeighborQueue(params.maxProbes(), true);
        for (int ordinal = 0; ordinal < numClusters; ordinal++) {
            if (clusters.clusterSize(ordinal) != 0) {
                final float[] centroid = centroids.vectorValue(ordinal);
                nearest.insertWithOverflow(ordinal, distanceKey(similarity, query, queryNormSq, centroid, centroids));
            }
        }
        return closestFirst(nearest);
    }

    /**
     * Distance-like key from the query to a centroid, smaller meaning closer. Only EUCLIDEAN is a true distance:
     *
     * <ul>
     *   <li>EUCLIDEAN — {@code ‖q‖² − 2⟨q,c⟩ + ‖c‖²}, i.e. {@code d²}.
     *   <li>DOT_PRODUCT / MAXIMUM_INNER_PRODUCT — {@code −⟨q,c⟩}.
     *   <li>COSINE — {@code −⟨q,c⟩ / ‖c‖}; the norm matters because centroids of unit vectors are not unit norm.
     * </ul>
     */
    private static float distanceKey(
        VectorSimilarityFunction similarity,
        float[] query,
        float queryNormSq,
        float[] centroid,
        CentroidVectorValues centroids
    ) {
        float dot = VectorUtil.dotProduct(query, centroid);
        return switch (similarity) {
            case EUCLIDEAN -> queryNormSq - 2 * dot + normSq(centroids, centroid);
            case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> -dot;
            case COSINE -> -dot / (float) Math.sqrt(normSq(centroids, centroid));
        };
    }

    /**
     * {@code ‖c‖²} for the centroid: read from the record when the region stores one, else measured from the vector
     * (a correctness fallback that costs a second pass, not a cheap path).
     */
    private static float normSq(CentroidVectorValues centroids, float[] centroid) {
        return centroids.hasNorm() ? centroids.norm() : VectorUtil.dotProduct(centroid, centroid);
    }

    /** Drain the heap into ordinals, closest-first. {@code pop} yields farthest first, so fill from the back. */
    private static int[] closestFirst(final NeighborQueue nearest) {
        int[] probes = new int[nearest.size()];
        for (int i = probes.length - 1; i >= 0; i--) {
            probes[i] = nearest.pop();
        }
        return probes;
    }
}
