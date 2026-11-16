/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.opensearch.knn.clusterann.read.CentroidVectorValues;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Chooses which clusters a query should visit.
 *
 * <p>Sweeps every centroid, keeps the {@code maxProbes} nearest non-empty clusters, and returns their ordinals
 * closest-first. Under a filter whose accepted vectors all fit in {@code k}, it keeps every cluster holding one
 * instead; under any other filter it goes past {@code maxProbes} when that is what holding {@code k} accepted vectors
 * takes. The scan relies on that order and does not restore it.
 *
 * <p>Ranks on a distance-like key (smaller is closer)
 *
 * <p>The ceiling bounds what a query may spend, unless a filter leaves it short of {@code k};
 * {@link CentroidKneePruner} cuts what it should not. Filtered plans are cut too, with a selectivity-scaled knee
 * factor, but never short of {@code k} accepted vectors.
 *
 * <p>Stateless and thread-safe: the only cursor it uses is a private copy taken per call.
 */
public final class CentroidPlanner {

    private static final int[] NO_PROBES = new int[0];

    private CentroidPlanner() {}

    /**
     * Rank this field's centroids against the query and name the clusters to visit, considering only clusters reached
     * by the document-space filter when one is present.
     *
     * <p>A filter that is null or has null {@link AcceptDocs#bits()} takes the unfiltered path without reading
     * ordinal-to-centroid assignments. Lucene drops a filter that rewrites to match-all and hands each leaf its live
     * docs, so a match-all filter on a segment without deletions arrives here with null bits. Any other filter takes
     * the filtered path, even if it happens to accept every document.
     *
     * <p>When every accepted vector fits in {@code k}, the caller wants all of them, so every centroid holding one is
     * probed rather than only the {@code maxProbes} nearest.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within; a selective filter may raise the ceiling above {@code maxProbes}
     * @param k the number of results the caller wants
     * @param acceptDocs document-space filter, or null for an unfiltered search
     * @param vectorDocsSupplier a new iterator over the documents that have this vector field, exposing each one's
     *     vector ordinal; called only when a sparse field is filtered
     * @return centroid ordinals to probe, ordered closest-first; empty if no non-empty cluster matches
     */
    public static int[] plan(
        Clusters clusters,
        float[] query,
        PlanParams params,
        int k,
        AcceptDocs acceptDocs,
        IOSupplier<KnnVectorValues.DocIndexIterator> vectorDocsSupplier
    ) throws IOException {
        if (clusters.numClusters() == 0) {
            return NO_PROBES;
        }
        if (acceptDocs == null || acceptDocs.bits() == null) {
            return plan(clusters, query, params);
        }

        final LongValues ordToCentroid = clusters.ordToCentroid();
        final FixedBitSet acceptedCentroids = new FixedBitSet(clusters.numClusters());
        final int[] acceptedCountsPerCluster = new int[clusters.numClusters()];
        final DocIdSetIterator acceptedDocs = acceptDocs.iterator();
        int acceptedVectors = 0;
        if (clusters.hasDenseVectorValues()) {
            for (int doc = acceptedDocs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = acceptedDocs.nextDoc()) {
                final int centroid = (int) ordToCentroid.get(doc);
                acceptedCentroids.set(centroid);
                acceptedCountsPerCluster[centroid]++;
                acceptedVectors++;
            }
        } else {
            final KnnVectorValues.DocIndexIterator vectorDocs = vectorDocsSupplier.get();
            final DocIdSetIterator acceptedVectorDocs = ConjunctionUtils.intersectIterators(List.of(acceptedDocs, vectorDocs));
            while (acceptedVectorDocs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                final int centroid = (int) ordToCentroid.get(vectorDocs.index());
                acceptedCentroids.set(centroid);
                acceptedCountsPerCluster[centroid]++;
                acceptedVectors++;
            }
        }
        final int eligibleClusters = acceptedCentroids.cardinality();
        // True count of clusters containing at least one accepted doc (the filter's eligible set), recorded here
        // where it is known — distinct from the planned probe count the reader records as PROBES_PLANNED.
        ClusterANNStat.FILTER_ELIGIBLE_CLUSTERS.record(eligibleClusters);
        if (eligibleClusters == 0) {
            return NO_PROBES;
        }
        final double selectivity = (double) acceptedVectors / clusters.numVectors();
        final int budget = PlanParams.forSelectivity(clusters.numClusters(), selectivity).maxProbes();
        // Never cut when the caller wants every accepted vector.
        if (acceptedVectors <= k) {
            return closestFirst(rank(clusters, query, eligibleClusters, acceptedCentroids, acceptedCountsPerCluster, k));
        }
        final int maxProbes = Math.max(budget, params.maxProbes());
        final NeighborQueue nearest = rank(clusters, query, maxProbes, acceptedCentroids, acceptedCountsPerCluster, k);
        if (nearest.size() > maxProbes) {
            ClusterANNStat.FILTER_EXTENDED_PAST_MAX_PROBES.increment();
        }
        return closestFirstPruned(nearest, params.minProbes(), selectivity, acceptedCountsPerCluster, k);
    }

    /**
     * Rank every centroid in this field against the query and name the clusters to visit.
     *
     * <p>Costs one pass over the centroid region ({@code numClusters × dimension} floats), before any posting is
     * touched. Empty clusters are dropped during the sweep so they never take up a probe slot.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within
     * @return centroid ordinals to probe, ordered closest-first; empty if the field has no non-empty cluster
     */
    static int[] plan(Clusters clusters, float[] query, PlanParams params) throws IOException {
        if (clusters.numClusters() == 0) {
            return NO_PROBES;
        }
        return closestFirstPruned(rank(clusters, query, params.maxProbes(), null, null, 0), params.minProbes(), 1.0, null, 0);
    }

    /**
     * Rank the candidate centroids, keeping the {@code maxProbes} nearest non-empty ones; given accepted counts, the
     * nearest that fill {@code maxProbes} and hold {@code k} accepted vectors, which may be more.
     *
     * @param eligibleCentroids ordinals to consider, or null when nothing is filtered out
     * @param acceptedCountsPerCluster accepted vectors per cluster, or null to keep at most {@code maxProbes}
     * @param k accepted vectors to hold; read only with {@code acceptedCountsPerCluster}
     */
    static NeighborQueue rank(
        Clusters clusters,
        float[] query,
        int maxProbes,
        FixedBitSet eligibleCentroids,
        int[] acceptedCountsPerCluster,
        int k
    ) throws IOException {
        final int numClusters = clusters.numClusters();
        final VectorSimilarityFunction similarity = clusters.clusterMeta().similarityFunction();
        final CentroidVectorValues centroids = clusters.centroids();
        final float queryNormSq = similarity == VectorSimilarityFunction.EUCLIDEAN ? VectorUtil.dotProduct(query, query) : 0f;

        // A cap for insertWithOverflow; only the initial capacity for add(), which grows past it.
        final NeighborQueue nearest = new NeighborQueue(maxProbes, true);
        // Accepted vectors in the clusters the heap holds.
        long held = 0;
        for (int ordinal = nextCandidate(eligibleCentroids, 0); ordinal < numClusters; ordinal = nextCandidate(
            eligibleCentroids,
            ordinal + 1
        )) {
            if (clusters.clusterSize(ordinal) != 0) {
                final float[] centroid = centroids.vectorValue(ordinal);
                final float key = distanceKey(similarity, query, queryNormSq, centroid, centroids.norm());
                if (acceptedCountsPerCluster == null) {
                    nearest.insertWithOverflow(ordinal, key);
                } else {
                    nearest.add(ordinal, key);
                    held += acceptedCountsPerCluster[ordinal];
                    // Keep the nearest clusters that fill the budget and hold k.
                    while (nearest.size() > maxProbes && held - acceptedCountsPerCluster[nearest.topNode()] >= k) {
                        held -= acceptedCountsPerCluster[nearest.pop()];
                    }
                }
            }
        }

        return nearest;
    }

    /** The first candidate ordinal at or after {@code from}: {@code from} itself when nothing is filtered out. */
    private static int nextCandidate(final FixedBitSet eligible, final int from) {
        if (eligible == null) {
            return from;
        }
        return from < eligible.length() ? eligible.nextSetBit(from) : DocIdSetIterator.NO_MORE_DOCS;
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
        float centroidNorm
    ) {
        float dot = VectorUtil.dotProduct(query, centroid);
        return switch (similarity) {
            case EUCLIDEAN -> queryNormSq - 2 * dot + centroidNorm;
            case MAXIMUM_INNER_PRODUCT -> -dot;
            case DOT_PRODUCT -> throw new IllegalStateException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
            case COSINE -> -dot / (float) Math.sqrt(centroidNorm);
        };
    }

    /** Drain the heap into ordinals, closest-first. */
    private static int[] closestFirst(final NeighborQueue nearest) {
        final int[] probes = new int[nearest.size()];
        for (int i = probes.length - 1; i >= 0; i--) {
            probes[i] = nearest.pop();
        }
        return probes;
    }

    /**
     * Drain the heap into ordinals, closest-first, cutting at the knee.
     *
     * @param selectivity fraction of vectors the filter accepts, which sets the knee factor; 1 when unfiltered
     * @param acceptedCountsPerCluster accepted vectors per cluster, or null when unfiltered
     * @param k results wanted; read only with {@code acceptedCountsPerCluster}
     */
    private static int[] closestFirstPruned(
        final NeighborQueue nearest,
        final int minProbes,
        final double selectivity,
        final int[] acceptedCountsPerCluster,
        final int k
    ) {
        final int ranked = nearest.size();
        final int[] probes = new int[ranked];
        final float[] distances = new float[ranked];
        for (int i = ranked - 1; i >= 0; i--) {
            distances[i] = nearest.topScore();
            probes[i] = nearest.pop();
        }

        final int nearestHoldingK = acceptedCountsPerCluster == null ? 0 : holdingK(probes, 0, acceptedCountsPerCluster, k);
        if (nearestHoldingK == ranked) {
            return probes;
        }

        final int keep = Math.max(CentroidKneePruner.prune(distances, minProbes, selectivity), nearestHoldingK);
        if (keep < ranked) {
            // The knee pruner cut the plan short: the adaptive early-stop fired for this query.
            ClusterANNStat.CENTROID_KNEE_EARLY_STOP.increment();
        }
        return keep == ranked ? probes : Arrays.copyOf(probes, keep);
    }

    /**
     * Length of the shortest prefix of {@code probes} holding {@code k} accepted vectors, or all of them when the
     * ranked clusters hold fewer. Primary assignments only.
     */
    private static int holdingK(final int[] probes, final int keep, final int[] acceptedCountsPerCluster, final int k) {
        long held = 0;
        for (int i = 0; i < keep; i++) {
            held += acceptedCountsPerCluster[probes[i]];
        }
        int extended = keep;
        while (held < k && extended < probes.length) {
            held += acceptedCountsPerCluster[probes[extended++]];
        }
        return extended;
    }
}
