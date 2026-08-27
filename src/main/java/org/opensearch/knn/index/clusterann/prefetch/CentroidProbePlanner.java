/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.prefetch;

import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.codec.CentroidVectorValues;

import java.io.IOException;
import java.util.Arrays;

/**
 * Plans which centroids to probe for a query: scores every centroid geometrically against the query
 * (via {@link CentroidVectorValues}), selects the nearest set via the "knee" in the sorted-distance
 * curve, and returns their ordinals closest-first together with the per-centroid geometric quantity
 * that ranking and CLIP consume — {@code ‖q−c‖²} for L2, {@code ⟨q,c⟩} for inner product.
 *
 * <p>Ranking is done in distance space (not similarity): the downstream consumer is CLIP, which is
 * geometric, so the planner produces {@code ⟨q,c⟩}/{@code ‖q−c‖} directly rather than collapsing to
 * a similarity and inverting it back. Stateless: all work happens in {@link #plan}.
 */
public final class CentroidProbePlanner {

    private CentroidProbePlanner() {}

    // nprobe multiplier default. The configurable per-index value lives in the index setting
    // KNNSettings.KNN_ALGO_PARAM_CLUSTERANN_NPROBE_MULTIPLIER (see getClusterANNNprobeMultiplier).
    // TODO(future): resolve that per-index value at the ClusterANN query layer and thread it into
    // plan() (the codec read path has no index context here), and allow a per-query override —
    // mirroring ef_search. Until then, this process-wide default/override is the interim seam.
    private static volatile int NPROBE_MULTIPLIER = KNNSettings.INDEX_KNN_DEFAULT_CLUSTERANN_NPROBE_MULTIPLIER;

    /**
     * The selected probes, closest-first.
     *
     * @param ordinals centroid ordinals to probe (length == nprobe)
     * @param geom     per-probe geometric quantity aligned with {@code ordinals}:
     *                 {@code ‖q−c‖²} for L2, {@code ⟨q,c⟩} for inner product
     */
    record Ranking(int[] ordinals, float[] geom) {}

    /**
     * Ranks all centroids against {@code query} under {@code metric} and returns the nearest set to
     * probe (closest-first), with the geometric quantity CLIP will use. The metric is supplied by the
     * caller — {@link CentroidVectorValues} is only a vector store and does not interpret it.
     */
    private static Ranking plan(CentroidVectorValues values, float[] query, DistanceMetric metric, int k) throws IOException {
        int n = values.size();
        float[] geom = new float[n];    // ‖q−c‖² (L2) or ⟨q,c⟩ (IP)
        float[] rankKey = new float[n]; // distance-like: lower == closer
        computeGeometry(values, query, metric, geom, rankKey);

        int[] ordsClosestFirst = sortAscending(rankKey, n);
        float[] sortedKey = new float[n];
        for (int i = 0; i < n; i++) {
            sortedKey[i] = rankKey[ordsClosestFirst[i]];
        }

        int nprobe = calculateNprobe(sortedKey, n, k);
        int[] probeOrds = new int[nprobe];
        float[] probeGeom = new float[nprobe];
        for (int i = 0; i < nprobe; i++) {
            int c = ordsClosestFirst[i];
            probeOrds[i] = c;
            probeGeom[i] = geom[c];
        }
        return new Ranking(probeOrds, probeGeom);
    }

    /**
     * The centroid ordinals to probe, closest-first — the ready-to-walk probe list. Public seam for the
     * codec read path (which is in another package).
     *
     * <p>Ordinals alone are enough: a cluster is reached by ordinal, and it resolves its own offsets and
     * sizes. The geometry {@link #plan} also computes is not returned — nothing consumes it yet, though a
     * centroid-level bound (skipping a whole cluster before opening its posting) would.
     */
    public static int[] planProbes(CentroidVectorValues values, float[] query, DistanceMetric metric, int k)
        throws IOException {
        return plan(values, query, metric, k).ordinals();
    }

    /**
     * Scores every centroid: fills {@code geom} with the CLIP quantity ({@code ‖q−c‖²} for L2,
     * {@code ⟨q,c⟩} otherwise) and {@code rankKey} with a distance-like value (lower == closer) used
     * for sorting/knee. {@link CentroidVectorValues#vectorValue} reads each centroid's vector and its
     * {@code ‖c‖²} in one seek; the norm is then available via {@link CentroidVectorValues#norm()}.
     */
    private static void computeGeometry(CentroidVectorValues values, float[] query, DistanceMetric metric,
                                        float[] geom, float[] rankKey) throws IOException {
        int n = values.size();
        boolean euclidean = metric == DistanceMetric.L2;
        float queryNormSq = euclidean ? VectorUtil.dotProduct(query, query) : 0f;
        for (int c = 0; c < n; c++) {
            float dot = VectorUtil.dotProduct(query, values.vectorValue(c));
            if (euclidean) {
                float d2 = Math.max(0f, queryNormSq + values.norm() - 2f * dot); // ‖q−c‖²
                geom[c] = d2;
                rankKey[c] = d2;
            } else {
                geom[c] = dot;      // ⟨q,c⟩ (higher == closer)
                rankKey[c] = -dot;  // distance-like (lower == closer)
            }
        }
    }

    /** Centroid ordinals sorted by {@code key} ascending (closest first). */
    private static int[] sortAscending(float[] key, int n) {
        long[] packed = new long[n];
        for (int c = 0; c < n; c++) {
            packed[c] = ((long) sortableBits(key[c]) << 32) | (c & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        int[] ords = new int[n];
        for (int i = 0; i < n; i++) {
            ords[i] = (int) packed[i];
        }
        return ords;
    }

    /** Monotonic int mapping of a float so ascending int order == ascending float order. */
    private static int sortableBits(float v) {
        int bits = Float.floatToIntBits(v);
        return bits ^ ((bits >> 31) | 0x80000000);
    }

    /**
     * Adaptive nprobe via the "knee" in the sorted-distance curve: walk from the closest centroid and
     * stop at the first sharp jump in distance (a cluster boundary). {@code sortedKey} is ascending.
     */
    private static int calculateNprobe(float[] sortedKey, int numCentroids, int k) {
        if (numCentroids <= 10) return numCentroids;

        int maxNprobe = Math.min(NPROBE_MULTIPLIER * (int) Math.sqrt(numCentroids), numCentroids);
        int minNprobe = Math.max(10, (int) Math.sqrt(numCentroids));

        float closest = sortedKey[0];
        float range = sortedKey[maxNprobe - 1] - closest;
        if (range <= 0) return maxNprobe;

        float avgStep = range / maxNprobe;
        int adaptiveNprobe = maxNprobe;
        for (int i = minNprobe; i < maxNprobe - 1; i++) {
            float step = sortedKey[i + 1] - sortedKey[i];
            if (step > avgStep * 3.0f) {
                adaptiveNprobe = i + 1;
                break;
            }
        }

        return Math.max(minNprobe, adaptiveNprobe);
    }
}
