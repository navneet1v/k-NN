/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

/**
 * CLIP-style {@link PostingPruner} using provably-safe geometric bounds over the posting's sorted
 * {@code ‖c−v‖} column.
 *
 * <ul>
 *   <li><b>L2</b>: {@code ‖q−v‖² ≥ (‖q−c‖ − ‖c−v‖)²} (reverse triangle inequality). As a function of
 *       {@code x = ‖c−v‖} the bound is a parabola with vertex at {@code x* = ‖q−c‖}; the posting is
 *       sorted ascending by {@code x}, so hopeless blocks sit on both sides of the vertex — before it
 *       we {@code SKIP} (the competitive band is still ahead), at/after it we {@code TERMINATE}.</li>
 *   <li><b>IP / cosine</b>: {@code ⟨q,v⟩ ≤ ⟨q,c⟩ + ‖q‖·‖c−v‖} (Cauchy–Schwarz). Monotonic in {@code x};
 *       the posting is sorted so the ceiling is descending, so a below-threshold block means
 *       {@code TERMINATE}.</li>
 * </ul>
 *
 * <p>Bounds are computed from the block's {@code ‖c−v‖} range (its endpoints, since the column is
 * sorted) and the query geometry — never from block codes. Rotation is orthonormal, so the
 * transformed centroid / rotated query preserve {@code ‖q−c‖}, {@code ⟨q,c⟩}, {@code ‖q‖}.
 */
final class ClipPostingPruner implements PostingPruner {

    private final float[] dist; // ‖c−v‖ per position (sorted), read sequentially by the cluster
    private final VectorSimilarityFunction sim;
    private final boolean euclidean;
    private final float vertex; // ‖q−c‖ (L2 parabola vertex)
    private final float qc;     // ⟨q,c⟩ (IP/cosine)
    private final float qNorm;  // ‖q‖   (IP/cosine)

    ClipPostingPruner(float[] query, float[] centroid, float[] distancesToCentroid, VectorSimilarityFunction sim) {
        this.dist = distancesToCentroid;
        this.sim = sim;
        this.euclidean = sim == VectorSimilarityFunction.EUCLIDEAN;
        if (euclidean) {
            this.vertex = (float) Math.sqrt(Math.max(0f, VectorUtil.squareDistance(query, centroid)));
            this.qc = 0f;
            this.qNorm = 0f;
        } else {
            this.qc = VectorUtil.dotProduct(query, centroid);
            this.qNorm = (float) Math.sqrt(Math.max(0f, VectorUtil.dotProduct(query, query)));
            this.vertex = 0f;
        }
    }

    @Override
    public Decision inspect(int blockStart, int blockLen, float minCompetitiveSimilarity) {
        if (blockLen <= 0) {
            return Decision.SKIP;
        }
        // The column is sorted, so the block's ‖c−v‖ range is bracketed by its endpoints.
        float a = dist[blockStart];
        float b = dist[blockStart + blockLen - 1];
        float lo = Math.min(a, b);
        float hi = Math.max(a, b);

        if (euclidean) {
            float xClosest = Math.max(lo, Math.min(vertex, hi)); // point in [lo,hi] nearest the vertex
            float gap = xClosest - vertex;
            float lb = gap * gap;                                // lower bound on ‖q−v‖²
            float maxSim = 1f / (1f + Math.max(lb, 0f));         // upper bound on similarity
            if (maxSim >= minCompetitiveSimilarity) {
                return Decision.SCORE;
            }
            // Below the vertex (band still ahead) → SKIP; at/after the vertex → TERMINATE.
            return hi < vertex ? Decision.SKIP : Decision.TERMINATE;
        }

        // IP / cosine: highest ceiling in the block is at the largest ‖c−v‖.
        float ceiling = qc + qNorm * hi;
        float maxSim = sim == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            ? (ceiling >= 0 ? ceiling + 1f : 1f / (1f - ceiling))
            : Math.max((1f + ceiling) / 2f, 0f);
        if (maxSim >= minCompetitiveSimilarity) {
            return Decision.SCORE;
        }
        return Decision.TERMINATE; // ceiling descending along the posting
    }
}
