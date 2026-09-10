package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.util.Random;

/**
 * Lloyd's k-means with k-means++ seeding, plus SOAR spills — a straightforward {@link Clustering} so the format has
 * one and can be exercised end to end.
 *
 * <p>Deliberately unsophisticated. The point of {@link Clustering} being an interface is that the written bytes depend
 * only on the result, so a hierarchical or sampled implementation drops in here with nothing downstream changing. A
 * better clustering makes postings more selective and pruning more effective; it cannot make a segment unreadable.
 *
 * <p>Distances are Euclidean regardless of the field's metric. Under inner product a cluster is still a region of
 * space, and the stored distance column, the residuals and every geometric bound are all Euclidean — so partitioning
 * by anything else would put the postings in a different space from the bounds computed over them.
 *
 * <h2>SOAR</h2>
 *
 * <p>A vector near a boundary is listed in a second cluster so it can be found from either side. The second is chosen
 * to minimise {@code ‖r₂‖² + λ·⟨r₁,r₂⟩²/‖r₁‖²}: among nearby centroids, the one whose residual points somewhere
 * <em>different</em> from the primary residual. A second copy whose residual is parallel to the first quantizes to
 * much the same error and buys little; an orthogonal one gives the scan a genuinely different estimate.
 */
public final class LloydClustering implements Clustering {

    /** Iterations of Lloyd's algorithm. Assignments stop moving well before this on real data. */
    private static final int MAX_ITERATIONS = 15;

    /** Candidate centroids considered as a SOAR spill, nearest first. */
    private static final int SOAR_CANDIDATES = 8;

    @Override
    public ClusteringResult cluster(VectorSource vectors, VectorSimilarityFunction similarity, ClusterANNWriteParams params)
        throws IOException {

        int count = vectors.size();
        int dimension = vectors.dimension();
        int k = params.centroidCount(count);
        if (k == 0) {
            return new ClusteringResult(new float[0][], new int[0], new int[0]);
        }

        float[][] all = materialise(vectors, count, dimension);
        Random random = new Random(params.seed());
        float[][] centroids = seed(all, k, random);

        int[] primary = new int[count];
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            boolean moved = assign(all, centroids, primary);
            recentre(all, centroids, primary, random);
            if (!moved) {
                break;
            }
        }
        // The last recentre moved the centroids, so assignments are re-derived against where they finally are. Without
        // this a vector could be listed in a cluster it is no longer nearest to, and its residual would be larger than
        // the encoding assumes.
        assign(all, centroids, primary);

        int[] soar = spill(all, centroids, primary, params.soarLambda(), k);
        return new ClusteringResult(centroids, primary, soar);
    }

    /**
     * k-means++ seeding: the first centroid uniformly, each next one with probability proportional to its squared
     * distance from the nearest chosen centroid. It costs a pass per centroid and buys a start that Lloyd's can
     * actually improve on, where uniform seeding often leaves clusters collapsed.
     */
    private static float[][] seed(float[][] vectors, int k, Random random) {
        float[][] centroids = new float[k][];
        centroids[0] = vectors[random.nextInt(vectors.length)].clone();

        float[] nearest = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            nearest[i] = VectorUtil.squareDistance(vectors[i], centroids[0]);
        }

        for (int chosen = 1; chosen < k; chosen++) {
            double total = 0;
            for (float distance : nearest) {
                total += distance;
            }
            int pick;
            if (total <= 0) {
                // Every remaining vector coincides with a chosen centroid, so weights carry no information.
                pick = random.nextInt(vectors.length);
            } else {
                double target = random.nextDouble() * total;
                double running = 0;
                pick = vectors.length - 1;
                for (int i = 0; i < vectors.length; i++) {
                    running += nearest[i];
                    if (running >= target) {
                        pick = i;
                        break;
                    }
                }
            }
            centroids[chosen] = vectors[pick].clone();
            for (int i = 0; i < vectors.length; i++) {
                nearest[i] = Math.min(nearest[i], VectorUtil.squareDistance(vectors[i], centroids[chosen]));
            }
        }
        return centroids;
    }

    /** Assign each vector to its nearest centroid; reports whether anything changed. */
    private static boolean assign(float[][] vectors, float[][] centroids, int[] primary) {
        boolean moved = false;
        for (int ord = 0; ord < vectors.length; ord++) {
            int nearest = nearest(vectors[ord], centroids, -1);
            if (primary[ord] != nearest) {
                primary[ord] = nearest;
                moved = true;
            }
        }
        return moved;
    }

    /** Move each centroid to the mean of its members; a centroid that lost all of them is re-seeded on a vector. */
    private static void recentre(float[][] vectors, float[][] centroids, int[] primary, Random random) {
        int dimension = centroids[0].length;
        float[][] sums = new float[centroids.length][dimension];
        int[] counts = new int[centroids.length];

        for (int ord = 0; ord < vectors.length; ord++) {
            int centroid = primary[ord];
            counts[centroid]++;
            for (int i = 0; i < dimension; i++) {
                sums[centroid][i] += vectors[ord][i];
            }
        }

        for (int c = 0; c < centroids.length; c++) {
            if (counts[c] == 0) {
                // An empty cluster would be a posting with no vectors and a centroid that attracts none. Re-seeding is
                // cheaper than carrying it, and keeps centroidCount meaning what it says.
                centroids[c] = vectors[random.nextInt(vectors.length)].clone();
                continue;
            }
            for (int i = 0; i < dimension; i++) {
                centroids[c][i] = sums[c][i] / counts[c];
            }
        }
    }

    /**
     * A second cluster per vector, or {@link ClusteringResult#NO_SOAR}.
     *
     * <p>Only worth doing when there is somewhere else to put it, so a field with one cluster spills nothing. The
     * candidates are the nearest centroids other than the primary, and the objective penalises a residual aligned with
     * the primary's.
     */
    private static int[] spill(float[][] vectors, float[][] centroids, int[] primary, float lambda, int k) {
        int[] soar = new int[vectors.length];
        if (k < 2) {
            java.util.Arrays.fill(soar, ClusteringResult.NO_SOAR);
            return soar;
        }

        int dimension = centroids[0].length;
        float[] primaryResidual = new float[dimension];
        float[] candidateResidual = new float[dimension];

        for (int ord = 0; ord < vectors.length; ord++) {
            float[] vector = vectors[ord];
            int owner = primary[ord];
            residual(vector, centroids[owner], primaryResidual);
            float primaryNormSq = VectorUtil.dotProduct(primaryResidual, primaryResidual);
            if (primaryNormSq <= 0f) {
                // The vector sits on its centroid, so it quantizes exactly and a second copy adds nothing.
                soar[ord] = ClusteringResult.NO_SOAR;
                continue;
            }

            int best = ClusteringResult.NO_SOAR;
            float bestScore = Float.MAX_VALUE;
            int considered = 0;
            for (int c = 0; c < centroids.length && considered < SOAR_CANDIDATES; c++) {
                if (c == owner) {
                    continue;
                }
                considered++;
                residual(vector, centroids[c], candidateResidual);
                float alignment = VectorUtil.dotProduct(primaryResidual, candidateResidual);
                float score = VectorUtil.dotProduct(candidateResidual, candidateResidual)
                    + lambda * alignment * alignment / primaryNormSq;
                if (score < bestScore) {
                    bestScore = score;
                    best = c;
                }
            }
            soar[ord] = best;
        }
        return soar;
    }

    private static void residual(float[] vector, float[] centroid, float[] destination) {
        for (int i = 0; i < vector.length; i++) {
            destination[i] = vector[i] - centroid[i];
        }
    }

    /** Nearest centroid to {@code vector}, skipping {@code exclude}. */
    private static int nearest(float[] vector, float[][] centroids, int exclude) {
        int best = 0;
        float bestDistance = Float.MAX_VALUE;
        for (int c = 0; c < centroids.length; c++) {
            if (c == exclude) {
                continue;
            }
            float distance = VectorUtil.squareDistance(vector, centroids[c]);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = c;
            }
        }
        return best;
    }

    /**
     * Every vector on the heap, because Lloyd's makes a pass per iteration and re-reading them from a merge's temporary
     * file each time would dominate the cost. A clustering that samples instead would not need this, which is another
     * reason it sits behind the interface.
     */
    private static float[][] materialise(VectorSource vectors, int count, int dimension) throws IOException {
        float[][] all = new float[count][];
        for (int ord = 0; ord < count; ord++) {
            all[ord] = vectors.vector(ord).clone();
        }
        return all;
    }
}
