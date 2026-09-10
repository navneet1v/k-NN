package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;

/**
 * Turns per-vector assignments into per-cluster postings, ordered the way the read side needs them.
 *
 * <p>This is neither clustering nor writing, and it is worth its own name because it is where the read side's
 * pruning becomes sound. Clustering says which cluster a vector belongs to; the byte layout says where the columns
 * go; this decides the <b>order within a posting</b>, and that order is the one thing a reader trusts without
 * being able to verify it.
 *
 * <p>Distances are always {@code ‖c−v‖}, whatever the field's metric. Even for inner product, where similarity
 * is not a distance, the posting is ordered by geometric distance from the centroid — that is the quantity the
 * stored column holds and the quantity every bound is expressed in.
 */
public final class PostingArranger {

    private PostingArranger() {}

    /**
     * Group {@code clustering}'s assignments into one {@link Posting} per centroid, each sorted by {@code ‖c−v‖}.
     *
     * <p>Two passes rather than growable lists: the first counts each cluster's members so every array can be
     * allocated at its exact size, the second fills them. A vector contributes to two clusters when SOAR gave it a
     * secondary, and is counted in both.
     *
     * @param centroids the centroids distances are measured to — the rotated ones for a rotated field, since that is
     *     the space the stored codes live in
     */
    public static Posting[] arrange(ClusteringResult clustering, VectorSource vectors, float[][] centroids,
        VectorSimilarityFunction similarity) throws IOException {

        int numCentroids = centroids.length;
        int[] counts = new int[numCentroids];
        for (int ord = 0; ord < clustering.numVectors(); ord++) {
            counts[clustering.primary()[ord]]++;
            if (clustering.soar()[ord] != ClusteringResult.NO_SOAR) {
                counts[clustering.soar()[ord]]++;
            }
        }

        int[][] ordinals = new int[numCentroids][];
        float[][] distances = new float[numCentroids][];
        FixedBitSet[] soarBits = new FixedBitSet[numCentroids];
        int[] filled = new int[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            ordinals[c] = new int[counts[c]];
            distances[c] = new float[counts[c]];
            // FixedBitSet rejects a zero length, and an empty posting has no entry to mark anyway.
            soarBits[c] = new FixedBitSet(Math.max(counts[c], 1));
        }

        for (int ord = 0; ord < clustering.numVectors(); ord++) {
            float[] vector = vectors.vector(ord);
            int primary = clustering.primary()[ord];
            add(ordinals[primary], distances[primary], filled, primary, ord, distance(vector, centroids[primary]));

            int soar = clustering.soar()[ord];
            if (soar != ClusteringResult.NO_SOAR) {
                soarBits[soar].set(filled[soar]);
                add(ordinals[soar], distances[soar], filled, soar, ord, distance(vector, centroids[soar]));
            }
        }

        Posting[] postings = new Posting[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            // Sorting moves the ordinals and the distances together, so the SOAR bits — which were set against the
            // pre-sort positions — have to be permuted with them rather than left behind.
            sortByDistance(distances[c], ordinals[c], soarBits[c]);
            postings[c] = new Posting(c, ordinals[c], distances[c], soarBits[c]);
        }
        return postings;
    }

    private static void add(int[] ordinals, float[] distances, int[] filled, int centroid, int ord, float distance) {
        ordinals[filled[centroid]] = ord;
        distances[filled[centroid]] = distance;
        filled[centroid]++;
    }

    /** {@code ‖c−v‖}. The square root is kept: the stored column is a distance, not a squared one. */
    private static float distance(float[] vector, float[] centroid) {
        return (float) Math.sqrt(VectorUtil.squareDistance(vector, centroid));
    }

    /**
     * Insertion sort over the three parallel columns.
     *
     * <p>Insertion sort because postings are small — a few hundred entries at the target cluster size — and because
     * it keeps the three columns in step without allocating an index permutation to apply afterwards.
     */
    private static void sortByDistance(float[] distances, int[] ordinals, FixedBitSet soar) {
        for (int i = 1; i < distances.length; i++) {
            float distance = distances[i];
            int ordinal = ordinals[i];
            boolean isSoar = soar.get(i);
            int j = i - 1;
            while (j >= 0 && distances[j] > distance) {
                distances[j + 1] = distances[j];
                ordinals[j + 1] = ordinals[j];
                setTo(soar, j + 1, soar.get(j));
                j--;
            }
            distances[j + 1] = distance;
            ordinals[j + 1] = ordinal;
            setTo(soar, j + 1, isSoar);
        }
    }

    private static void setTo(FixedBitSet bits, int index, boolean value) {
        if (value) {
            bits.set(index);
        } else {
            bits.clear(index);
        }
    }
}
