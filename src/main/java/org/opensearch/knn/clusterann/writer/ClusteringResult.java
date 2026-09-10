package org.opensearch.knn.clusterann.writer;

/**
 * What clustering decided: where the centroids are, and which of them each vector belongs to.
 *
 * <p>A vector has one primary centroid and may have one SOAR centroid — a second cluster it is also listed in, so a
 * boundary vector can be found from either side. The two assignments are kept as parallel arrays indexed by ordinal
 * rather than as per-cluster lists, because that is the form clustering naturally produces; grouping them into
 * postings is a separate step.
 *
 * @param centroids one row per centroid, in the space the vectors were clustered in
 * @param primary {@code primary[ord]} is the centroid that owns this vector
 * @param soar {@code soar[ord]} is a second centroid this vector is also listed in, or {@link #NO_SOAR}
 */
public record ClusteringResult(float[][] centroids, int[] primary, int[] soar) {

    /** {@code soar[ord]} of a vector listed in one cluster only. */
    public static final int NO_SOAR = -1;

    public ClusteringResult {
        if (primary.length != soar.length) {
            throw new IllegalArgumentException("primary has " + primary.length + " entries, soar has " + soar.length);
        }
        for (int ord = 0; ord < primary.length; ord++) {
            requireCentroid(primary[ord], centroids.length, "primary", ord);
            if (soar[ord] != NO_SOAR) {
                requireCentroid(soar[ord], centroids.length, "soar", ord);
                if (soar[ord] == primary[ord]) {
                    // The two copies would be quantized against the same centroid, so the second would be an exact
                    // duplicate: extra bytes, extra scoring, and no reachability the first does not already give.
                    throw new IllegalArgumentException("soar[" + ord + "] repeats the primary centroid " + primary[ord]);
                }
            }
        }
    }

    public int numCentroids() {
        return centroids.length;
    }

    public int numVectors() {
        return primary.length;
    }

    private static void requireCentroid(int centroid, int numCentroids, String name, int ord) {
        if (centroid < 0 || centroid >= numCentroids) {
            throw new IllegalArgumentException(
                name + "[" + ord + "]=" + centroid + " is outside [0, " + numCentroids + ")"
            );
        }
    }
}
