package org.opensearch.knn.clusterann.read.orchestration;

public interface ScanContext {

    /**
     * Whether this context depends only on the query, so every cluster of the scan can be scored with it.
     *
     * <p>A centroid-relative scheme centres the query on the centroid of the cluster that prepared it, so its
     * context is valid for that cluster alone — the default, {@code false}. An absolute, centroid-free scheme
     * quantizes the query identically whatever cluster it lands in, so one context serves the whole scan and the
     * scan prepares it once instead of once per probe.
     *
     * <p>This asks the context rather than the cluster that produced it, so the answer cannot disagree with the
     * value it describes.
     */
    default boolean isQueryScoped() {
        return false;
    }

    interface Float extends ScanContext {
        float[] query();
    }
}
