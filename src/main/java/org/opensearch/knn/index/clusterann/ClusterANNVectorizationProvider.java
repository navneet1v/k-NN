/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import java.util.logging.Logger;

/**
 * Discovers and loads the best available {@link BulkVectorOps} implementation.
 *
 * <p>On Java 21+ with the {@code jdk.incubator.vector} module available, loads
 * {@code PanamaBulkVectorOps} which uses the Panama Vector API for SIMD-accelerated
 * 4-at-a-time distance computation. Falls back to {@link DefaultBulkVectorOps} (scalar
 * with {@code Math.fma}) otherwise.
 *
 * <p>This mirrors the provider pattern used by Lucene's {@code VectorizationProvider}
 * and Elasticsearch's {@code ESVectorizationProvider}.
 */
final class ClusterANNVectorizationProvider {

    private static final Logger LOG = Logger.getLogger(ClusterANNVectorizationProvider.class.getName());
    private static final BulkVectorOps INSTANCE = loadBest();

    private ClusterANNVectorizationProvider() {}

    static BulkVectorOps getInstance() {
        return INSTANCE;
    }

    private static BulkVectorOps loadBest() {
        int version = Runtime.version().feature();
        if (version >= 21) {
            try {
                // Check if the vector module is available
                Class.forName("jdk.incubator.vector.FloatVector");
                var cls = Class.forName("org.opensearch.knn.index.clusterann.PanamaBulkVectorOps");
                BulkVectorOps impl = (BulkVectorOps) cls.getDeclaredConstructor().newInstance();
                LOG.info("ClusterANN using Panama SIMD bulk vector ops (Java " + version + ")");
                return impl;
            } catch (Exception e) {
                LOG.fine("Panama Vector API not available, using scalar fallback: " + e.getMessage());
            }
        }
        LOG.info("ClusterANN using scalar bulk vector ops (Java " + version + ")");
        return new DefaultBulkVectorOps();
    }
}
