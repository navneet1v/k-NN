/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * The scratch a data-blind int8 scan reuses across every cluster it probes: one {@link DataBlindInt8BlockReader},
 * whose buffers are sized by the field's geometry rather than by any one cluster.
 *
 * <p>{@link DataBlindInt8Cluster#scorer} resets that reader onto its own blocks as it starts scoring, so the scan
 * pays for a block's buffers once instead of once per probe.
 *
 * <p>Public so {@link org.opensearch.knn.clusterann.read.ClusterFactory} can make one from a field's
 * metadata without a cluster in hand; {@link #reader()} stays package-private, so only this family can reach inside.
 */
@NotThreadSafe
public final class DataBlindInt8ScanBuffers implements ScanBuffers {

    private final DataBlindInt8BlockReader reader;
    private final int[] clusterOrdinals;

    /**
     * @param maxClusterSize entries in the field's largest cluster, so the ordinals buffer is sized once here and
     *     never grows mid-scan — a scan that started from an empty buffer reallocated on its first cluster and
     *     again on every bigger one it met
     */
    public DataBlindInt8ScanBuffers(final int blockSize, final int dimension, final int maxClusterSize) {
        this.reader = new DataBlindInt8BlockReader(blockSize, dimension);
        this.clusterOrdinals = new int[maxClusterSize];
    }

    DataBlindInt8BlockReader reader() {
        return reader;
    }

    /**
     * The scan's buffer for one cluster's ordinals, SOAR entries included, returned without preserving what it
     * held — the caller refills it from the posting. It is sized for the field's largest cluster, so it is longer
     * than {@code size} for every smaller one: its length is not the cluster's size, and a caller that needs the
     * count must be told it.
     *
     * @throws IllegalArgumentException if {@code size} exceeds what the field's metadata promised, rather than
     *     letting the caller's read run off the end of the buffer
     */
    int[] clusterOrdinals(final int size) {
        if (size > clusterOrdinals.length) {
            throw new IllegalArgumentException(
                "cluster needs " + size + " ordinals but the field's largest holds " + clusterOrdinals.length
            );
        }
        return clusterOrdinals;
    }

    /** Bytes the scan holds here. Counted on the buffers, not per cluster, since every cluster shares them. */
    long ramBytesUsed() {
        return reader.ramBytesUsed() + RamUsageEstimator.sizeOf(clusterOrdinals);
    }
}
