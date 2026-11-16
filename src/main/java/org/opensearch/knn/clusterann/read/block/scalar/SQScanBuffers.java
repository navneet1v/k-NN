package org.opensearch.knn.clusterann.read.block.scalar;

import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * The scratch a scalar-quantized scan reuses across every cluster it probes: one
 * {@link ScalarQuantizedBlockReader}, whose buffers are sized by the field's geometry rather than by any one cluster.
 *
 * <p>{@link ScalarQuantizedCluster#scorer} resets that reader onto its own blocks as it starts scoring, so the scan
 * pays for a block's buffers once instead of once per probe.
 *
 * <p>Public so {@link org.opensearch.knn.clusterann.read.ClusterFactory} can make one from a field's
 * metadata without a cluster in hand; {@link #reader()} stays package-private, so only this family can reach inside.
 */
@NotThreadSafe
public final class SQScanBuffers implements ScanBuffers {

    private final ScalarQuantizedBlockReader reader;

    public SQScanBuffers(final int blockSize, final int dimension, final ScalarEncoding encoding) {
        this.reader = new ScalarQuantizedBlockReader(blockSize, dimension, encoding);
    }

    ScalarQuantizedBlockReader reader() {
        return reader;
    }
}
