package org.opensearch.knn.clusterann.writer;

/**
 * Every decision the writer makes about how a field is built and stored.
 *
 * <p>The read side has no equivalent, because it makes no decisions — it is told. That asymmetry is the point:
 * {@code .clam} is this record, minus the parts the reader can live without, serialized. Whenever the reader would
 * otherwise have to guess at something, it belongs here and in {@code .clam}.
 *
 * <p>Split by whether the reader needs it:
 *
 * <ul>
 *   <li><b>Recorded</b> — {@code quantizerId}, {@code docBits}, {@code rotationId}, {@code blockSize}. Without any
 *       of these the stored bytes cannot be interpreted at all.
 *   <li><b>Not recorded</b> — {@code targetClusterSize}, {@code soarLambda}, {@code seed}. These shaped the
 *       clustering, and {@code centroidCount} is the observable result. They are carried through
 *       {@code quantizerParams} anyway, because a segment whose build parameters are unknown cannot be explained
 *       after the fact.
 * </ul>
 *
 * @param quantizerId which quantization family encoded the vectors
 * @param docBits stored code width per dimension
 * @param rotationId which rotation the vectors were stored under
 * @param blockSize vectors per block, the last block of a posting excepted
 * @param targetClusterSize the average posting size aimed for, which sets the centroid count
 * @param soarLambda how strongly SOAR penalises a secondary centroid aligned with the primary residual
 * @param seed makes the clustering and the rotation reproducible
 * @param monotonicBlockShift block shift for the {@code ord → doc} monotonic encoding; larger means fewer, coarser
 *     blocks and less heap when the segment opens
 */
public record ClusterANNWriteParams(
    int quantizerId,
    int docBits,
    int rotationId,
    int blockSize,
    int targetClusterSize,
    float soarLambda,
    long seed,
    int monotonicBlockShift
) {

    /** Lucene's own choice for monotonic doc-id blocks, and the size its docvalues and vector formats use. */
    public static final int DEFAULT_MONOTONIC_BLOCK_SHIFT = 16;

    public ClusterANNWriteParams {
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be positive, got: " + blockSize);
        }
        if (targetClusterSize <= 0) {
            throw new IllegalArgumentException("targetClusterSize must be positive, got: " + targetClusterSize);
        }
        if (docBits <= 0) {
            throw new IllegalArgumentException("docBits must be positive, got: " + docBits);
        }
    }

    /**
     * How many centroids a field of {@code vectorCount} vectors gets: enough for postings of about
     * {@code targetClusterSize}, and never more than there are vectors, since a cluster needs at least one.
     */
    public int centroidCount(int vectorCount) {
        if (vectorCount == 0) {
            return 0;
        }
        int estimate = Math.max(1, (vectorCount + targetClusterSize - 1) / targetClusterSize);
        return Math.min(estimate, vectorCount);
    }
}
