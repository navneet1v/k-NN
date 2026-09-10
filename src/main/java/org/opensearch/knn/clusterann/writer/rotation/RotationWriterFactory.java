package org.opensearch.knn.clusterann.writer.rotation;

import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;

/**
 * Picks the rotation a field is written under.
 *
 * <p>The twin of {@code RotationFactory}, and the only place the write side names a rotation family — so adding one
 * is a case here and a case there, with the id in {@code .clam} joining them.
 */
public final class RotationWriterFactory {

    /**
     * Block width for the random Gaussian rotation. Ninety-six keeps a block's matrix at 36 kilobytes and its
     * application at {@code 96²} multiply-adds, while still mixing enough dimensions per block to be worth doing.
     */
    public static final int DEFAULT_BLOCK_DIMENSION = 96;

    private RotationWriterFactory() {}

    /**
     * @param rotationId which family to use, as it will be recorded in {@code .clam}
     * @param seed makes the generated rotation reproducible
     */
    public static RotationWriter create(int rotationId, int dimension, long seed) {
        return switch (rotationId) {
            case ClusterANNFieldMeta.ROTATION_NONE -> new IdentityRotationWriter(dimension);
            case ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN -> new RandomGaussianRotationWriter(
                dimension,
                Math.min(DEFAULT_BLOCK_DIMENSION, dimension),
                seed
            );
            default -> throw new IllegalArgumentException("Unknown rotationId: " + rotationId);
        };
    }
}
