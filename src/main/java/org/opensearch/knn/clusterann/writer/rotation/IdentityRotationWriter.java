package org.opensearch.knn.clusterann.writer.rotation;

import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;

/**
 * No rotation: vectors are stored in the space they arrived in.
 *
 * <p>Exists so the writer never branches on whether a field is rotated. A copy is cheaper than the alternative — a
 * nullable rotation, tested at every call site — and this costs nothing that a rotated field would not pay anyway.
 */
public final class IdentityRotationWriter implements RotationWriter {

    private final int dimension;

    public IdentityRotationWriter(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public int rotationId() {
        return ClusterANNFieldMeta.ROTATION_NONE;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public void rotate(float[] src, float[] dest) {
        System.arraycopy(src, 0, dest, 0, dimension);
    }

    /** Nothing to persist, so the field carries no {@code .clar} region. */
    @Override
    public long write(IndexOutput clar) {
        return 0L;
    }
}
