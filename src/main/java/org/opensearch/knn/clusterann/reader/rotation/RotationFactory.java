/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.rotation;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;

/**
 * Builds the {@link Rotation} a field's {@code rotationId} names.
 *
 * <p>This is the one place that knows which rotation families exist, so adding Hadamard is an arm here and a class
 * beside this one. Everything else in the read path holds a {@link Rotation} and never learns which kind it is — which
 * matters because the families do not agree on what {@code .clar} even contains.
 *
 * <p>Reads nothing: what a family needs from {@code .clar} it reads on first use.
 */
public final class RotationFactory {

    private RotationFactory() {}

    /**
     * The rotation for one field.
     *
     * @param fieldMeta the field's entry, which carries the id and the dimension
     * @param clar this field's region of {@code .clar}, already cut at {@code clarOffset}, or {@code null} for a field
     *     that carries no rotation
     * @return the rotation to apply to a query before it is scored against this field's stored vectors
     * @throws IllegalArgumentException if the entry names a rotation this reader cannot apply
     */
    public static Rotation create(ClusterANNFieldMeta fieldMeta, IndexInput clar) {
        int rotationId = fieldMeta.rotationId();
        return switch (rotationId) {
            case ClusterANNFieldMeta.ROTATION_NONE -> new IdentityRotation(fieldMeta.dimension());
            case ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN -> new RandomGaussianRotation(clar, fieldMeta.dimension());
            default -> throw new IllegalArgumentException("Unsupported rotationId: " + rotationId);
        };
    }
}
