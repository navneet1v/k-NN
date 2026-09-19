/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

/**
 * The {@code .clar} layout of an {@link IdentityRotation}, which is no layout at all.
 *
 * <p>A field with no rotation has no region: {@link #write} lays down nothing and returns zero, and {@link #read}
 * needs no bytes. That is the same statement {@code clarOffset = NO_ROTATION} makes in {@code .clam}.
 */
final class IdentityRotationFormat implements RotationFormat<IdentityRotation> {

    /** The one instance; it holds nothing that varies per field beyond the dimension it is asked for. */
    static final IdentityRotationFormat INSTANCE = new IdentityRotationFormat();

    private IdentityRotationFormat() {}

    @Override
    public IdentityRotation create(int dimension) {
        return new IdentityRotation(dimension);
    }
}
