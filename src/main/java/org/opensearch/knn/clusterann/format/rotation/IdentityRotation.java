/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.util.RamUsageEstimator;

/**
 * The rotation of a field stored unrotated: a copy.
 *
 * <p>Exists so a caller need not check whether the field is rotated before rotating — one copy of the query costs
 * nothing next to the centroid sweep that follows.
 */
public final class IdentityRotation implements Rotation {

    private static final long BASE_RAM_USAGE = RamUsageEstimator.shallowSizeOfInstance(IdentityRotation.class);

    private final int dimension;

    public IdentityRotation(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public void rotate(float[] src, float[] dest) {
        if (src.length != dimension || dest.length != dimension) {
            throw new IllegalArgumentException(
                "This rotation is " + dimension + "-dimensional, got src=" + src.length + " dest=" + dest.length
            );
        }
        System.arraycopy(src, 0, dest, 0, dimension);
    }

    @Override
    public long ramBytesUsed() {
        return BASE_RAM_USAGE;
    }
}
