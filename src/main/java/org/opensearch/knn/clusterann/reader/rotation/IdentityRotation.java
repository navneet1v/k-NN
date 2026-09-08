/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.rotation;

import org.apache.lucene.util.RamUsageEstimator;

/**
 * The rotation of a field that was stored unrotated: a copy, and no {@code .clar} at all.
 *
 * <p>Exists so that a caller need not ask whether the field is rotated before rotating — the branch would be in every
 * search path otherwise, and one copy of the query costs nothing next to the centroid sweep that follows it.
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
        System.arraycopy(src, 0, dest, 0, dimension);
    }

    @Override
    public long ramBytesUsed() {
        return BASE_RAM_USAGE;
    }
}
