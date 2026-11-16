/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import javax.annotation.Nullable;
import java.io.IOException;

/**
 * The {@code .clar} layout of a {@link HadamardRotation}: a single {@code long} seed.
 *
 * <p>A Hadamard rotation holds no matrix — its signs and permutation are regenerated from the seed — so the
 * region is eight bytes. {@link #create} seeds from {@link HadamardRotation#DEFAULT_SEED}; {@link #write}
 * records the rotation's seed; {@link #read} rebuilds the identical rotation from (seed, dimension). Writing
 * the seed rather than defaulting it keeps a field reproducible even if the default changes in a later format.
 *
 * <pre>
 * seed   long
 * </pre>
 */
final class HadamardRotationFormat implements RotationFormat<HadamardRotation> {

    static final HadamardRotationFormat INSTANCE = new HadamardRotationFormat();

    private HadamardRotationFormat() {}

    @Override
    public HadamardRotation create(final int dimension) {
        return HadamardRotation.create(dimension, HadamardRotation.DEFAULT_SEED);
    }

    @Override
    public long write(final IndexOutput out, final HadamardRotation rotation) throws IOException {
        out.writeLong(rotation.seed());
        return Long.BYTES;
    }

    @Override
    public HadamardRotation read(@Nullable final IndexInput region, final int dimension) throws IOException {
        if (region == null) {
            // write() always lays down an 8-byte seed, so a HADAMARD field always has a .clar region. A null
            // region here is metadata inconsistency, not an empty region — and defaulting the seed would rebuild a
            // different rotation than the one used at write time, scoring against a space the vectors were never
            // stored in with nothing to signal it. Refuse rather than default, as RotationFormat documents.
            throw new IllegalStateException(
                "HADAMARD field has no .clar region; its seed cannot be recovered (dimension=" + dimension + ")"
            );
        }
        final long seed = region.readLong();
        return HadamardRotation.create(dimension, seed);
    }
}
