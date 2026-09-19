/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;

import java.io.IOException;
import java.util.Objects;

/**
 * A {@link FloatVectorValues} over a subset of a source's vectors, selected and reordered by an array of
 * source ordinals: position {@code i} is the source vector at {@code ordinals[i]}. Its {@link #size()} is the
 * number of ordinals and its {@link #dimension()} is the source's.
 *
 * <p>Each vector is copied into a single per-view scratch buffer, which is what {@link #vectorValue(int)}
 * returns; successive calls overwrite it, and the source's backing array is never handed out or mutated.
 *
 * <p>The {@code ordinals} array is held by reference, not copied, so reordering it afterwards changes what
 * this view returns.
 */
final class PreparedFloatVectorValues extends FloatVectorValues {

    private final FloatVectorValues source;
    private final int[] ordinals;
    private final float[] scratch;

    PreparedFloatVectorValues(final FloatVectorValues source, final int[] ordinals) {
        this.source = source;
        this.ordinals = ordinals;
        this.scratch = new float[source.dimension()];
    }

    @Override
    public int size() {
        return ordinals.length;
    }

    @Override
    public int dimension() {
        return source.dimension();
    }

    @Override
    public float[] vectorValue(final int position) throws IOException {
        Objects.checkIndex(position, ordinals.length);
        final float[] value = source.vectorValue(ordinals[position]);
        System.arraycopy(value, 0, scratch, 0, scratch.length);
        return scratch;
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return new PreparedFloatVectorValues(source.copy(), ordinals);
    }

    @Override
    public KnnVectorValues.DocIndexIterator iterator() {
        // TODO(clusterann): have the block writer consume this view directly (via its iterator) instead of the
        // positional count loop; then this can return a real iterator over the members rather than throwing.
        // A cluster has no document space: the members are ordinals in nearest-first (distance) order, which
        // a DocIdSetIterator's monotonically-increasing docID contract cannot represent. Callers address this
        // view positionally via vectorValue(int); there is no meaningful iterator.
        throw new UnsupportedOperationException("PreparedFloatVectorValues is addressed by position, not iterated");
    }
}
