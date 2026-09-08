/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.rotation;

import org.apache.lucene.util.Accountable;

import java.io.IOException;

/**
 * The rotation a field's vectors were stored under, as something that can be <em>applied</em> to a vector.
 *
 * <p>Applied rather than exposed, because the families differ in what they store more than in what they compute. A
 * random Gaussian rotation is a permutation and a set of diagonal blocks in {@code .clar}, costing
 * {@code O(Σ bDim²)}; a Hadamard rotation stores almost nothing and costs {@code O(d log d)}, since the transform is
 * implicit and computed rather than read. An interface handing out a matrix would force both to materialise the
 * {@code d²} floats neither of them holds, which is the whole reason not to use one.
 *
 * <p>Read lazily: constructing one does no IO, and whatever a family needs from {@code .clar} is read on the first
 * {@link #rotate}. So a field can carry a rotation that a query never pays for.
 *
 * <p>Implementations are thread-safe: one per field, shared by every search of the segment.
 *
 * <p>{@link Accountable} because what a rotation costs on the heap is a property of the family — eight blocks of 96 at
 * {@code d = 768} is 288 kilobytes, while a Hadamard rotation is a handful of bytes.
 */
public interface Rotation extends Accountable {

    /** The dimension this rotation operates on; both arguments to {@link #rotate} must be this long. */
    int dimension();

    /**
     * Rotate {@code src} into {@code dest}.
     *
     * <p>{@code src} is not modified, so a caller may keep the unrotated query — which it needs, since the plan is made
     * in the unrotated space while the scan runs in the rotated one. The two arrays must be distinct.
     *
     * @param src the vector to rotate, of length {@link #dimension()}
     * @param dest where to write the result, of length {@link #dimension()}
     * @throws IOException if the rotation's data cannot be read
     */
    void rotate(float[] src, float[] dest) throws IOException;
}
