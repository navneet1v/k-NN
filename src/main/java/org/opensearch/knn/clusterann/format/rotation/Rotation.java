/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.util.Accountable;

import java.io.IOException;

/**
 * A field's rotation: how to turn one vector into another, and nothing else.
 *
 * <p>Applied rather than exposed as a matrix, because the implementations differ in what they hold more than in what they
 * compute — a block-diagonal rotation keeps a permutation and diagonal blocks, a Hadamard rotation keeps almost nothing
 * and computes the transform instead. An interface handing out a matrix would force both to materialise the {@code d²}
 * floats neither of them has.
 *
 * <p>Serialization is deliberately absent. Which bytes an implementation lays down, and how they are read back, belongs to its
 * {@link RotationFormat}: a rotation that also wrote itself would carry two unrelated jobs, and the layout would be
 * stated somewhere other than the class that parses it.
 *
 * <p>Read lazily: obtaining one does no IO, and whatever an implementation needs from {@code .clar} is read on the first
 * {@link #rotate}, so an unqueried field pays nothing. Thread-safe — one per field, shared by every search of the
 * segment. {@link Accountable} because heap cost is a property of the implementation: eight blocks of 96 at {@code d = 768} is
 * 288 kilobytes, a Hadamard rotation a handful of bytes.
 */
public interface Rotation extends Accountable {

    /** The dimension this rotation operates on; both arguments to {@link #rotate} must be this long. */
    int dimension();

    /**
     * Rotate {@code src} into {@code dest}. {@code src} is left unmodified — the caller keeps the unrotated query,
     * since the plan is made in unrotated space while the scan runs in rotated space. The arrays must be distinct.
     *
     * @param src the vector to rotate, of length {@link #dimension()}
     * @param dest where to write the result, of length {@link #dimension()}
     * @throws IOException if the rotation's data cannot be read
     */
    void rotate(float[] src, float[] dest) throws IOException;
}
