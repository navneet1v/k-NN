/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import org.opensearch.common.Nullable;
import java.io.IOException;

/**
 * One rotation implementation's {@code .clar} layout: how to generate, write, and read a rotation back.
 *
 * <p>Both directions sit in one class per implementation, because a layout stated in a writer and again in a reader can
 * drift — silently, since a rotation read wrongly is still a rotation. The {@link Rotation} it produces knows only how
 * to rotate; where its bytes go is this class's business.
 *
 * <p>An implementation does not know its {@code rotationId}; that mapping lives in {@link RotationFormats}, so the id
 * is stated in one place. Adding one is a class here and an arm in that registry.
 */
public interface RotationFormat<T extends Rotation> {

    /**
     * Generate a rotation for a field of this dimension. Called at index time, then applied to every vector and laid
     * down with {@link #write}.
     *
     * @param dimension the field's vector dimension
     * @throws IllegalArgumentException if this implementation cannot rotate that dimension
     */
    T create(int dimension);

    /**
     * Append {@code rotation} to {@code out} in this implementation's layout.
     *
     * @param out the {@code .clar} output, positioned where this field's region begins
     * @param rotation a rotation of this implementation; one from elsewhere is refused rather than misread
     * @return bytes written (the field's {@code clarLength}; zero for an implementation that stores nothing). The
     *     offset is the caller's to record.
     * @throws IllegalArgumentException if {@code rotation} belongs to another implementation
     * @throws IOException if the bytes cannot be written
     */
    default long write(IndexOutput out, T rotation) throws IOException {
        return 0L;
    };

    /**
     * The rotation a field's region of {@code .clar} describes. Reads nothing — an implementation that needs bytes
     * takes them on the first {@link Rotation#rotate}, so an unqueried field never pays for them.
     *
     * @param region this field's region of {@code .clar}, cut at {@code clarOffset}, or {@code null} for an
     *     implementation that stores nothing
     * @param dimension the field's vector dimension, from {@code .clam}
     */
    default T read(@Nullable IndexInput region, int dimension) throws IOException {
        return create(dimension);
    };
}
