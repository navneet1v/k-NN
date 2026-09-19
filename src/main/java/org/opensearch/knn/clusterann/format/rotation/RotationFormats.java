/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format.rotation;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;

import org.opensearch.common.Nullable;
import java.io.IOException;

/**
 * The rotation implementations this codec knows, by the id {@code .clam} records.
 *
 * <p>Adding an implementation is a class and an arm here; everything else holds a {@link Rotation} and never learns
 * which kind it is, which lets the implementations disagree about what {@code .clar} contains.
 *
 * <p>An unrecognised id is refused rather than defaulted — a segment rotated in a way this codec cannot reproduce
 * would otherwise be scored against the wrong space with nothing to say so.
 */
public final class RotationFormats {

    private RotationFormats() {}

    /**
     * The implementation an id names.
     *
     * <p>Wildcarded because an id is a runtime value while the rotation type is compile-time: nothing here knows which
     * type the id stands for. It serves {@link RotationFormat#read} and {@link RotationFormat#create} directly, while
     * {@link #write} has to reunite the two — see the note there.
     *
     * @throws IllegalArgumentException if no implementation claims it
     */
    public static RotationFormat<? extends Rotation> forId(int rotationId) {
        return switch (rotationId) {
            case ClusterANNFormatConstants.ROTATION_NONE -> IdentityRotationFormat.INSTANCE;
            case ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN -> BlockDiagonalRotationFormat.INSTANCE;
            default -> throw new IllegalArgumentException("Unsupported rotationId: " + rotationId);
        };
    }

    /**
     * A fresh rotation in the implementation {@code rotationId} names — the index side's entry point for a caller that
     * has the id from a field's configuration rather than a typed implementation.
     *
     * @param rotationId the implementation to generate in, as recorded in {@code .clam}
     * @param dimension the field's vector dimension
     * @throws IllegalArgumentException if no implementation claims the id, or it cannot rotate that dimension
     */
    public static Rotation create(int rotationId, int dimension) {
        return forId(rotationId).create(dimension);
    }

    /**
     * The rotation a field carries — the read side's entry point.
     *
     * @param rotationId the field's implementation, from {@code .clam}
     * @param region this field's region of {@code .clar}, cut at {@code clarOffset}, or {@code null} for an
     *     implementation that stores nothing
     * @param dimension the field's vector dimension
     */
    public static Rotation read(int rotationId, @Nullable IndexInput region, int dimension) throws IOException {
        return forId(rotationId).read(region, dimension);
    }

    /**
     * Lay {@code rotation} down in the layout {@code rotationId} names — the write side's counterpart to {@link #read},
     * for a caller that has an id rather than a typed implementation.
     *
     * <p><b>This is the one place the generic guarantee is traded away.</b> {@link RotationFormat#write} takes the
     * implementation's own type, so the compiler can refuse a mismatch; an id cannot carry that type, so pairing them
     * is an unchecked cast. A caller holding the implementation it called {@link RotationFormat#create} on should call
     * {@code format.write(out, rotation)} instead and keep the check. A mismatched pair fails with an
     * {@link IllegalArgumentException} rather than writing bytes nothing can read back.
     *
     * @param rotationId the implementation to write in, as recorded in {@code .clam}
     * @param out the {@code .clar} output, positioned where this field's region begins
     * @param rotation a rotation produced by that same implementation
     * @return bytes written (the field's {@code clarLength}; zero for an implementation that stores nothing)
     * @throws IllegalArgumentException if no implementation claims the id, or {@code rotation} was not produced by it
     */
    public static long write(int rotationId, IndexOutput out, Rotation rotation) throws IOException {
        return write(rotationId, forId(rotationId), out, rotation);
    }

    /**
     * Recovers the implementation's rotation type as {@code T} so {@link RotationFormat#write} can be called. The cast
     * is unchecked by construction — {@code forId} erased the type — and is why {@link #write} carries its warning.
     */
    @SuppressWarnings("unchecked")
    private static <T extends Rotation> long write(int rotationId, RotationFormat<T> format, IndexOutput out, Rotation rotation)
        throws IOException {
        try {
            return format.write(out, (T) rotation);
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("rotationId " + rotationId + " cannot write a " + rotation.getClass().getSimpleName(), e);
        }
    }
}
