package org.opensearch.knn.clusterann.writer.rotation;

import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

/**
 * The write twin of {@code Rotation}: chooses a rotation, applies it to everything stored, and persists whatever the
 * read side needs to reproduce it.
 *
 * <p>Two responsibilities that have to stay together. The rotation applied to the vectors and centroids at index time
 * and the rotation written to {@code .clar} must be the same one, and nothing at read time can detect a mismatch —
 * scores would simply be computed in the wrong space and come out plausible. So one object does both.
 *
 * <p>{@link #rotate} is called far more than once per vector: a vector belongs to one posting and possibly a SOAR
 * second, and its centroid is rotated too. Implementations should therefore hold their rotation, not re-derive it.
 */
public interface RotationWriter {

    /** The id recorded in {@code .clam}, which is how the reader picks the family that can undo this. */
    int rotationId();

    /** Dimension this rotation operates on; both arguments to {@link #rotate} must be this long. */
    int dimension();

    /**
     * Rotate {@code src} into {@code dest}. {@code src} is left alone, and the two arrays must be distinct.
     */
    void rotate(float[] src, float[] dest) throws IOException;

    /**
     * Write whatever {@code .clar} has to carry, at the output's current position.
     *
     * @return bytes written; zero for a rotation that stores nothing, in which case the field records no
     *     {@code .clar} region at all
     */
    long write(IndexOutput clar) throws IOException;
}
