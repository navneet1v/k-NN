/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ChecksumIndexInput;

import java.io.IOException;

/**
 * One field's entry in the {@code .clam} metadata file: the field's shape, how its vectors are encoded, and
 * the offsets that locate its data in the other files. Read once when the segment opens and then consulted
 * per query, so nothing here is a cursor and nothing here reads.
 *
 * <p>The entry is laid out in exactly the order it is read below. The preceding field number is consumed by
 * the caller, which needs it to resolve the {@code FieldInfo} before it can make sense of the rest.
 *
 * <pre>
 * dimension                    vInt
 * vectorCount                  vInt
 * centroidCount                vInt
 * similarityFunction           byte                     // L2 = 0, IP = 1, cosine = 2
 * docBits                      byte
 * rotationId                   byte                     // none = 0, random Gaussian = 1
 * quantizerId                  byte                     // SQ = 0, PQ = 1, RQ = 2, ...
 * quantizerParamsLength        int
 * quantizerParams              quantizerParamsLength bytes
 *
 * clacOffset                   long                     // .clac: cluster metadata
 * clacLength                   long
 * clacCentroidsOffset          long                     // .clac: centroids
 * clacRotatedCentroidsOffset   long                     // .clac: centroids in rotated space; rotated fields only
 *
 * clapOffset                   long                     // .clap: postings
 * clapLength                   long
 * clapCentroidOffsets          centroidCount longs      // where each centroid's posting starts
 * centroidLengths              centroidCount ints       // posting byte lengths, for prefetch sizing
 * clusterSizes                 centroidCount ints       // vectors per cluster, primary + SOAR
 *
 * clarOffset                   long                     // .clar: rotation matrix; rotated fields only
 * clarLength                   long                     // rotated fields only
 * </pre>
 *
 */
public record ClusterANNFieldMeta(int blockSize, int dimension, int vectorCount, int centroidCount,
    VectorSimilarityFunction similarityFunction, int docBits, int rotationId, int quantizerId, byte[] quantizerParams, long clacOffset,
    long clacLength, long clacCentroidsOffset, long clacRotatedCentroidsOffset, long clapOffset, long clapLength,
    long[] clapCentroidOffsets, int[] centroidLengths, int[] clusterSizes, long clarOffset, long clarLength) {

    /** {@link #rotationId()} of a field whose vectors were stored unrotated. */
    public static final int ROTATION_NONE = 0;

    /** {@link #rotationId()} of a field rotated by a dense random Gaussian matrix, held in {@code .clar}. */
    public static final int ROTATION_RANDOM_GAUSSIAN = 1;

    /**
     * {@link #clarOffset()}, {@link #clarLength()} and {@link #clacRotatedCentroidsOffset()} of a field that
     * carries no rotation.
     */
    public static final long NO_ROTATION = -1L;

    /** Bytes each cluster contributes to the arrays sized by {@code centroidCount}: one offset and two counts. */
    private static final int BYTES_PER_CENTROID = Long.BYTES + 2 * Integer.BYTES;

    private static final byte SIMILARITY_L2 = 0;
    private static final byte SIMILARITY_IP = 1;
    private static final byte SIMILARITY_COSINE = 2;

    public ClusterANNFieldMeta {
        requireLength(clapCentroidOffsets.length, centroidCount, "clapCentroidOffsets");
        requireLength(centroidLengths.length, centroidCount, "centroidLengths");
        requireLength(clusterSizes.length, centroidCount, "clusterSizes");

        // The id decides whether there is anything rotated to locate, so every rotated value must agree with it.
        requireRotationAgrees(rotationId, clarOffset, "clarOffset");
        requireRotationAgrees(rotationId, clarLength, "clarLength");
        requireRotationAgrees(rotationId, clacRotatedCentroidsOffset, "clacRotatedCentroidsOffset");
    }

    /**
     * Whether this field stores a rotation, and so has both a {@code .clar} offset for the matrix and a
     * {@code .clac} offset for the centroids in the rotated space.
     */
    public boolean hasRotation() {
        return rotationId != ROTATION_NONE;
    }

    /**
     * Whether the field contains any vectors.
     * @return boolean if true there won't be any corresponding offsets to other files
     */
    public boolean isEmpty() {
        return vectorCount == 0;
    }

    /**
     * Reads one field's entry from {@code meta}, positioned just past the field number.
     *
     * @param meta the metadata input, advanced past this entry on return
     * @param blockSize the block size from the file header, shared by every field
     * @throws CorruptIndexException if the entry describes a field that cannot exist
     */
    public static ClusterANNFieldMeta read(ChecksumIndexInput meta, int blockSize) throws IOException {
        int dimension = meta.readVInt();
        int vectorCount = meta.readVInt();
        int centroidCount = meta.readVInt();
        VectorSimilarityFunction similarityFunction = similarityFunction(meta.readByte(), meta);
        int docBits = meta.readByte();
        int rotationId = rotationId(meta.readByte(), meta);
        int quantizerId = meta.readByte();
        byte[] quantizerParams = readQuantizerParams(meta);

        // The counts have to make sense before anything derived from them is read.
        checkShape(meta, dimension, vectorCount, centroidCount);

        // Centroid geometry. Rotated centroids exist only for a rotated field; otherwise they would be the
        // centroids already located by clacCentroidsOffset, so the entry does not carry the offset at all.
        long clacOffset = meta.readLong();
        long clacLength = meta.readLong();
        long clacCentroidsOffset = meta.readLong();
        long clacRotatedCentroidsOffset = NO_ROTATION;
        if (rotationId != ROTATION_NONE) {
            clacRotatedCentroidsOffset = meta.readLong();
            checkNonNegative(meta, clacRotatedCentroidsOffset, "clacRotatedCentroidsOffset");
        }

        // Postings
        long clapOffset = meta.readLong();
        long clapLength = meta.readLong();
        checkFits(meta, (long) centroidCount * BYTES_PER_CENTROID, "centroidCount", centroidCount);
        long[] clapCentroidOffsets = readLongs(meta, centroidCount);
        int[] centroidSizeInBytes = readInts(meta, centroidCount);
        int[] centroidVectorCounts = readInts(meta, centroidCount);

        // The rotation matrix, on the same terms as the rotated centroids above.
        long clarOffset = NO_ROTATION;
        long clarLength = NO_ROTATION;
        if (rotationId != ROTATION_NONE) {
            clarOffset = meta.readLong();
            checkNonNegative(meta, clarOffset, "clarOffset");
            clarLength = meta.readLong();
            checkNonNegative(meta, clarLength, "clarLength");
        }

        checkOffsets(meta, clacOffset, clacLength, clacCentroidsOffset, clapOffset, clapLength);

        return new ClusterANNFieldMeta(
            blockSize,
            dimension,
            vectorCount,
            centroidCount,
            similarityFunction,
            docBits,
            rotationId,
            quantizerId,
            quantizerParams,
            clacOffset,
            clacLength,
            clacCentroidsOffset,
            clacRotatedCentroidsOffset,
            clapOffset,
            clapLength,
            clapCentroidOffsets,
            centroidSizeInBytes,
            centroidVectorCounts,
            clarOffset,
            clarLength
        );
    }

    /**
     * A rotation this reader knows how to apply. An unrecognised id means the segment was written by a format
     * that rotates in a way this one cannot reproduce, so scoring it would be silently wrong.
     */
    private static int rotationId(byte encoded, ChecksumIndexInput meta) throws IOException {
        return switch (encoded) {
            case ROTATION_NONE, ROTATION_RANDOM_GAUSSIAN -> encoded;
            default -> throw new CorruptIndexException("Unknown rotation: " + encoded, meta);
        };
    }

    private static VectorSimilarityFunction similarityFunction(byte encoded, ChecksumIndexInput meta) throws IOException {
        return switch (encoded) {
            case SIMILARITY_L2 -> VectorSimilarityFunction.EUCLIDEAN;
            case SIMILARITY_IP -> VectorSimilarityFunction.DOT_PRODUCT;
            case SIMILARITY_COSINE -> VectorSimilarityFunction.COSINE;
            default -> throw new CorruptIndexException("Unknown similarity function: " + encoded, meta);
        };
    }

    private static byte[] readQuantizerParams(ChecksumIndexInput meta) throws IOException {
        int length = meta.readInt();
        if (length < 0) {
            throw new CorruptIndexException("Negative quantizerParamsLength: " + length, meta);
        }
        if (length == 0) return new byte[0];

        checkFits(meta, length, "quantizerParamsLength", length);
        byte[] params = new byte[length];
        meta.readBytes(params, 0, length);
        return params;
    }

    /**
     * Rejects a count whose payload cannot fit in what is left of the file, before anything is allocated for it.
     *
     * <p>The bound is the file itself rather than a guess at a sane maximum: {@code bytes} of payload cannot
     * follow when the file does not hold that many more bytes. So a corrupt count fails here, instead of asking
     * the heap for gigabytes and dying before the footer is ever checked.
     */
    private static void checkFits(ChecksumIndexInput meta, long bytes, String name, int value) throws IOException {
        long remaining = meta.length() - meta.getFilePointer();
        if (bytes > remaining) {
            throw new CorruptIndexException(name + "=" + value + " needs " + bytes + " bytes, but only " + remaining + " remain", meta);
        }
    }

    private static long[] readLongs(ChecksumIndexInput meta, int count) throws IOException {
        long[] values = new long[count];
        meta.readLongs(values, 0, count);
        return values;
    }

    private static int[] readInts(ChecksumIndexInput meta, int count) throws IOException {
        int[] values = new int[count];
        meta.readInts(values, 0, count);
        return values;
    }

    /**
     * A cluster needs at least one vector, so {@code centroidCount} can never exceed {@code vectorCount}.
     * Checking that here also bounds the three arrays sized by it.
     */
    private static void checkShape(ChecksumIndexInput meta, int dimension, int vectorCount, int centroidCount) throws IOException {
        if (dimension <= 0) {
            throw new CorruptIndexException("Dimension must be positive, got: " + dimension, meta);
        }
        if (vectorCount < 0) {
            throw new CorruptIndexException("Negative vectorCount: " + vectorCount, meta);
        }
        if (centroidCount < 0 || centroidCount > vectorCount) {
            throw new CorruptIndexException("centroidCount must be in [0, " + vectorCount + "], got: " + centroidCount, meta);
        }
    }

    /** The offsets every field has. The rotation-dependent ones are checked where they are read. */
    private static void checkOffsets(
        ChecksumIndexInput meta,
        long clacOffset,
        long clacLength,
        long clacCentroidsOffset,
        long clapOffset,
        long clapLength
    ) throws IOException {
        checkNonNegative(meta, clacOffset, "clacOffset");
        checkNonNegative(meta, clacLength, "clacLength");
        checkNonNegative(meta, clacCentroidsOffset, "clacCentroidsOffset");
        checkNonNegative(meta, clapOffset, "clapOffset");
        checkNonNegative(meta, clapLength, "clapLength");
    }

    /**
     * An offset that only a rotated field has must be present exactly when the rotation is. Checked in the
     * record rather than only in {@link #read} so the write side cannot build an entry the read side rejects.
     */
    private static void requireRotationAgrees(int rotationId, long offset, String name) {
        if ((rotationId == ROTATION_NONE) != (offset == NO_ROTATION)) {
            throw new IllegalArgumentException("rotationId=" + rotationId + " disagrees with " + name + "=" + offset);
        }
    }

    private static void checkNonNegative(ChecksumIndexInput meta, long value, String name) throws IOException {
        if (value < 0) {
            throw new CorruptIndexException("Negative " + name + ": " + value, meta);
        }
    }

    private static void requireLength(int actual, int expected, String name) {
        if (actual != expected) {
            throw new IllegalArgumentException(name + " must hold centroidCount=" + expected + " entries, got: " + actual);
        }
    }
}
