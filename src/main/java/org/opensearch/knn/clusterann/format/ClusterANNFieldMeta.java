/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.experimental.Accessors;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

import org.opensearch.knn.clusterann.format.rotation.RotationScheme;

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
 * clacOffset                   long                     // .clac: cluster metadata (absolute)
 * clacLength                   long
 * clacCentroidsOffset          long                     // .clac: centroids (relative to clacOffset)
 * clacRotatedCentroidsOffset   long                     // .clac: centroids in rotated space, relative; rotated fields only
 *
 * clapOffset                   long                     // .clap: postings (absolute)
 * clapLength                   long
 * clapCentroidOffsets          centroidCount longs      // where each centroid's posting starts, relative to clapOffset
 * centroidLengths              centroidCount ints       // posting byte lengths, for prefetch sizing
 * clusterSizes                 centroidCount ints       // vectors per cluster, primary + SOAR
 *
 * clarOffset                   long                     // .clar: rotation matrix, absolute; rotated fields only
 * clarLength                   long                     // rotated fields only
 *
 * ordToDoc                     OrdToDocDISIReaderConfiguration   // ord&rarr;doc config; sparse data stream lives in .clap
 * </pre>
 *
 * <h2>Absolute and relative offsets</h2>
 *
 * <p>One offset per file — {@code clacOffset}, {@code clapOffset}, {@code clarOffset} — is absolute, and everything
 * that points inside one of those regions is relative to it. That is not a stylistic choice: the reader cuts the
 * field's region out of the file once and slices the inner regions out of <em>that</em>.
 *
 * <p>It also makes the entry checkable: an inner offset has to fall within its region's length
 */
@Getter
@Accessors(fluent = true)
@EqualsAndHashCode
public final class ClusterANNFieldMeta {

    /** Bytes each cluster contributes to the arrays sized by {@code centroidCount}: one offset and two counts. */
    private static final int BYTES_PER_CENTROID = Long.BYTES + 2 * Integer.BYTES;

    private static final byte SIMILARITY_L2 = 0;
    private static final byte SIMILARITY_MIP = 1;
    private static final byte SIMILARITY_COSINE = 2;

    /** Block shift for the {@code ordToDoc} monotonic addresses; matches Lucene's flat/HNSW formats. */
    private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
    /** Sentinel for a rotation offset/length that is absent on an unrotated field. */
    private static final long NO_ROTATION = -1L;

    private final int blockSize;
    private final int dimension;
    private final int vectorCount;
    private final int centroidCount;
    private final VectorSimilarityFunction similarityFunction;
    private final int docBits;
    private final int rotationId;
    private final int quantizerId;
    private final byte[] quantizerParams;
    private final long clacOffset;
    private final long clacLength;
    private final long clacCentroidsOffset;
    private final long clacRotatedCentroidsOffset;
    private final long clapOffset;
    private final long clapLength;
    private final long[] clapCentroidOffsets;
    private final int[] centroidLengths;
    private final int[] clusterSizes;
    private final long clarOffset;
    private final long clarLength;

    /** Read-derived config: {@code null} on the write side, and not part of the field's logical identity. */
    @EqualsAndHashCode.Exclude
    private final OrdToDocDISIReaderConfiguration ordToDoc;

    // One field entry is a wide, flat record of shape + offsets; the positional all-args constructor mirrors it.
    @SuppressWarnings("checkstyle:ParameterNumber")
    public ClusterANNFieldMeta(
        int blockSize,
        int dimension,
        int vectorCount,
        int centroidCount,
        VectorSimilarityFunction similarityFunction,
        int docBits,
        int rotationId,
        int quantizerId,
        byte[] quantizerParams,
        long clacOffset,
        long clacLength,
        long clacCentroidsOffset,
        long clacRotatedCentroidsOffset,
        long clapOffset,
        long clapLength,
        long[] clapCentroidOffsets,
        int[] centroidLengths,
        int[] clusterSizes,
        long clarOffset,
        long clarLength,
        OrdToDocDISIReaderConfiguration ordToDoc
    ) {
        requireLength(clapCentroidOffsets.length, centroidCount, "clapCentroidOffsets");
        requireLength(centroidLengths.length, centroidCount, "centroidLengths");
        requireLength(clusterSizes.length, centroidCount, "clusterSizes");

        // The id decides whether there is anything rotated to locate, so every rotated value must agree with it.
        requireRotationAgrees(rotationId, clarOffset, "clarOffset");
        requireRotationAgrees(rotationId, clarLength, "clarLength");
        requireRotationAgrees(rotationId, clacRotatedCentroidsOffset, "clacRotatedCentroidsOffset");

        this.blockSize = blockSize;
        this.dimension = dimension;
        this.vectorCount = vectorCount;
        this.centroidCount = centroidCount;
        this.similarityFunction = similarityFunction;
        this.docBits = docBits;
        this.rotationId = rotationId;
        this.quantizerId = quantizerId;
        this.quantizerParams = quantizerParams;
        this.clacOffset = clacOffset;
        this.clacLength = clacLength;
        this.clacCentroidsOffset = clacCentroidsOffset;
        this.clacRotatedCentroidsOffset = clacRotatedCentroidsOffset;
        this.clapOffset = clapOffset;
        this.clapLength = clapLength;
        this.clapCentroidOffsets = clapCentroidOffsets;
        this.centroidLengths = centroidLengths;
        this.clusterSizes = clusterSizes;
        this.clarOffset = clarOffset;
        this.clarLength = clarLength;
        this.ordToDoc = ordToDoc;
    }

    /**
     * Whether this field stores a rotation, and so has both a {@code .clar} offset for the matrix and a
     * {@code .clac} offset for the centroids in the rotated space.
     */
    public boolean hasRotation() {
        return RotationScheme.fromCode(rotationId).rotates();
    }

    /**
     * Whether the field contains any vectors.
     * @return boolean if true there won't be any corresponding offsets to other files
     */
    public boolean isEmpty() {
        return vectorCount == 0;
    }

    /**
     * A zeroed entry for a field with no vectors: zero counts, no rotation, and empty per-centroid arrays, with
     * the region offsets pointing at the current (empty) positions. It is written like any other entry so the
     * reader opens it uniformly; {@link #isEmpty()} then short-circuits before any region is consulted.
     *
     * @param blockSize          the shared block size
     * @param dimension          the field's vector dimension
     * @param similarityFunction the field's similarity function
     * @param docBits            the doc-side quantization bit width
     * @param quantizerId        the quantizer code (recorded for consistency; unused for an empty field)
     * @param clacOffset         the current {@code .clac} position (this field writes no centroids)
     * @param clapOffset         the current {@code .clap} position (this field writes no postings)
     */
    public static ClusterANNFieldMeta empty(
        int blockSize,
        int dimension,
        VectorSimilarityFunction similarityFunction,
        int docBits,
        int quantizerId,
        long clacOffset,
        long clapOffset
    ) {
        return new ClusterANNFieldMeta(
            blockSize,
            dimension,
            0,
            0,
            similarityFunction,
            docBits,
            RotationScheme.NONE.code(),
            quantizerId,
            new byte[0],
            clacOffset,
            0L,
            0L,
            NO_ROTATION,
            clapOffset,
            0L,
            new long[0],
            new int[0],
            new int[0],
            NO_ROTATION,
            NO_ROTATION,
            null
        );
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
        boolean rotated = RotationScheme.fromCode(rotationId).rotates();
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
        if (rotated) {
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
        if (rotated) {
            clarOffset = meta.readLong();
            checkNonNegative(meta, clarOffset, "clarOffset");
            clarLength = meta.readLong();
            checkNonNegative(meta, clarLength, "clarLength");
        }

        checkClacOffsets(meta, clacOffset, clacLength, clacCentroidsOffset, clacRotatedCentroidsOffset);
        checkClapOffsets(meta, clapOffset, clapLength, clapCentroidOffsets);

        // The ord->doc config closes the entry; its sparse data stream lives in .clap, read lazily at query time.
        OrdToDocDISIReaderConfiguration ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, vectorCount);

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
            clarLength,
            ordToDoc
        );
    }

    /**
     * Writes this entry to {@code .clam} in the exact byte layout {@link #read} consumes (documented at the
     * class level), positioned just past the field number (which the caller writes). {@code blockSize} is a
     * file-header field shared by every entry, so it is not written here.
     *
     * <p>The entry closes with the ord&rarr;doc mapping via the stock {@link OrdToDocDISIReaderConfiguration}:
     * its config is appended here to {@code meta}, and — for a sparse field — its {@code IndexedDISI} +
     * {@code DirectMonotonic} data stream is written to {@code data} (the {@code .clap} postings output the
     * config offsets point into). A dense field ({@code docsWithField.cardinality() == maxDoc}) writes only
     * the config. This is the inverse of the {@link OrdToDocDISIReaderConfiguration#fromStoredMeta} call in
     * {@link #read}, so the {@code ordToDoc} field is derived on read and is {@code null} here.
     *
     * @param meta          the metadata output ({@code .clam}) to append this entry to
     * @param data          the data output ({@code .clap}) for the sparse ord&rarr;doc stream
     * @param maxDoc        the segment's document count, used to detect the dense case
     * @param docsWithField the docs that carry this field, in ordinal order
     * @throws IOException if writing fails
     */
    public void write(IndexOutput meta, IndexOutput data, int maxDoc, DocsWithFieldSet docsWithField) throws IOException {
        meta.writeVInt(dimension);
        meta.writeVInt(vectorCount);
        meta.writeVInt(centroidCount);
        meta.writeByte(similarityCode(similarityFunction));
        meta.writeByte((byte) docBits);
        meta.writeByte((byte) rotationId);
        meta.writeByte((byte) quantizerId);
        meta.writeInt(quantizerParams.length);
        meta.writeBytes(quantizerParams, 0, quantizerParams.length);

        meta.writeLong(clacOffset);
        meta.writeLong(clacLength);
        meta.writeLong(clacCentroidsOffset);
        if (hasRotation()) {
            meta.writeLong(clacRotatedCentroidsOffset);
        }

        meta.writeLong(clapOffset);
        meta.writeLong(clapLength);
        for (long offset : clapCentroidOffsets) {
            meta.writeLong(offset);
        }
        for (int length : centroidLengths) {
            meta.writeInt(length);
        }
        for (int size : clusterSizes) {
            meta.writeInt(size);
        }

        if (hasRotation()) {
            meta.writeLong(clarOffset);
            meta.writeLong(clarLength);
        }

        OrdToDocDISIReaderConfiguration.writeStoredMeta(DIRECT_MONOTONIC_BLOCK_SHIFT, meta, data, vectorCount, maxDoc, docsWithField);
    }

    /** The on-disk code for a similarity function; the inverse of {@link #similarityFunction}. */
    private static byte similarityCode(VectorSimilarityFunction similarity) {
        return switch (similarity) {
            case EUCLIDEAN -> SIMILARITY_L2;
            case COSINE -> SIMILARITY_COSINE;
            case MAXIMUM_INNER_PRODUCT -> SIMILARITY_MIP;
            case DOT_PRODUCT -> throw new IllegalArgumentException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
        };
    }

    /**
     * A rotation this reader knows how to apply, its {@code .clam} byte mapped through {@link RotationScheme}. An
     * unrecognised code means the segment was written by a format that rotates in a way this one cannot reproduce,
     * so scoring it would be silently wrong.
     */
    private static int rotationId(byte encoded, ChecksumIndexInput meta) throws IOException {
        try {
            return RotationScheme.fromCode(encoded).code();
        } catch (final IllegalArgumentException e) {
            throw new CorruptIndexException("Unknown rotation: " + encoded, meta, e);
        }
    }

    private static VectorSimilarityFunction similarityFunction(byte encoded, ChecksumIndexInput meta) throws IOException {
        return switch (encoded) {
            case SIMILARITY_L2 -> VectorSimilarityFunction.EUCLIDEAN;
            case SIMILARITY_COSINE -> VectorSimilarityFunction.COSINE;
            case SIMILARITY_MIP -> VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
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
    private static void checkClacOffsets(
        ChecksumIndexInput meta,
        long clacOffset,
        long clacLength,
        long clacCentroidsOffset,
        long clacRotatedCentroidsOffset
    ) throws IOException {
        checkNonNegative(meta, clacOffset, "clacOffset");
        checkNonNegative(meta, clacLength, "clacLength");
        checkNonNegative(meta, clacCentroidsOffset, "clacCentroidsOffset");
        checkWithinRegion(meta, clacCentroidsOffset, clacLength, "clacCentroidsOffset", "clac");
        if (clacRotatedCentroidsOffset != NO_ROTATION) {
            checkWithinRegion(meta, clacRotatedCentroidsOffset, clacLength, "clacRotatedCentroidsOffset", "clac");
        }
    }

    /** As {@link #checkClacOffsets}, for {@code .clap} and its one relative offset per centroid. */
    private static void checkClapOffsets(ChecksumIndexInput meta, long clapOffset, long clapLength, long[] clapCentroidOffsets)
        throws IOException {
        checkNonNegative(meta, clapOffset, "clapOffset");
        checkNonNegative(meta, clapLength, "clapLength");
        for (int centroid = 0; centroid < clapCentroidOffsets.length; centroid++) {
            final String name = "clapCentroidOffsets[" + centroid + "]";
            checkNonNegative(meta, clapCentroidOffsets[centroid], name);
            checkWithinRegion(meta, clapCentroidOffsets[centroid], clapLength, name, "clap");
        }
    }

    private static void checkWithinRegion(ChecksumIndexInput meta, long offset, long regionLength, String name, String region)
        throws IOException {
        if (offset > regionLength) {
            throw new CorruptIndexException(
                name + "=" + offset + " is outside the field's ." + region + " region of " + regionLength + " bytes",
                meta
            );
        }
    }

    /**
     * An offset that only a rotated field has must be present exactly when the rotation is. Checked in the
     * record rather than only in {@link #read} so the write side cannot build an entry the read side rejects.
     */
    private static void requireRotationAgrees(int rotationId, long offset, String name) {
        final boolean rotates = RotationScheme.fromCode(rotationId).rotates();
        if (rotates == (offset == NO_ROTATION)) {
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
