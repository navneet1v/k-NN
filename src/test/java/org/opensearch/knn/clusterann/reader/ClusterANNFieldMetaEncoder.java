/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

/**
 * Writes one field's entry in the {@code .clam} layout, so a test can hand
 * {@link ClusterANNFieldMeta#read} bytes it controls value by value — including values the record
 * itself would refuse to hold.
 *
 * <p>Deliberately an independent encoder rather than a call back into the record: until the real writer
 * exists this is the only description of the layout on the write side, so if the two ever disagree the
 * round-trip tests are what notice.
 *
 * <p>Every value defaults to something valid and self-consistent, so a case sets only what it is about.
 * Fluent setters, all returning {@code this}.
 */
public final class ClusterANNFieldMetaEncoder {

    public static final byte SIMILARITY_L2 = 0;
    public static final byte SIMILARITY_IP = 1;
    public static final byte SIMILARITY_COSINE = 2;

    private int dimension = 8;
    private int vectorCount = 30;
    private int centroidCount = 3;
    private byte similarityFunction = SIMILARITY_L2;
    private byte docBits = 20;
    private byte quantizerId = 0;
    private byte[] quantizerParams = new byte[0];

    /** Written in place of {@code quantizerParams.length} when set, so a case can lie about the length. */
    private Integer quantizerParamsLength = null;

    /**
     * Written in place of {@link #centroidCount} when set, while the arrays keep their real length — so a case
     * can claim more clusters than the file could possibly hold without the encoder itself allocating them.
     */
    private Integer declaredCentroidCount = null;

    private long clacOffset = 0L;
    private long clacLength = 512L;
    private long clacCentroidsOffset = 64L;
    private long clacRotatedCentroidsOffset = 128L;

    private long clapOffset = 0L;
    private long clapLength = 900L;
    private long[] clapCentroidOffsets = { 0L, 100L, 300L };
    private int[] centroidLengths = { 100, 200, 600 };
    private int[] clusterSizes = { 10, 12, 8 };

    private byte rotationId = (byte) ClusterANNFieldMeta.ROTATION_NONE;

    /** Written only when {@link #rotationId} is not {@link ClusterANNFieldMeta#ROTATION_NONE}, like {@link #clacRotatedCentroidsOffset}. */
    private long clarOffset = 256L;

    /** Written on the same terms as {@link #clarOffset}. */
    private long clarLength = 64L;

    public ClusterANNFieldMetaEncoder dimension(int dimension) {
        this.dimension = dimension;
        return this;
    }

    public ClusterANNFieldMetaEncoder vectorCount(int vectorCount) {
        this.vectorCount = vectorCount;
        return this;
    }

    /**
     * Sets the centroid count and regenerates the three arrays sized by it, since the reader reads exactly
     * {@code centroidCount} entries of each and a case that changes the count rarely cares about the values.
     */
    public ClusterANNFieldMetaEncoder centroidCount(int centroidCount) {
        this.centroidCount = centroidCount;
        this.clapCentroidOffsets = new long[Math.max(centroidCount, 0)];
        this.centroidLengths = new int[Math.max(centroidCount, 0)];
        this.clusterSizes = new int[Math.max(centroidCount, 0)];
        for (int i = 0; i < clapCentroidOffsets.length; i++) {
            clapCentroidOffsets[i] = i * 100L;
            centroidLengths[i] = 100;
            clusterSizes[i] = 1;
        }
        return this;
    }

    public ClusterANNFieldMetaEncoder similarityFunction(byte similarityFunction) {
        this.similarityFunction = similarityFunction;
        return this;
    }

    public ClusterANNFieldMetaEncoder docBits(byte docBits) {
        this.docBits = docBits;
        return this;
    }

    public ClusterANNFieldMetaEncoder quantizerId(byte quantizerId) {
        this.quantizerId = quantizerId;
        return this;
    }

    public ClusterANNFieldMetaEncoder quantizerParams(byte[] quantizerParams) {
        this.quantizerParams = quantizerParams;
        return this;
    }

    public ClusterANNFieldMetaEncoder quantizerParamsLength(int quantizerParamsLength) {
        this.quantizerParamsLength = quantizerParamsLength;
        return this;
    }

    public ClusterANNFieldMetaEncoder declaredCentroidCount(int declaredCentroidCount) {
        this.declaredCentroidCount = declaredCentroidCount;
        return this;
    }

    public ClusterANNFieldMetaEncoder clacOffset(long clacOffset) {
        this.clacOffset = clacOffset;
        return this;
    }

    public ClusterANNFieldMetaEncoder clacLength(long clacLength) {
        this.clacLength = clacLength;
        return this;
    }

    public ClusterANNFieldMetaEncoder clacCentroidsOffset(long clacCentroidsOffset) {
        this.clacCentroidsOffset = clacCentroidsOffset;
        return this;
    }

    public ClusterANNFieldMetaEncoder clacRotatedCentroidsOffset(long clacRotatedCentroidsOffset) {
        this.clacRotatedCentroidsOffset = clacRotatedCentroidsOffset;
        return this;
    }

    public ClusterANNFieldMetaEncoder clapOffset(long clapOffset) {
        this.clapOffset = clapOffset;
        return this;
    }

    public ClusterANNFieldMetaEncoder clapLength(long clapLength) {
        this.clapLength = clapLength;
        return this;
    }

    public ClusterANNFieldMetaEncoder clapCentroidOffsets(long... clapCentroidOffsets) {
        this.clapCentroidOffsets = clapCentroidOffsets;
        return this;
    }

    public ClusterANNFieldMetaEncoder centroidLengths(int... centroidLengths) {
        this.centroidLengths = centroidLengths;
        return this;
    }

    public ClusterANNFieldMetaEncoder clusterSizes(int... clusterSizes) {
        this.clusterSizes = clusterSizes;
        return this;
    }

    public ClusterANNFieldMetaEncoder rotationId(byte rotationId) {
        this.rotationId = rotationId;
        return this;
    }

    /**
     * Whether this entry describes a rotated field, and so whether the layout carries its rotation values at all —
     * which is also what decides whether the segment has a {@code .clar} file for a reader to open.
     */
    public boolean hasRotation() {
        return rotationId != ClusterANNFieldMeta.ROTATION_NONE;
    }

    public ClusterANNFieldMetaEncoder clarOffset(long clarOffset) {
        this.clarOffset = clarOffset;
        return this;
    }

    public ClusterANNFieldMetaEncoder clarLength(long clarLength) {
        this.clarLength = clarLength;
        return this;
    }

    /** Writes the entry, in the order {@link ClusterANNFieldMeta#read} reads it. */
    public void write(DataOutput out) throws IOException {
        out.writeVInt(dimension);
        out.writeVInt(vectorCount);
        out.writeVInt(declaredCentroidCount == null ? centroidCount : declaredCentroidCount);
        out.writeByte(similarityFunction);
        out.writeByte(docBits);
        out.writeByte(rotationId);
        out.writeByte(quantizerId);
        out.writeInt(quantizerParamsLength == null ? quantizerParams.length : quantizerParamsLength);
        out.writeBytes(quantizerParams, 0, quantizerParams.length);

        out.writeLong(clacOffset);
        out.writeLong(clacLength);
        out.writeLong(clacCentroidsOffset);
        if (rotationId != ClusterANNFieldMeta.ROTATION_NONE) {
            out.writeLong(clacRotatedCentroidsOffset);
        }

        out.writeLong(clapOffset);
        out.writeLong(clapLength);
        for (int i = 0; i < centroidCount; i++) {
            out.writeLong(clapCentroidOffsets[i]);
        }
        for (int i = 0; i < centroidCount; i++) {
            out.writeInt(centroidLengths[i]);
        }
        for (int i = 0; i < centroidCount; i++) {
            out.writeInt(clusterSizes[i]);
        }

        if (rotationId != ClusterANNFieldMeta.ROTATION_NONE) {
            out.writeLong(clarOffset);
            out.writeLong(clarLength);
        }

        writeDenseOrdToDoc(out);
    }

    /**
     * The ord-to-doc mapping's stored meta, in the dense form: every document carries a vector, so the ordinal
     * <em>is</em> the document id and there is no mapping to store. {@code -1} is the marker that says so, and it is
     * what stops {@link org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration#fromStoredMeta} reading the
     * addresses that would follow for a sparse field.
     */
    public static void writeDenseOrdToDoc(DataOutput out) throws IOException {
        out.writeLong(-1L);         // docsWithFieldOffset, -1 meaning dense
        out.writeLong(0L);          // docsWithFieldLength
        out.writeShort((short) 0);  // jumpTableEntryCount
        out.writeByte((byte) 0);    // denseRankPower
    }

    /**
     * A dense configuration, for a test that builds a {@link ClusterANNFieldMeta} directly rather than decoding one.
     *
     * @param vectorCount the field's vector count, which the configuration carries as its size
     */
    public static OrdToDocDISIReaderConfiguration denseOrdToDoc(int vectorCount) throws IOException {
        try (Directory directory = new ByteBuffersDirectory()) {
            try (IndexOutput out = directory.createOutput("ordToDoc", IOContext.DEFAULT)) {
                writeDenseOrdToDoc(out);
            }
            try (IndexInput in = directory.openInput("ordToDoc", IOContext.DEFAULT)) {
                return OrdToDocDISIReaderConfiguration.fromStoredMeta(in, vectorCount);
            }
        }
    }
}
