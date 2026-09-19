/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.format;

import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN;

/**
 * Round-trip tests for {@link ClusterANNFieldMeta#write}: it must emit exactly what {@link ClusterANNFieldMeta#read}
 * consumes, so {@code read(write(x))} reproduces {@code x}. Covers both the rotated and unrotated layouts, the
 * trailing ord&rarr;doc config (dense and sparse), plus a guard that the comparison actually catches a mismatch
 * (so the round-trip assertions are not vacuous).
 *
 * <p>The ord&rarr;doc config closes each entry: its fixed part is written to the {@code .clam} meta output, and a
 * sparse field's {@code IndexedDISI}/{@code DirectMonotonic} data stream is written to the separate {@code .clap}
 * data output. A dense field (every doc has the vector) writes only the config marker and no data stream.
 */
class ClusterANNFieldMetaWriteTest {

    private static final int BLOCK_SIZE = 32;
    /** Sentinel for a rotation offset/length that is absent on an unrotated field. */
    private static final long NO_ROTATION = -1L;

    /** The fixed dense ord&rarr;doc config trailing a dense field: offset(-1) | length(0) | jumpTable(-1) | rankPower(-1). */
    private static final int DENSE_ORD_TO_DOC_BYTES = Long.BYTES + Long.BYTES + Short.BYTES + Byte.BYTES;

    @Test
    void write_emitsExactByteLayout() throws IOException {
        ClusterANNFieldMeta meta = rotatedField();
        try (Directory dir = new ByteBuffersDirectory()) {
            writeEntry(dir, meta, /* maxDoc, dense */ 100, denseDocs(100));
            // Read the raw bytes directly (no ClusterANNFieldMeta.read) to pin the on-disk layout write() emits.
            try (IndexInput in = dir.openInput("meta", IOContext.DEFAULT)) {
                assertEquals(128, in.readVInt(), "dimension");
                assertEquals(100, in.readVInt(), "vectorCount");
                assertEquals(2, in.readVInt(), "centroidCount");
                assertEquals((byte) 0, in.readByte(), "similarity code (L2)");
                assertEquals(1, in.readByte(), "docBits");
                assertEquals((byte) ROTATION_RANDOM_GAUSSIAN, in.readByte(), "rotationId");
                assertEquals(0, in.readByte(), "quantizerId");
                assertEquals(2, in.readInt(), "quantizerParamsLength");
                byte[] params = new byte[2];
                in.readBytes(params, 0, 2);
                assertArrayEquals(new byte[] { 7, 8 }, params, "quantizerParams");

                assertEquals(0L, in.readLong(), "clacOffset");
                assertEquals(512L, in.readLong(), "clacLength");
                assertEquals(64L, in.readLong(), "clacCentroidsOffset");
                assertEquals(320L, in.readLong(), "clacRotatedCentroidsOffset (rotated field only)");

                assertEquals(0L, in.readLong(), "clapOffset");
                assertEquals(800L, in.readLong(), "clapLength");
                assertEquals(0L, in.readLong(), "clapCentroidOffsets[0]");
                assertEquals(400L, in.readLong(), "clapCentroidOffsets[1]");
                assertEquals(400, in.readInt(), "centroidLengths[0]");
                assertEquals(400, in.readInt(), "centroidLengths[1]");
                assertEquals(50, in.readInt(), "clusterSizes[0]");
                assertEquals(50, in.readInt(), "clusterSizes[1]");

                assertEquals(900L, in.readLong(), "clarOffset (rotated field only)");
                assertEquals(195L, in.readLong(), "clarLength (rotated field only)");

                // Dense ord->doc config: offset -1 (dense), length 0, no jump table, no rank power.
                assertEquals(-1L, in.readLong(), "ordToDoc docsWithFieldOffset (dense)");
                assertEquals(0L, in.readLong(), "ordToDoc docsWithFieldLength");
                assertEquals((short) -1, in.readShort(), "ordToDoc jumpTableEntryCount");
                assertEquals((byte) -1, in.readByte(), "ordToDoc denseRankPower");
                assertEquals(in.length(), in.getFilePointer(), "no trailing bytes");
            }
            // A dense field stores no ord->doc data stream in .clap.
            try (IndexInput data = dir.openInput("data", IOContext.DEFAULT)) {
                assertEquals(0L, data.length(), "dense field writes no ord->doc data stream");
            }
        }
    }

    @Test
    void write_emitsExactByteLayout_unrotatedOmitsRotationFields() throws IOException {
        ClusterANNFieldMeta meta = unrotatedField();
        try (Directory dir = new ByteBuffersDirectory()) {
            writeEntry(dir, meta, /* maxDoc, dense */ 40, denseDocs(40));
            try (IndexInput in = dir.openInput("meta", IOContext.DEFAULT)) {
                assertEquals(64, in.readVInt(), "dimension");
                assertEquals(40, in.readVInt(), "vectorCount");
                assertEquals(1, in.readVInt(), "centroidCount");
                assertEquals((byte) 1, in.readByte(), "similarity code (IP)");
                assertEquals(1, in.readByte(), "docBits");
                assertEquals((byte) ROTATION_NONE, in.readByte(), "rotationId");
                assertEquals(0, in.readByte(), "quantizerId");
                assertEquals(0, in.readInt(), "quantizerParamsLength (empty)");

                assertEquals(0L, in.readLong(), "clacOffset");
                assertEquals(256L, in.readLong(), "clacLength");
                assertEquals(0L, in.readLong(), "clacCentroidsOffset");
                // No clacRotatedCentroidsOffset for an unrotated field.

                assertEquals(0L, in.readLong(), "clapOffset");
                assertEquals(300L, in.readLong(), "clapLength");
                assertEquals(0L, in.readLong(), "clapCentroidOffsets[0]");
                assertEquals(300, in.readInt(), "centroidLengths[0]");
                assertEquals(40, in.readInt(), "clusterSizes[0]");

                // No clarOffset / clarLength for an unrotated field: the ord->doc config follows immediately.
                assertEquals(
                    DENSE_ORD_TO_DOC_BYTES,
                    in.length() - in.getFilePointer(),
                    "unrotated entry carries no rotation fields, only the dense ord->doc config"
                );
            }
        }
    }

    @Test
    void write_thenRead_roundTripsRotatedField() throws IOException {
        ClusterANNFieldMeta actual = writeThenRead(rotatedField(), 100, denseDocs(100));
        assertEquals(rotatedField(), actual);
        assertTrue(actual.ordToDoc().isDense(), "every doc has the vector -> dense ord->doc");
    }

    @Test
    void write_thenRead_roundTripsUnrotatedField() throws IOException {
        ClusterANNFieldMeta actual = writeThenRead(unrotatedField(), 40, denseDocs(40));
        assertEquals(unrotatedField(), actual);
        assertTrue(actual.ordToDoc().isDense(), "every doc has the vector -> dense ord->doc");
    }

    @Test
    void write_thenRead_roundTripsSparseOrdToDoc() throws IOException {
        DocsWithFieldSet docs = docs(0, 4, 9); // 3 of 10 docs carry the field
        try (Directory dir = new ByteBuffersDirectory()) {
            writeEntry(dir, sparseField(), /* maxDoc */ 10, docs);
            try (ChecksumIndexInput in = dir.openChecksumInput("meta")) {
                ClusterANNFieldMeta actual = ClusterANNFieldMeta.read(in, BLOCK_SIZE);
                assertEquals(sparseField(), actual);
                assertFalse(actual.ordToDoc().isDense(), "not every doc has the vector -> sparse");
                assertFalse(actual.ordToDoc().isEmpty(), "the field has vectors");
            }
            // A sparse field writes its IndexedDISI + DirectMonotonic stream to the .clap data output.
            try (IndexInput data = dir.openInput("data", IOContext.DEFAULT)) {
                assertTrue(data.length() > 0, "sparse field writes an ord->doc data stream");
            }
        }
    }

    @Test
    void write_thenRead_comparisonFailsOnMismatch() throws IOException {
        ClusterANNFieldMeta actual = writeThenRead(rotatedField(), 100, denseDocs(100));

        // The bytes write() emitted read back as exactly what was written.
        assertEquals(rotatedField(), actual);

        // A field that differs from what was written must not compare equal, so the round-trip check above is
        // meaningful and not trivially satisfied.
        ClusterANNFieldMeta mismatch = new ClusterANNFieldMeta(
            BLOCK_SIZE,
            129,
            100,
            2,
            VectorSimilarityFunction.EUCLIDEAN,
            1,
            ROTATION_RANDOM_GAUSSIAN,
            0,
            new byte[] { 7, 8 },
            0L,
            512L,
            64L,
            320L,
            0L,
            800L,
            new long[] { 0L, 400L },
            new int[] { 400, 400 },
            new int[] { 50, 50 },
            900L,
            195L,
            null
        );
        assertNotEquals(mismatch, actual);
    }

    private static ClusterANNFieldMeta rotatedField() {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            128,
            100,
            2,
            VectorSimilarityFunction.EUCLIDEAN,
            1,
            ROTATION_RANDOM_GAUSSIAN,
            0,
            new byte[] { 7, 8 },
            0L,
            512L,
            64L,
            320L,
            0L,
            800L,
            new long[] { 0L, 400L },
            new int[] { 400, 400 },
            new int[] { 50, 50 },
            900L,
            195L,
            null
        );
    }

    private static ClusterANNFieldMeta unrotatedField() {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            64,
            40,
            1,
            VectorSimilarityFunction.DOT_PRODUCT,
            1,
            ROTATION_NONE,
            0,
            new byte[0],
            0L,
            256L,
            0L,
            NO_ROTATION,
            0L,
            300L,
            new long[] { 0L },
            new int[] { 300 },
            new int[] { 40 },
            NO_ROTATION,
            NO_ROTATION,
            null
        );
    }

    private static ClusterANNFieldMeta sparseField() {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            8,
            3,
            1,
            VectorSimilarityFunction.DOT_PRODUCT,
            1,
            ROTATION_NONE,
            0,
            new byte[0],
            0L,
            96L,
            0L,
            NO_ROTATION,
            0L,
            120L,
            new long[] { 0L },
            new int[] { 120 },
            new int[] { 3 },
            NO_ROTATION,
            NO_ROTATION,
            null
        );
    }

    /** Writes {@code meta} then reads it back with {@link ClusterANNFieldMeta#read}. */
    private static ClusterANNFieldMeta writeThenRead(ClusterANNFieldMeta meta, int maxDoc, DocsWithFieldSet docs) throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            writeEntry(dir, meta, maxDoc, docs);
            try (ChecksumIndexInput in = dir.openChecksumInput("meta")) {
                return ClusterANNFieldMeta.read(in, meta.blockSize());
            }
        }
    }

    /** Writes one entry to a {@code "meta"} output and its sparse ord&rarr;doc stream to a {@code "data"} output. */
    private static void writeEntry(Directory dir, ClusterANNFieldMeta meta, int maxDoc, DocsWithFieldSet docs) throws IOException {
        try (
            IndexOutput metaOut = dir.createOutput("meta", IOContext.DEFAULT);
            IndexOutput dataOut = dir.createOutput("data", IOContext.DEFAULT)
        ) {
            meta.write(metaOut, dataOut, maxDoc, docs);
        }
    }

    /** A {@link DocsWithFieldSet} where every doc in {@code [0, count)} carries the field (the dense case). */
    private static DocsWithFieldSet denseDocs(int count) throws IOException {
        DocsWithFieldSet docs = new DocsWithFieldSet();
        for (int doc = 0; doc < count; doc++) {
            docs.add(doc);
        }
        return docs;
    }

    /** A {@link DocsWithFieldSet} with vectors only at the given doc ids, in increasing order. */
    private static DocsWithFieldSet docs(int... docIds) throws IOException {
        DocsWithFieldSet docs = new DocsWithFieldSet();
        for (int doc : docIds) {
            docs.add(doc);
        }
        return docs;
    }

}
