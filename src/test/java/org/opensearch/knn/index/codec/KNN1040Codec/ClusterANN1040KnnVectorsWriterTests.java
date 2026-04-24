/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.*;
import org.opensearch.knn.index.clusterann.codec.*;
import org.opensearch.knn.index.clusterann.codec.*;;

/**
 * Tests for {@link ClusterANN1040KnnVectorsWriter} v2 format (2 files).
 */
public class ClusterANN1040KnnVectorsWriterTests extends KNNTestCase {

    private static final int DIM = 32;
    private static final long SEED = 42L;

    // ========== File Creation ==========

    @SneakyThrows
    public void testWriteCreatesTwoFiles() {
        Path tempDir = createTempDir("clusterann-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();
            int numVectors = 200;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                addFieldWithVectors(writer, numVectors, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.flush(numVectors, null);
                writer.finish();
            }

            assertTrue("Meta file should exist", fileExists(dir, "seg0.clam"));
            assertTrue("Postings file should exist", fileExists(dir, "seg0.clap"));
            assertFalse("Old centroids file should NOT exist", fileExists(dir, "seg0.clac"));
            assertFalse("Old quantized file should NOT exist", fileExists(dir, "seg0.claq"));
        }
    }

    // ========== Meta ==========

    @SneakyThrows
    public void testMetaContainsFieldInfo() {
        Path tempDir = createTempDir("clusterann-meta-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();
            int numVectors = 200;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                addFieldWithVectors(writer, numVectors, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.flush(numVectors, null);
                writer.finish();
            }

            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(metaIn, CODEC_NAME, VERSION_START, VERSION_CURRENT, writeState.segmentInfo.getId(), "");

                int fieldNumber = metaIn.readInt();
                assertTrue("Field number should be non-negative", fieldNumber >= 0);

                int numVecs = metaIn.readInt();
                assertEquals(numVectors, numVecs);

                int dimension = metaIn.readInt();
                assertEquals(DIM, dimension);

                int numCentroids = metaIn.readInt();
                assertTrue("Should have centroids", numCentroids > 0);

                String metricName = metaIn.readString();
                assertEquals("L2", metricName);

                byte docBits = metaIn.readByte();
                assertEquals(1, docBits);

                long postingsOffset = metaIn.readLong();
                assertTrue("Postings offset should be non-negative", postingsOffset >= 0);
            }
        }
    }

    @SneakyThrows
    public void testMetaEndOfFieldsMarker() {
        Path tempDir = createTempDir("clusterann-eof-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                addFieldWithVectors(writer, 200, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.flush(200, null);
                writer.finish();
            }

            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(metaIn, CODEC_NAME, VERSION_START, VERSION_CURRENT, writeState.segmentInfo.getId(), "");

                // Skip field meta + centroids + centroidMap + offsetTable
                // Read until we find END_OF_FIELDS
                long fileLen = metaIn.length();
                long footerLen = CodecUtil.footerLength();
                // Seek to just before footer, the END_OF_FIELDS marker should be there
                metaIn.seek(fileLen - footerLen - 4); // 4 bytes for the int marker
                int marker = metaIn.readInt();
                assertEquals(END_OF_FIELDS, marker);
            }
        }
    }

    // ========== Postings ==========

    @SneakyThrows
    public void testPostingsContainAllVectors() {
        Path tempDir = createTempDir("clusterann-postings-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();
            int numVectors = 2000;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                addFieldWithVectors(writer, numVectors, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.flush(numVectors, null);
                writer.finish();
            }

            // Read meta to get numCentroids
            int numCentroids;
            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(metaIn, CODEC_NAME, VERSION_START, VERSION_CURRENT, writeState.segmentInfo.getId(), "");
                metaIn.readInt(); // fieldNumber
                metaIn.readInt(); // numVectors
                metaIn.readInt(); // dimension
                numCentroids = metaIn.readInt();
                metaIn.readString(); // metricName
                metaIn.readByte(); // docBits
                metaIn.readLong(); // postingsOffset
                // Skip centroidDocCounts + centroidNorms
                metaIn.skipBytes((long) numCentroids * Integer.BYTES);
                metaIn.skipBytes((long) numCentroids * Float.BYTES);
            }

            // Read postings file
            try (IndexInput postIn = dir.openInput("seg0.clap", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(postIn, CODEC_NAME, VERSION_START, VERSION_CURRENT, writeState.segmentInfo.getId(), "");

                // Skip alignment padding (align to SECTION_ALIGNMENT after header)
                long pos = postIn.getFilePointer();
                long aligned = (pos + SECTION_ALIGNMENT - 1) & ~(SECTION_ALIGNMENT - 1);
                postIn.seek(aligned);

                // Read all primary + soar posting lists
                int totalPrimary = 0;
                int totalSoar = 0;
                for (int c = 0; c < numCentroids; c++) {
                    // Primary
                    int[] primaryDocIds = PostingListCodec.read(postIn);
                    totalPrimary += primaryDocIds.length;
                    int primaryOrdCount = postIn.readVInt();
                    int[] primaryOrds = new int[primaryOrdCount];
                    for (int j = 0; j < primaryOrdCount; j++)
                        primaryOrds[j] = postIn.readVInt();
                    assertEquals(primaryDocIds.length, primaryOrds.length);
                    // Skip block-columnar quantized section
                    int packedBytes = ScalarBitEncoding.fromDocBits((byte) 1).docPackedBytes(DIM);
                    long quantBytes = (long) primaryDocIds.length * packedBytes + (long) primaryDocIds.length * Integer.BYTES * 4;
                    postIn.skipBytes(quantBytes);

                    // SOAR
                    int[] soarDocIds = PostingListCodec.read(postIn);
                    totalSoar += soarDocIds.length;
                    int soarOrdCount = postIn.readVInt();
                    int[] soarOrds = new int[soarOrdCount];
                    for (int j = 0; j < soarOrdCount; j++)
                        soarOrds[j] = postIn.readVInt();
                    assertEquals(soarDocIds.length, soarOrds.length);
                    postIn.skipBytes((long) soarDocIds.length * packedBytes + (long) soarDocIds.length * Integer.BYTES * 4);
                }
                assertEquals("All vectors should be in primary postings", numVectors, totalPrimary);
                assertTrue("SOAR should assign some vectors", totalSoar > 0);
            }
        }
    }

    // ========== Empty Field ==========

    @SneakyThrows
    public void testEmptyFieldWritesZeroMeta() {
        Path tempDir = createTempDir("clusterann-empty-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                // Add field but no vectors
                addFieldWithVectors(writer, 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.flush(0, null);
                writer.finish();
            }

            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(metaIn, CODEC_NAME, VERSION_START, VERSION_CURRENT, writeState.segmentInfo.getId(), "");
                metaIn.readInt(); // fieldNumber
                int numVecs = metaIn.readInt();
                assertEquals(0, numVecs);
            }
        }
    }

    // ========== Codec Headers ==========

    @SneakyThrows
    public void testCodecHeadersAndFooters() {
        Path tempDir = createTempDir("clusterann-codec-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                addFieldWithVectors(writer, 100, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.flush(100, null);
                writer.finish();
            }

            // Verify both files have valid headers and footers
            for (String ext : new String[] { "clam", "clap" }) {
                try (IndexInput in = dir.openInput("seg0." + ext, IOContext.READONCE)) {
                    CodecUtil.checkIndexHeader(in, CODEC_NAME, VERSION_START, VERSION_CURRENT, writeState.segmentInfo.getId(), "");
                    in.seek(in.length() - CodecUtil.footerLength());
                    CodecUtil.retrieveChecksum(in);
                }
            }
        }
    }

    // ========== Helpers ==========

    private SegmentWriteState createWriteState(Directory dir, String segmentName) {
        SegmentInfo segInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            segmentName,
            0,
            false,
            false,
            null,
            Collections.emptyMap(),
            new byte[16],
            Collections.emptyMap(),
            null
        );
        FieldInfo fieldInfo = new FieldInfo(
            "test_field",
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            org.apache.lucene.index.DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0,
            0,
            0,
            0,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        return new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segInfo, fieldInfos, null, IOContext.DEFAULT);
    }

    @SuppressWarnings("unchecked")
    private FlatVectorsWriter mockFlatVectorsWriter() throws IOException {
        FlatVectorsWriter flatWriter = mock(FlatVectorsWriter.class);
        FlatFieldVectorsWriter<float[]> fieldWriter = mock(FlatFieldVectorsWriter.class);
        doReturn(fieldWriter).when(flatWriter).addField(any());
        doReturn(new ArrayList<float[]>()).when(fieldWriter).getVectors();
        return flatWriter;
    }

    @SuppressWarnings("unchecked")
    private void addFieldWithVectors(ClusterANN1040KnnVectorsWriter writer, int numVectors, int dim, VectorSimilarityFunction simFunc)
        throws IOException {
        FieldInfo fieldInfo = new FieldInfo(
            "test_field",
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            org.apache.lucene.index.DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0,
            0,
            0,
            dim,
            VectorEncoding.FLOAT32,
            simFunc,
            false,
            false
        );

        FlatFieldVectorsWriter<float[]> flatFieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);

        // Generate vectors
        Random rng = new Random(SEED);
        List<float[]> vectors = new ArrayList<>(numVectors);
        for (int i = 0; i < numVectors; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++)
                v[d] = rng.nextFloat();
            vectors.add(v);
        }

        // Mock the flat field writer to return our vectors
        doReturn(vectors).when(flatFieldWriter).getVectors();
        if (numVectors > 0) {
            DocsWithFieldSet docsWithField = new DocsWithFieldSet();
            for (int i = 0; i < numVectors; i++) {
                docsWithField.add(i);
            }
            doReturn(docsWithField).when(flatFieldWriter).getDocsWithFieldSet();
        }
    }

    private boolean fileExists(Directory dir, String name) {
        try {
            dir.openInput(name, IOContext.READONCE).close();
            return true;
        } catch (IOException e) {
            return false;
        }
    }
}
