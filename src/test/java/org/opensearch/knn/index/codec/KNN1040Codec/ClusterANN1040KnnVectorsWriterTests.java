/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
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
import java.util.Collections;
import java.util.Random;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

/**
 * Component tests for {@link ClusterANN1040KnnVectorsWriter} that exercise the full
 * write path: vector collection, IVF building, and file serialization.
 */
public class ClusterANN1040KnnVectorsWriterTests extends KNNTestCase {

    private static final int DIM = 32;
    private static final long SEED = 42L;

    // ========== File Creation ==========

    @SneakyThrows
    public void testWriteCreatesAllFourFiles() {
        Path tempDir = createTempDir("clusterann-writer-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vector_field", 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);

                // Add vectors
                addRandomVectors(fieldWriter, 500, DIM, SEED);

                writer.flush(500, null);
                writer.finish();
            }

            // Verify all 4 files exist
            assertTrue("Meta file should exist", fileExists(dir, "seg0.clam"));
            assertTrue("Centroids file should exist", fileExists(dir, "seg0.clac"));
            assertTrue("Postings file should exist", fileExists(dir, "seg0.clap"));
            assertTrue("Quantized file should exist", fileExists(dir, "seg0.claq"));
        }
    }

    // ========== Metadata ==========

    @SneakyThrows
    public void testMetaContainsFieldInfo() {
        Path tempDir = createTempDir("clusterann-meta-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("my_vectors", 3, DIM, VectorSimilarityFunction.DOT_PRODUCT);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, 300, DIM, SEED);
                writer.flush(300, null);
                writer.finish();
            }

            // Read and verify metadata
            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );

                int fieldNumber = metaIn.readInt();
                int numVectors = metaIn.readInt();
                int dimension = metaIn.readInt();
                int numCentroids = metaIn.readInt();
                String metric = metaIn.readString();
                byte docBits = metaIn.readByte();

                assertEquals(3, fieldNumber);
                assertEquals(300, numVectors);
                assertEquals(DIM, dimension);
                assertTrue("Should have at least 1 centroid", numCentroids >= 1);
                assertEquals("INNER_PRODUCT", metric);
                assertEquals(1, docBits);
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
                writer.flush(0, null);
                writer.finish();
            }

            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );

                int marker = metaIn.readInt();
                assertEquals(ClusterANN1040KnnVectorsWriter.END_OF_FIELDS, marker);
            }
        }
    }

    // ========== Centroids ==========

    @SneakyThrows
    public void testCentroidsFileContainsValidFloats() {
        Path tempDir = createTempDir("clusterann-centroids-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();
            int numVectors = 1000;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vec", 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, numVectors, DIM, SEED);
                writer.flush(numVectors, null);
                writer.finish();
            }

            // Read meta to get numCentroids
            int numCentroids;
            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );
                metaIn.readInt(); // fieldNumber
                metaIn.readInt(); // numVectors
                metaIn.readInt(); // dimension
                numCentroids = metaIn.readInt();
            }

            // Verify centroids file has correct size
            try (IndexInput centIn = dir.openInput("seg0.clac", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    centIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );

                // Read all centroids and verify they're valid floats
                for (int c = 0; c < numCentroids; c++) {
                    for (int d = 0; d < DIM; d++) {
                        float val = Float.intBitsToFloat(centIn.readInt());
                        assertFalse("Centroid value should not be NaN", Float.isNaN(val));
                        assertFalse("Centroid value should not be Inf", Float.isInfinite(val));
                    }
                }
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
            int numVectors = 1500;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vec", 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, numVectors, DIM, SEED);
                writer.flush(numVectors, null);
                writer.finish();
            }

            // Read meta
            int numCentroids;
            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );
                metaIn.readInt();
                metaIn.readInt();
                metaIn.readInt();
                numCentroids = metaIn.readInt();
            }

            // Read postings and count total vectors
            try (IndexInput postIn = dir.openInput("seg0.clap", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    postIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );

                // Primary postings
                int totalPrimary = 0;
                for (int c = 0; c < numCentroids; c++) {
                    int size = postIn.readVInt();
                    totalPrimary += size;
                    for (int i = 0; i < size; i++) {
                        int docId = postIn.readVInt();
                        assertTrue("DocId should be in range", docId >= 0 && docId < numVectors);
                    }
                }
                assertEquals("All vectors should be in primary postings", numVectors, totalPrimary);

                // SOAR postings (just verify readable, count may vary)
                int totalSoar = 0;
                for (int c = 0; c < numCentroids; c++) {
                    int size = postIn.readVInt();
                    totalSoar += size;
                    for (int i = 0; i < size; i++) {
                        postIn.readVInt();
                    }
                }
                assertTrue("SOAR should assign some vectors", totalSoar > 0);
            }
        }
    }

    // ========== Quantized Vectors ==========

    @SneakyThrows
    public void testQuantizedFileHasCorrectSize() {
        Path tempDir = createTempDir("clusterann-quant-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();
            int numVectors = 200;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vec", 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, numVectors, DIM, SEED);
                writer.flush(numVectors, null);
                writer.finish();
            }

            // Expected size per vector: packedBytes + 4 correction ints
            int packedBytesPerVec = (DIM + 7) / 8; // 1-bit: 4 bytes for 32 dims
            int bytesPerVector = packedBytesPerVec + 16; // 4 floats as ints

            try (IndexInput quantIn = dir.openInput("seg0.claq", IOContext.READONCE)) {
                int headerSize = CodecUtil.indexHeaderLength(ClusterANN1040KnnVectorsWriter.CODEC_NAME, "");
                int footerSize = CodecUtil.footerLength();
                long dataSize = quantIn.length() - headerSize - footerSize;

                assertEquals("Quantized data size should match numVectors * bytesPerVector", (long) numVectors * bytesPerVector, dataSize);
            }
        }
    }

    @SneakyThrows
    public void testQuantizedCorrectionsAreValid() {
        Path tempDir = createTempDir("clusterann-corrections-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();
            int numVectors = 100;

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vec", 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, numVectors, DIM, SEED);
                writer.flush(numVectors, null);
                writer.finish();
            }

            int packedBytesPerVec = (DIM + 7) / 8;

            try (IndexInput quantIn = dir.openInput("seg0.claq", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    quantIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );

                for (int i = 0; i < numVectors; i++) {
                    // Skip packed codes
                    quantIn.seek(quantIn.getFilePointer() + packedBytesPerVec);

                    // Read corrections
                    float lower = Float.intBitsToFloat(quantIn.readInt());
                    float upper = Float.intBitsToFloat(quantIn.readInt());
                    float correction = Float.intBitsToFloat(quantIn.readInt());
                    int componentSum = quantIn.readInt();

                    assertFalse("lower should not be NaN", Float.isNaN(lower));
                    assertFalse("upper should not be NaN", Float.isNaN(upper));
                    assertFalse("correction should not be NaN", Float.isNaN(correction));
                    assertTrue("upper >= lower", upper >= lower);
                }
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
                FieldInfo fieldInfo = createFieldInfo("empty_vec", 5, DIM, VectorSimilarityFunction.EUCLIDEAN);
                writer.addField(fieldInfo);
                // Don't add any vectors
                writer.flush(0, null);
                writer.finish();
            }

            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );

                assertEquals(5, metaIn.readInt()); // fieldNumber
                assertEquals(0, metaIn.readInt()); // numVectors = 0
            }
        }
    }

    // ========== Multiple Metrics ==========

    @SneakyThrows
    public void testWriteWithCosineMetric() {
        Path tempDir = createTempDir("clusterann-cosine-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vec", 0, DIM, VectorSimilarityFunction.COSINE);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, 400, DIM, SEED);
                writer.flush(400, null);
                writer.finish();
            }

            try (IndexInput metaIn = dir.openInput("seg0.clam", IOContext.READONCE)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                    ClusterANN1040KnnVectorsWriter.VERSION_START,
                    ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                    writeState.segmentInfo.getId(),
                    ""
                );
                metaIn.readInt();
                metaIn.readInt();
                metaIn.readInt();
                metaIn.readInt();
                String metric = metaIn.readString();
                assertEquals("COSINE", metric);
            }
        }
    }

    // ========== Codec Integrity ==========

    @SneakyThrows
    public void testCodecHeadersAndFooters() {
        Path tempDir = createTempDir("clusterann-integrity-test");
        try (Directory dir = new MMapDirectory(tempDir)) {
            SegmentWriteState writeState = createWriteState(dir, "seg0");
            FlatVectorsWriter flatWriter = mockFlatVectorsWriter();

            try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, 1)) {
                FieldInfo fieldInfo = createFieldInfo("vec", 0, DIM, VectorSimilarityFunction.EUCLIDEAN);
                KnnFieldVectorsWriter<?> fieldWriter = writer.addField(fieldInfo);
                addRandomVectors(fieldWriter, 100, DIM, SEED);
                writer.flush(100, null);
                writer.finish();
            }

            // Verify all files have valid headers and footers
            for (String ext : new String[] { "seg0.clam", "seg0.clac", "seg0.clap", "seg0.claq" }) {
                try (IndexInput in = dir.openInput(ext, IOContext.READONCE)) {
                    CodecUtil.checkIndexHeader(
                        in,
                        ClusterANN1040KnnVectorsWriter.CODEC_NAME,
                        ClusterANN1040KnnVectorsWriter.VERSION_START,
                        ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
                        writeState.segmentInfo.getId(),
                        ""
                    );
                    // Verify footer is present (check file length accounts for it)
                    assertTrue("File should have footer", in.length() >= CodecUtil.footerLength());
                }
            }
        }
    }

    // ========== Helpers ==========

    @SuppressWarnings("unchecked")
    private static void addRandomVectors(KnnFieldVectorsWriter<?> fieldWriter, int count, int dim, long seed) throws IOException {
        Random rng = new Random(seed);
        KnnFieldVectorsWriter<float[]> typedWriter = (KnnFieldVectorsWriter<float[]>) fieldWriter;
        for (int i = 0; i < count; i++) {
            float[] vec = new float[dim];
            for (int d = 0; d < dim; d++)
                vec[d] = rng.nextFloat();
            typedWriter.addValue(i, vec);
        }
    }

    private SegmentWriteState createWriteState(Directory dir, String segmentName) {
        SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            segmentName,
            0,
            false,
            false,
            mock(org.apache.lucene.codecs.Codec.class),
            Collections.emptyMap(),
            new byte[16],
            Collections.emptyMap(),
            null
        );
        return new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segmentInfo, new FieldInfos(new FieldInfo[0]), null, IOContext.DEFAULT);
    }

    private FieldInfo createFieldInfo(String name, int number, int dim, VectorSimilarityFunction simFunc) {
        return new FieldInfo(
            name,
            number,
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
    }

    @SuppressWarnings("unchecked")
    private FlatVectorsWriter mockFlatVectorsWriter() throws IOException {
        FlatVectorsWriter flatWriter = mock(FlatVectorsWriter.class);
        FlatFieldVectorsWriter<?> mockFieldWriter = mock(FlatFieldVectorsWriter.class);
        doReturn(mockFieldWriter).when(flatWriter).addField(any(FieldInfo.class));
        return flatWriter;
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
