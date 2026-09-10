package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.reader.ClusterFactory;
import org.opensearch.knn.clusterann.reader.Clusters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end: write a field with {@link ClustersWriter}, then open it with the read path that has to consume it.
 *
 * <p>This is the test the write path most needs. Every stage records offsets the next one depends on, and the reader
 * addresses everything through them — so an offset measured against the file where it should have been measured
 * against the field's region, or a column written in the wrong order, produces a segment that opens and then reads
 * the wrong bytes. Only running the real reader over the real output catches that.
 */
class ClustersWriterRoundTripTests {

    private static final int DIMENSION = 32;
    private static final int VECTOR_COUNT = 60;
    private static final int FIELD_NUMBER = 0;
    private static final int MAX_DOC = VECTOR_COUNT;

    /** Small enough that 60 vectors make several clusters, so the per-centroid arrays are actually exercised. */
    private static final int TARGET_CLUSTER_SIZE = 16;

    @ParameterizedTest
    @ValueSource(ints = { ClusterANNFieldMeta.ROTATION_NONE, ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN })
    void writtenFieldOpensAndItsClustersAreConsistent(int rotationId) throws IOException {
        ClusterANNWriteParams params = params(rotationId);

        try (Directory directory = new ByteBuffersDirectory()) {
            write(directory, params);

            ClusterANNFieldMeta fieldMeta = readMeta(directory, params);
            assertEquals(DIMENSION, fieldMeta.dimension());
            assertEquals(VECTOR_COUNT, fieldMeta.vectorCount());
            assertEquals(rotationId, fieldMeta.rotationId());
            assertEquals(rotationId != ClusterANNFieldMeta.ROTATION_NONE, fieldMeta.hasRotation());
            assertTrue(fieldMeta.centroidCount() > 1, "the target cluster size should produce several clusters");

            // Every vector is listed once as a primary member, and again wherever SOAR spilled it — so the postings
            // hold at least as many entries as there are vectors, never fewer.
            int listed = 0;
            for (int size : fieldMeta.clusterSizes()) {
                listed += size;
            }
            assertTrue(listed >= VECTOR_COUNT, "postings hold " + listed + " entries for " + VECTOR_COUNT + " vectors");

            // The per-centroid offsets and lengths must tile the field's .clap region without gaps, which is what makes
            // each posting sliceable and is the invariant the reader silently relies on.
            long expected = fieldMeta.clapCentroidOffsets()[0];
            for (int c = 0; c < fieldMeta.centroidCount(); c++) {
                assertEquals(expected, fieldMeta.clapCentroidOffsets()[c], "posting " + c + " starts where " + (c - 1) + " ended");
                expected += fieldMeta.centroidLengths()[c];
            }
            assertEquals(fieldMeta.clapLength(), expected, "the postings must fill the region they claim");

            openAndScan(directory, fieldMeta);
        }
    }

    /** Opens the field the way the reader does, and pulls a scorer out of every cluster. */
    private static void openAndScan(Directory directory, ClusterANNFieldMeta fieldMeta) throws IOException {
        try (
            IndexInput postings = directory.openInput("clap", IOContext.DEFAULT);
            IndexInput centroids = directory.openInput("clac", IOContext.DEFAULT);
            IndexInput rotation = fieldMeta.hasRotation() ? directory.openInput("clar", IOContext.DEFAULT) : null
        ) {
            Clusters clusters = new Clusters(
                postings.slice("clap-field", fieldMeta.clapOffset(), fieldMeta.clapLength()),
                centroids.slice("clac-field", fieldMeta.clacOffset(), fieldMeta.clacLength()),
                fieldMeta.hasRotation() ? rotation.slice("clar-field", fieldMeta.clarOffset(), fieldMeta.clarLength()) : null,
                fieldMeta
            );

            assertEquals(fieldMeta.centroidCount(), clusters.numClusters());
            assertEquals(VECTOR_COUNT, clusters.numVectors());
            assertNotNull(clusters.centroids());

            for (int ordinal = 0; ordinal < clusters.numClusters(); ordinal++) {
                assertEquals(fieldMeta.clusterSizes()[ordinal], clusters.get(ordinal).size(), "cluster " + ordinal);
            }
        }
    }

    private static ClusterANNFieldMeta readMeta(Directory directory, ClusterANNWriteParams params) throws IOException {
        try (ChecksumIndexInput meta = directory.openChecksumInput("clam")) {
            assertEquals(params.blockSize(), meta.readVInt());
            assertEquals(FIELD_NUMBER, meta.readInt());
            return ClusterANNFieldMeta.read(meta, params.blockSize());
        }
    }

    private static void write(Directory directory, ClusterANNWriteParams params) throws IOException {
        try (
            IndexOutput meta = directory.createOutput("clam", IOContext.DEFAULT);
            IndexOutput centroids = directory.createOutput("clac", IOContext.DEFAULT);
            IndexOutput postings = directory.createOutput("clap", IOContext.DEFAULT);
            IndexOutput rotation = params.rotationId() == ClusterANNFieldMeta.ROTATION_NONE
                ? null
                : directory.createOutput("clar", IOContext.DEFAULT)
        ) {
            meta.writeVInt(params.blockSize());
            ClustersWriter writer = new ClustersWriter(meta, centroids, postings, rotation, params, new LloydClustering());
            writer.write(fieldInfo(), vectors(), MAX_DOC);
        }
    }

    /** An empty field writes a full-shape entry with nothing to point at, so the reader parses one layout. */
    @Test
    void emptyFieldStillWritesAnEntry() throws IOException {
        ClusterANNWriteParams params = params(ClusterANNFieldMeta.ROTATION_NONE);
        try (Directory directory = new ByteBuffersDirectory()) {
            try (
                IndexOutput meta = directory.createOutput("clam", IOContext.DEFAULT);
                IndexOutput centroids = directory.createOutput("clac", IOContext.DEFAULT);
                IndexOutput postings = directory.createOutput("clap", IOContext.DEFAULT)
            ) {
                meta.writeVInt(params.blockSize());
                new ClustersWriter(meta, centroids, postings, null, params, new LloydClustering()).write(
                    fieldInfo(),
                    VectorSource.fromList(List.of(), null, DIMENSION),
                    MAX_DOC
                );
            }

            ClusterANNFieldMeta fieldMeta = readMeta(directory, params);
            assertTrue(fieldMeta.isEmpty());
            assertEquals(0, fieldMeta.centroidCount());
            // The dimension is the field's real one, not a placeholder: the reader rejects a non-positive dimension.
            assertEquals(DIMENSION, fieldMeta.dimension());
        }
    }

    private static ClusterANNWriteParams params(int rotationId) {
        return new ClusterANNWriteParams(
            ClusterFactory.QUANTIZER_SQ,
            1,
            rotationId,
            8,
            TARGET_CLUSTER_SIZE,
            1.0f,
            7L,
            ClusterANNWriteParams.DEFAULT_MONOTONIC_BLOCK_SHIFT
        );
    }

    /** Vectors spread over a few loose groups, so clustering has something to find. */
    private static VectorSource vectors() {
        List<float[]> vectors = new ArrayList<>();
        int[] docIds = new int[VECTOR_COUNT];
        for (int ord = 0; ord < VECTOR_COUNT; ord++) {
            float[] vector = new float[DIMENSION];
            int group = ord % 4;
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = (float) (Math.sin(0.21 * i + group * 2.0) + 0.03 * ord);
            }
            vectors.add(vector);
            docIds[ord] = ord;
        }
        return VectorSource.fromList(vectors, docIds, DIMENSION);
    }

    private static FieldInfo fieldInfo() {
        return new FieldInfo(
            "vector",
            FIELD_NUMBER,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
    }
}
