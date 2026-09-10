package org.opensearch.knn.clusterann.writer;

import org.opensearch.knn.clusterann.reader.block.scalar.ScalarEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.reader.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.reader.ClusterFactory;
import org.opensearch.knn.clusterann.writer.block.scalar.ScalarQuantizedBlockWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The guards on the write path — the arguments a caller can get wrong, and what happens when they do.
 *
 * <p>These matter more here than they would on the read side. A reader that mis-parses fails on a checksum sooner or
 * later; a writer that accepts a bad argument produces a segment that is internally consistent and wrong, and no later
 * check will notice. So the rejections are part of the format, not defensive noise.
 */
class WriteGuardsTests {

    private static final int DIMENSION = 16;

    @Test
    void params_rejectShapesThatCannotProduceAnIndex() {
        assertThrows(IllegalArgumentException.class, () -> params(0, 8, 1), "blockSize");
        assertThrows(IllegalArgumentException.class, () -> params(8, 0, 1), "targetClusterSize");
        assertThrows(IllegalArgumentException.class, () -> params(8, 8, 0), "docBits");
    }

    /**
     * The centroid count follows from the target posting size, and is capped at the vector count because a cluster
     * needs at least one member — the reader enforces the same bound when it reads the entry back.
     */
    @Test
    void params_centroidCountFollowsTheTargetSize() {
        ClusterANNWriteParams params = params(8, 100, 1);
        assertEquals(0, params.centroidCount(0), "no vectors, no centroids");
        assertEquals(1, params.centroidCount(50), "fewer vectors than the target is still one cluster");
        assertEquals(2, params.centroidCount(150));
        // The cap at the vector count cannot actually bite: ceil(n / target) <= n for any target of one or more. It is
        // kept because the reader rejects centroidCount > vectorCount, and the two bounds should say the same thing.
        assertEquals(1, params.centroidCount(3));
        assertEquals(3, params(8, 1, 1).centroidCount(3), "a target of one gives a cluster per vector");
    }

    @Test
    void clusteringResult_rejectsAssignmentsThatCannotBeWritten() {
        float[][] centroids = { new float[DIMENSION], new float[DIMENSION] };

        assertThrows(
            IllegalArgumentException.class,
            () -> new ClusteringResult(centroids, new int[] { 0 }, new int[0]),
            "parallel arrays must agree in length"
        );
        assertThrows(
            IllegalArgumentException.class,
            () -> new ClusteringResult(centroids, new int[] { 5 }, new int[] { ClusteringResult.NO_SOAR }),
            "a primary outside the centroid range"
        );
        // A spill into the primary's own cluster would be a byte-identical second copy: more bytes, more scoring, and
        // no cluster the vector was not already reachable from.
        assertThrows(
            IllegalArgumentException.class,
            () -> new ClusteringResult(centroids, new int[] { 1 }, new int[] { 1 }),
            "a spill that repeats the primary"
        );

        ClusteringResult valid = new ClusteringResult(centroids, new int[] { 0 }, new int[] { 1 });
        assertEquals(2, valid.numCentroids());
        assertEquals(1, valid.numVectors());
    }

    /** The reader derives where block zero begins from the entry count alone, so both sides must agree exactly. */
    @Test
    void posting_headerBytesMatchWhatTheReaderComputes() {
        assertEquals(0, Posting.headerBytes(0));
        // 4 ordinals + 1 bitset byte + 4 distance bytes
        assertEquals(9, Posting.headerBytes(1));
        // 8 entries fill one bitset byte exactly; the ninth needs a second
        assertEquals(8 * 8 + 1, Posting.headerBytes(8));
        assertEquals(9 * 8 + 2, Posting.headerBytes(9));

        Posting posting = new Posting(3, new int[] { 1, 2 }, new float[] { 0.1f, 0.2f }, new FixedBitSet(2));
        assertEquals(3, posting.centroidOrdinal());
        assertEquals(2, posting.size());
        assertEquals(Posting.headerBytes(2), posting.headerBytes());
    }

    @Test
    void vectorSource_fallsBackToOrdinalsWhenThereAreNoDocIds() throws IOException {
        float[] first = new float[DIMENSION];
        List<float[]> vectors = new ArrayList<>(List.of(first, new float[DIMENSION]));

        VectorSource withDocs = VectorSource.fromList(vectors, new int[] { 4, 9 }, DIMENSION);
        assertEquals(4, withDocs.docId(0));
        assertEquals(9, withDocs.docId(1));

        VectorSource withoutDocs = VectorSource.fromList(vectors, null, DIMENSION);
        assertEquals(0, withoutDocs.docId(0), "no doc ids means the ordinal is the doc");
        assertEquals(1, withoutDocs.docId(1));
        assertEquals(2, withoutDocs.size());
        assertEquals(DIMENSION, withoutDocs.dimension());
        assertSame(first, withoutDocs.vector(0));
    }

    @Test
    void clusterWriterFactory_rejectsAFamilyItCannotWrite() {
        ClusterANNWriteParams unsupported = new ClusterANNWriteParams(
            99,
            1,
            ClusterANNFieldMeta.ROTATION_NONE,
            8,
            16,
            1.0f,
            1L,
            ClusterANNWriteParams.DEFAULT_MONOTONIC_BLOCK_SHIFT
        );
        assertThrows(
            IllegalArgumentException.class,
            () -> ClusterWriterFactory.create(unsupported, DIMENSION, VectorSimilarityFunction.EUCLIDEAN)
        );
        assertNotNull(ClusterWriterFactory.create(params(8, 16, 1), DIMENSION, VectorSimilarityFunction.EUCLIDEAN));
    }

    /**
     * Four-bit doc codes are rejected because the read side's dot kernels do not cover them. Writing a width the
     * reader cannot score would produce a segment that opens and then fails on every query — worse than refusing.
     */
    @Test
    void blockWriter_rejectsAWidthTheReaderCannotScore() {
        assertThrows(
            IllegalArgumentException.class,
            () -> new ScalarQuantizedBlockWriter(
                null,
                8,
                DIMENSION,
                ScalarEncoding.PACKED_NIBBLE,
                new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN)
            )
        );
    }

    @Test
    void blockWriter_rejectsAVectorOfTheWrongLength() {
        ScalarQuantizedBlockWriter writer = new ScalarQuantizedBlockWriter(
            null,
            8,
            DIMENSION,
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE,
            new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN)
        );
        assertEquals(8, writer.blockSize());
        assertTrue(writer.fixedBlockBytes() > 0);
        assertThrows(IllegalArgumentException.class, () -> writer.addVector(new float[DIMENSION + 1], new float[DIMENSION]));
    }

    /** A field with no vectors clusters into nothing, rather than into one empty cluster. */
    @Test
    void clustering_handlesAFieldWithNoVectors() throws IOException {
        ClusteringResult result = new LloydClustering().cluster(
            VectorSource.fromList(List.of(), null, DIMENSION),
            VectorSimilarityFunction.EUCLIDEAN,
            params(8, 16, 1)
        );
        assertEquals(0, result.numCentroids());
        assertEquals(0, result.numVectors());
    }

    /** With one cluster there is nowhere to spill, so SOAR must leave every vector alone. */
    @Test
    void clustering_spillsNothingWhenThereIsOnlyOneCluster() throws IOException {
        List<float[]> vectors = new ArrayList<>();
        for (int ord = 0; ord < 4; ord++) {
            float[] vector = new float[DIMENSION];
            vector[ord % DIMENSION] = 1f + ord;
            vectors.add(vector);
        }
        ClusteringResult result = new LloydClustering().cluster(
            VectorSource.fromList(vectors, null, DIMENSION),
            VectorSimilarityFunction.EUCLIDEAN,
            params(8, 1000, 1)
        );
        assertEquals(1, result.numCentroids());
        for (int soar : result.soar()) {
            assertEquals(ClusteringResult.NO_SOAR, soar);
        }
    }

    private static ClusterANNWriteParams params(int blockSize, int targetClusterSize, int docBits) {
        return new ClusterANNWriteParams(
            ClusterFactory.QUANTIZER_SQ,
            docBits,
            ClusterANNFieldMeta.ROTATION_NONE,
            blockSize,
            targetClusterSize,
            1.0f,
            1L,
            ClusterANNWriteParams.DEFAULT_MONOTONIC_BLOCK_SHIFT
        );
    }
}
