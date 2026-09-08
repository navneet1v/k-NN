/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.InOrder;
import org.opensearch.knn.clusterann.reader.orchestration.ClusterScan;
import org.opensearch.knn.clusterann.reader.orchestration.ScanContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Fast, isolated unit tests for {@link Clusters} (JUnit 5).
 *
 * <p>{@link Clusters} is the query-independent handle a search runs over, shared across concurrent searches — so what
 * matters is that it answers layout questions without reading, hands out clusters addressed by ordinal, and gives each
 * one private cursors rather than sharing its own.
 */
class ClustersTests {

    private static final int DIMENSION = 4;
    private static final int CENTROID_COUNT = 3;
    private static final int VECTOR_COUNT = 19;
    private static final int BLOCK_SIZE = 4;

    /** Different per cluster, so a cluster taking the wrong entry is visible. */
    private static final int[] CLUSTER_SIZES = { 10, 6, 3 };

    private static final long[] CLAP_CENTROID_OFFSETS = { 0L, 1000L, 2000L };
    private static final int[] CENTROID_LENGTHS = { 1000, 1000, 1000 };
    private static final long CLAP_OFFSET = 512L;

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- layout questions

    /** Answered from the metadata alone, so a planner can rank clusters without touching any posting. */
    @Test
    void testGeometry_thenComesFromTheFieldMetadata() throws IOException {
        // given / when
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN);

        // then
        assertEquals(CENTROID_COUNT, clusters.numClusters());
        assertEquals(VECTOR_COUNT, clusters.numVectors());
    }

    @ParameterizedTest(name = "cluster {0} holds {1}")
    @CsvSource({ "0, 10", "1, 6", "2, 3" })
    void testClusterSize_thenIsAvailableWithoutBuildingTheCluster(int ordinal, int expectedSize) throws IOException {
        // given / when / then
        assertEquals(expectedSize, clusters(VectorSimilarityFunction.EUCLIDEAN).clusterSize(ordinal));
    }

    // ---------------------------------------------------------------- get

    @ParameterizedTest(name = "cluster {0}")
    @CsvSource({ "0, 10", "1, 6", "2, 3" })
    void testGet_thenReturnsTheClusterForThatOrdinal(int ordinal, int expectedSize) throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN);

        // when
        Cluster cluster = clusters.get(ordinal);

        // then
        assertEquals(ordinal, cluster.ordinal());
        assertEquals(expectedSize, cluster.size());
    }

    @Test
    void testGet_thenEachClusterIsItsOwnObject() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN);

        // when
        Cluster first = clusters.get(0);
        Cluster second = clusters.get(0);

        // then
        assertNotSame(first, second, "the same ordinal twice still means two independent cursors");
        assertEquals(first.ordinal(), second.ordinal());
    }

    @Test
    void testGet_thenReadsNothing() throws IOException {
        // given
        CountingInput centroidsInput = countingInput("clac", 4096);
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, centroidsInput);

        // when
        for (int ordinal = 0; ordinal < CENTROID_COUNT; ordinal++) {
            clusters.get(ordinal);
        }

        // then
        assertEquals(0, centroidsInput.reads(), "a cluster that is never scanned must not read its centroid");
    }

    @Test
    void testGet_whenOrdinalIsOutOfRange_thenThrows() throws IOException {
        // given
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN);

        // when / then — the ordinal indexes the metadata's per-cluster arrays
        assertThrows(IndexOutOfBoundsException.class, () -> clusters.get(CENTROID_COUNT));
    }

    // ---------------------------------------------------------------- centroid region

    /**
     * Where the transformed centroids begin depends on the field's similarity: raw records carry a trailing norm for
     * L2 and not otherwise, so the raw region is a different length and the transformed one starts elsewhere. Both
     * must resolve to a readable region rather than off the end of the file.
     */
    @ParameterizedTest(name = "{0}")
    @EnumSource(VectorSimilarityFunction.class)
    void testConstructor_thenLocatesTheTransformedCentroidsForEverySimilarity(VectorSimilarityFunction similarity) throws IOException {
        // given / when
        Clusters clusters = clusters(similarity);

        // then — the region is only read when a cluster is scanned, so what is asserted here is that it was locatable
        assertEquals(CENTROID_COUNT, clusters.numClusters());
        assertTrue(clusters.get(0).size() > 0);
    }

    // ---------------------------------------------------------------- scan

    /**
     * A scan is the two-step handshake in one call: the query is prepared against the cluster, and the context that
     * comes back is the one the scorer is built from — the cluster is handed nothing it did not produce itself. The
     * mock is the whole point here, since what is under test is the sequence rather than either step.
     */
    @Test
    void testScan_thenPreparesTheQueryBeforeScoringWithIt() throws IOException {
        // given
        Cluster cluster = mock(Cluster.class);
        ScanContext context = mock(ScanContext.class);
        PostingScorer expected = mock(PostingScorer.class);
        ScanParams scanParams = ScanParams.of(new float[DIMENSION]);
        Bits accepted = new FixedBitSet(VECTOR_COUNT);
        when(cluster.prepareScan(scanParams)).thenReturn(context);
        when(cluster.scorer(context, accepted)).thenReturn(expected);

        // when
        PostingScorer scorer = clusters(VectorSimilarityFunction.EUCLIDEAN).scan().scan(cluster, scanParams, accepted);

        // then
        assertSame(expected, scorer, "the cluster's own scorer is what a scan returns");
        InOrder inOrder = inOrder(cluster);
        inOrder.verify(cluster).prepareScan(scanParams);
        inOrder.verify(cluster).scorer(context, accepted);
    }

    /** No filter means every ordinal, which is the cluster's business — the scan must not translate it into one. */
    @Test
    void testScan_whenThereIsNoFilter_thenPassesItThroughUntouched() throws IOException {
        // given
        Cluster cluster = mock(Cluster.class);
        ScanContext context = mock(ScanContext.class);
        PostingScorer expected = mock(PostingScorer.class);
        ScanParams scanParams = ScanParams.of(new float[DIMENSION]);
        when(cluster.prepareScan(scanParams)).thenReturn(context);
        when(cluster.scorer(context, null)).thenReturn(expected);

        // when / then
        assertSame(expected, clusters(VectorSimilarityFunction.EUCLIDEAN).scan().scan(cluster, scanParams, null));
    }

    /**
     * Preparation is relative to one centroid, so a context cannot cross clusters. One scanner serving several
     * clusters therefore has to prepare per cluster and score each with its own context.
     */
    @Test
    void testScan_whenTwoClustersAreScanned_thenEachIsScoredWithItsOwnContext() throws IOException {
        // given
        Cluster first = mock(Cluster.class);
        Cluster second = mock(Cluster.class);
        ScanContext firstContext = mock(ScanContext.class);
        ScanContext secondContext = mock(ScanContext.class);
        ScanParams scanParams = ScanParams.of(new float[DIMENSION]);
        when(first.prepareScan(scanParams)).thenReturn(firstContext);
        when(second.prepareScan(scanParams)).thenReturn(secondContext);
        ClusterScan scan = clusters(VectorSimilarityFunction.EUCLIDEAN).scan();

        // when — the same scanner, as a query reuses it across every cluster it probes
        scan.scan(first, scanParams, null);
        scan.scan(second, scanParams, null);

        // then
        verify(first).scorer(firstContext, null);
        verify(second).scorer(secondContext, null);
    }

    /** Preparing reads the centroid, so it can fail. Scoring against a context that was never built cannot happen. */
    @Test
    void testScan_whenPreparingTheQueryFails_thenNothingIsScored() throws IOException {
        // given
        Cluster cluster = mock(Cluster.class);
        ScanParams scanParams = ScanParams.of(new float[DIMENSION]);
        when(cluster.prepareScan(scanParams)).thenThrow(new IOException("centroid unreadable"));
        ClusterScan scan = clusters(VectorSimilarityFunction.EUCLIDEAN).scan();

        // when
        IOException e = assertThrows(IOException.class, () -> scan.scan(cluster, scanParams, null));

        // then
        assertEquals("centroid unreadable", e.getMessage(), "the failure surfaces as itself, not as a missing scorer");
        verify(cluster, never()).scorer(any(), any());
    }

    // ---------------------------------------------------------------- helpers

    private Clusters clusters(VectorSimilarityFunction similarity) throws IOException {
        return clusters(similarity, open("clac", 4096));
    }

    private Clusters clusters(VectorSimilarityFunction similarity, IndexInput centroids) throws IOException {
        // The inputs are the field's own regions, cut at clapOffset and clacOffset, as the reader hands them over.
        return new Clusters(open("clap", 3000), centroids, null, fieldMeta(similarity));
    }

    private static ClusterANNFieldMeta fieldMeta(VectorSimilarityFunction similarity) {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            DIMENSION,
            VECTOR_COUNT,
            CENTROID_COUNT,
            similarity,
            1,                                      // docBits, one bit per dimension
            ClusterANNFieldMeta.ROTATION_NONE,
            ClusterFactory.QUANTIZER_SQ,
            new byte[0],
            0L,                                     // clacOffset
            4096L,                                  // clacLength
            64L,                                    // clacCentroidsOffset
            ClusterANNFieldMeta.NO_ROTATION,        // clacRotatedCentroidsOffset
            CLAP_OFFSET,
            3000L,                                  // clapLength
            CLAP_CENTROID_OFFSETS,
            CENTROID_LENGTHS,
            CLUSTER_SIZES,
            ClusterANNFieldMeta.NO_ROTATION,        // clarOffset
            ClusterANNFieldMeta.NO_ROTATION         // clarLength
        );
    }

    private IndexInput open(String name, int bytes) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(new byte[bytes], 0, bytes);
        }
        return directory.openInput(name, IOContext.DEFAULT);
    }

    private CountingInput countingInput(String name, int bytes) throws IOException {
        return new CountingInput(open(name, bytes), new int[1]);
    }

    /** Counts reads, so "get() reads nothing" is asserted rather than assumed. Clones share the counter. */
    private static final class CountingInput extends org.apache.lucene.store.FilterIndexInput {

        private final int[] reads;

        private CountingInput(IndexInput in, int[] reads) {
            super("counting(" + in + ")", in);
            this.reads = reads;
        }

        private int reads() {
            return reads[0];
        }

        @Override
        public byte readByte() throws IOException {
            reads[0]++;
            return in.readByte();
        }

        @Override
        public void readBytes(byte[] bytes, int offset, int length) throws IOException {
            reads[0]++;
            in.readBytes(bytes, offset, length);
        }

        @Override
        public void readFloats(float[] floats, int offset, int length) throws IOException {
            reads[0]++;
            in.readFloats(floats, offset, length);
        }

        @Override
        public CountingInput clone() {
            return new CountingInput(in.clone(), reads);
        }
    }
}
