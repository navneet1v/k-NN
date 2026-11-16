/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.read.block.scalar.JavaAdcScoreComputer;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.read.orchestration.ClusterScan;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;
import org.opensearch.knn.clusterann.write.CentroidsWriter;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidData;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidOffsets;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
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
        assertEquals(0, centroidsInput.bytesRead(), "a cluster that is never scanned must not read its centroid");
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

    @Test
    void testOrdToCentroid_thenReturnsPrimaryAssignmentsFromRegion1() throws IOException {
        int[] ordToCentroid = { 0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 2, 1, 0, 1 };

        LongValues assignments = clustersWithAssignments(ordToCentroid).ordToCentroid();

        for (int ordinal = 0; ordinal < ordToCentroid.length; ordinal++) {
            assertEquals(ordToCentroid[ordinal], assignments.get(ordinal), "ordinal " + ordinal);
        }
    }

    @Test
    void testOrdToCentroid_thenDoesNotReadAssignmentsUntilLookup() throws IOException {
        int[] ordToCentroid = { 0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 2, 1, 0, 1 };
        ClacFixture fixture = clacWithAssignments(ordToCentroid);
        CountingInput input = new CountingInput(fixture.input(), new int[1]);
        Clusters clusters = clusters(
            VectorSimilarityFunction.EUCLIDEAN,
            input,
            fieldMeta(VectorSimilarityFunction.EUCLIDEAN, ordToCentroid.length, fixture.length(), fixture.offsets().clacCentroidsOffset())
        );
        assertEquals(0, input.bytesRead(), "constructing Clusters must not read region 1");

        LongValues assignments = clusters.ordToCentroid();

        assertEquals(0, input.bytesRead(), "building the accessor must not read region 1");
        assertEquals(ordToCentroid[7], assignments.get(7));
        assertEquals(Integer.BYTES, input.bytesRead(), "one lookup must read only its own assignment");
    }

    @Test
    void testOrdToCentroid_thenCreatesQueryPrivateRandomAccessLookup() throws IOException {
        int[] ordToCentroid = { 0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 2, 1, 0, 1 };
        ClacFixture fixture = clacWithAssignments(ordToCentroid);
        CountingInput input = new CountingInput(fixture.input(), new int[1]);
        Clusters clusters = clusters(
            VectorSimilarityFunction.EUCLIDEAN,
            input,
            fieldMeta(VectorSimilarityFunction.EUCLIDEAN, ordToCentroid.length, fixture.length(), fixture.offsets().clacCentroidsOffset())
        );
        input.resetRandomAccessSlices();

        LongValues first = clusters.ordToCentroid();
        LongValues second = clusters.ordToCentroid();

        assertEquals(2, input.randomAccessSlices(), "each query must get its own random-access slice");
        assertNotSame(first, second);
    }

    @Test
    void testOrdToCentroid_whenUsedConcurrently_thenReturnsPrimaryAssignments() throws Exception {
        int[] ordToCentroid = { 0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 2, 1, 0, 1 };
        Clusters clusters = clustersWithAssignments(ordToCentroid);
        LongValues forwardAssignments = clusters.ordToCentroid();
        LongValues backwardAssignments = clusters.ordToCentroid();
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);

        try {
            Future<?> forward = executor.submit(() -> {
                start.await();
                for (int ordinal = 0; ordinal < ordToCentroid.length; ordinal++) {
                    assertEquals(ordToCentroid[ordinal], forwardAssignments.get(ordinal), "forward ordinal " + ordinal);
                }
                return null;
            });
            Future<?> backward = executor.submit(() -> {
                start.await();
                for (int ordinal = ordToCentroid.length - 1; ordinal >= 0; ordinal--) {
                    assertEquals(ordToCentroid[ordinal], backwardAssignments.get(ordinal), "backward ordinal " + ordinal);
                }
                return null;
            });

            start.countDown();

            forward.get();
            backward.get();
        } finally {
            executor.shutdownNow();
        }
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
        when(cluster.scorer(eq(context), eq(accepted), any())).thenReturn(expected);

        // when
        PostingScorer scorer = clusters(VectorSimilarityFunction.EUCLIDEAN).scan().scorer(cluster, scanParams, accepted);

        // then
        assertSame(expected, scorer, "the cluster's own scorer is what a scan returns");
        InOrder inOrder = inOrder(cluster);
        inOrder.verify(cluster).prepareScan(scanParams);
        inOrder.verify(cluster).scorer(eq(context), eq(accepted), any());
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
        when(cluster.scorer(eq(context), isNull(), any())).thenReturn(expected);

        // when / then
        assertSame(expected, clusters(VectorSimilarityFunction.EUCLIDEAN).scan().scorer(cluster, scanParams, null));
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
        scan.scorer(first, scanParams, null);
        scan.scorer(second, scanParams, null);

        // then
        verify(first).scorer(eq(firstContext), isNull(), any());
        verify(second).scorer(eq(secondContext), isNull(), any());
    }

    /**
     * Buffers come from the field's metadata, not from any one cluster, so a scan holds one set and hands the same
     * instance to every cluster it probes. A set per probe is what this is here to prevent.
     */
    @Test
    void testScan_whenTwoClustersAreScanned_thenOneSetOfBuffersIsSharedByBoth() throws IOException {
        // given — a data-blind int8 field, since that is the family that keeps buffers
        Cluster first = mock(Cluster.class);
        Cluster second = mock(Cluster.class);
        ScanParams scanParams = ScanParams.of(new float[DIMENSION]);
        when(first.prepareScan(scanParams)).thenReturn(mock(ScanContext.class));
        when(second.prepareScan(scanParams)).thenReturn(mock(ScanContext.class));
        ClusterScan scan = int8Clusters().scan();

        // when
        scan.scorer(first, scanParams, null);
        scan.scorer(second, scanParams, null);

        // then
        ArgumentCaptor<ScanBuffers> firstBuffers = ArgumentCaptor.forClass(ScanBuffers.class);
        ArgumentCaptor<ScanBuffers> secondBuffers = ArgumentCaptor.forClass(ScanBuffers.class);
        verify(first).scorer(any(), any(), firstBuffers.capture());
        verify(second).scorer(any(), any(), secondBuffers.capture());
        assertNotNull(firstBuffers.getValue(), "an int8 field's scan carries buffers");
        assertSame(firstBuffers.getValue(), secondBuffers.getValue(), "both clusters must be handed the same buffers, not one set each");
    }

    /** A field whose clusters are data-blind int8, so {@code ClusterFactory#newScanBuffers()} returns a real set. */
    private Clusters int8Clusters() throws IOException {
        return clusters(
            VectorSimilarityFunction.EUCLIDEAN,
            open("clac", 4096),
            fieldMeta(VectorSimilarityFunction.EUCLIDEAN, VECTOR_COUNT, 4096L, 64L, ClusterANNFormatConstants.QUANTIZER_DATA_BLIND_INT8)
        );
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
        IOException e = assertThrows(IOException.class, () -> scan.scorer(cluster, scanParams, null));

        // then
        assertEquals("centroid unreadable", e.getMessage(), "the failure surfaces as itself, not as a missing scorer");
        verify(cluster, never()).scorer(any(), any(), any());
    }

    // ---------------------------------------------------------------- helpers

    private Clusters clusters(VectorSimilarityFunction similarity) throws IOException {
        return clusters(similarity, open("clac", 4096), fieldMeta(similarity));
    }

    private Clusters clusters(VectorSimilarityFunction similarity, IndexInput centroids) throws IOException {
        return clusters(similarity, centroids, fieldMeta(similarity));
    }

    private Clusters clusters(VectorSimilarityFunction similarity, IndexInput centroids, ClusterANNFieldMeta fieldMeta) throws IOException {
        // The inputs are the field's own regions, cut at clapOffset and clacOffset, as the reader hands them over.
        return new Clusters(open("clap", 3000), centroids, null, fieldMeta, JavaAdcScoreComputer.INSTANCE);
    }

    private Clusters clustersWithAssignments(int[] ordToCentroid) throws IOException {
        ClacFixture fixture = clacWithAssignments(ordToCentroid);
        return clusters(
            VectorSimilarityFunction.EUCLIDEAN,
            fixture.input(),
            fieldMeta(VectorSimilarityFunction.EUCLIDEAN, ordToCentroid.length, fixture.length(), fixture.offsets().clacCentroidsOffset())
        );
    }

    private ClacFixture clacWithAssignments(int[] ordToCentroid) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        CentroidOffsets offsets;
        try (IndexOutput out = directory.createOutput("clac", IOContext.DEFAULT)) {
            offsets = CentroidsWriter.write(
                out,
                new CentroidData(ordToCentroid, new float[CENTROID_COUNT][DIMENSION]),
                RotationFormats.create(ClusterANNFormatConstants.ROTATION_NONE, DIMENSION)
            );
        }
        IndexInput input = directory.openInput("clac", IOContext.DEFAULT);
        return new ClacFixture(input, input.length(), offsets);
    }

    private static ClusterANNFieldMeta fieldMeta(VectorSimilarityFunction similarity) throws IOException {
        return fieldMeta(similarity, VECTOR_COUNT, 4096L, 64L);
    }

    private static ClusterANNFieldMeta fieldMeta(
        VectorSimilarityFunction similarity,
        int vectorCount,
        long clacLength,
        long clacCentroidsOffset
    ) throws IOException {
        return fieldMeta(similarity, vectorCount, clacLength, clacCentroidsOffset, ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ);
    }

    private static ClusterANNFieldMeta fieldMeta(
        VectorSimilarityFunction similarity,
        int vectorCount,
        long clacLength,
        long clacCentroidsOffset,
        int quantizerId
    ) throws IOException {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            DIMENSION,
            vectorCount,
            CENTROID_COUNT,
            similarity,
            1,                                      // docBits, one bit per dimension
            ClusterANNFormatConstants.ROTATION_NONE,
            quantizerId,
            new byte[0],
            0L,                                     // clacOffset
            clacLength,
            clacCentroidsOffset,
            -1L,        // clacRotatedCentroidsOffset
            CLAP_OFFSET,
            3000L,                                  // clapLength
            CLAP_CENTROID_OFFSETS,
            CENTROID_LENGTHS,
            CLUSTER_SIZES,
            -1L,         // clarLength
            -1L,
            // Dense: the fields under test have a vector per document, so the ordinal is the document id.
            ClusterANNFieldMetaEncoder.denseOrdToDoc(vectorCount)
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

    private record ClacFixture(IndexInput input, long length, CentroidOffsets offsets) {
    }

    /** Counts bytes read, so lazy access is asserted rather than assumed. Clones and slices share the counter. */
    private static final class CountingInput extends org.apache.lucene.store.FilterIndexInput {

        private final int[] bytesRead;
        private final int[] randomAccessSlices;

        private CountingInput(IndexInput in, int[] bytesRead) {
            this(in, bytesRead, new int[1]);
        }

        private CountingInput(IndexInput in, int[] bytesRead, int[] randomAccessSlices) {
            super("counting(" + in + ")", in);
            this.bytesRead = bytesRead;
            this.randomAccessSlices = randomAccessSlices;
        }

        private int bytesRead() {
            return bytesRead[0];
        }

        private int randomAccessSlices() {
            return randomAccessSlices[0];
        }

        private void resetRandomAccessSlices() {
            randomAccessSlices[0] = 0;
        }

        @Override
        public byte readByte() throws IOException {
            bytesRead[0]++;
            return in.readByte();
        }

        @Override
        public void readBytes(byte[] bytes, int offset, int length) throws IOException {
            bytesRead[0] += length;
            in.readBytes(bytes, offset, length);
        }

        @Override
        public int readInt() throws IOException {
            bytesRead[0] += Integer.BYTES;
            return in.readInt();
        }

        @Override
        public void readFloats(float[] floats, int offset, int length) throws IOException {
            bytesRead[0] += length * Float.BYTES;
            in.readFloats(floats, offset, length);
        }

        @Override
        public RandomAccessInput randomAccessSlice(long offset, long length) throws IOException {
            randomAccessSlices[0]++;
            RandomAccessInput slice = in.randomAccessSlice(offset, length);
            return new RandomAccessInput() {
                @Override
                public long length() {
                    return slice.length();
                }

                @Override
                public byte readByte(long position) throws IOException {
                    bytesRead[0]++;
                    return slice.readByte(position);
                }

                @Override
                public short readShort(long position) throws IOException {
                    bytesRead[0] += Short.BYTES;
                    return slice.readShort(position);
                }

                @Override
                public int readInt(long position) throws IOException {
                    bytesRead[0] += Integer.BYTES;
                    return slice.readInt(position);
                }

                @Override
                public long readLong(long position) throws IOException {
                    bytesRead[0] += Long.BYTES;
                    return slice.readLong(position);
                }
            };
        }

        /**
         * Counted through, not around. The centroids are read from a slice of this input, and {@link
         * org.apache.lucene.store.FilterIndexInput#slice} hands back the underlying input — so without this override a
         * read through the slice would go uncounted, and a {@code reads() == 0} assertion would hold whether or not
         * anything was read.
         */
        @Override
        public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
            return new CountingInput(in.slice(sliceDescription, offset, length), bytesRead, randomAccessSlices);
        }

        @Override
        public CountingInput clone() {
            return new CountingInput(in.clone(), bytesRead, randomAccessSlices);
        }
    }
}
