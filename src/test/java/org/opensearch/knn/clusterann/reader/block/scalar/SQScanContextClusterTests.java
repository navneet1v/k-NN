/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.reader.Centroid;
import org.opensearch.knn.clusterann.reader.PostingScorer;
import org.opensearch.knn.clusterann.reader.ScanParams;
import org.opensearch.knn.clusterann.reader.orchestration.ScanContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, isolated unit tests for {@link ScalarQuantizedCluster} (JUnit 5).
 */
class SQScanContextClusterTests {

    private static final ScalarEncoding ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
    private static final int DIMENSION = 64;
    private static final int PACKED_BYTES = ENCODING.getDocPackedLength(DIMENSION);

    private static final int BLOCK_SIZE = 4;
    private static final int CLUSTER_SIZE = 10;
    private static final int CENTROID_ORDINAL = 7;
    private static final String FILE = "posting";

    /** Global ordinals, ascending by distance from the centroid rather than by ordinal. */
    private static final int[] ORDINALS = { 40, 3, 91, 12, 55, 7, 88, 21, 64, 30 };

    /** Positions 1, 4 and 9 are SOAR copies — deliberately not byte-aligned, so the bit maths has to be right. */
    private static final int[] SOAR_POSITIONS = { 1, 4, 9 };

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- header

    @Test
    void testConstructor_thenReportsItsIdentityAndSizeWithoutReadingBlocks() throws IOException {
        // given / when
        ScalarQuantizedCluster cluster = cluster();

        // then
        assertEquals(CENTROID_ORDINAL, cluster.ordinal());
        assertEquals(CLUSTER_SIZE, cluster.size());
    }

    // ---------------------------------------------------------------- scoring

    /**
     * The end-to-end shape: the cluster hands out a scorer whose ordinals are its own, in posting order. If the
     * header were read short, the block slice would start mid-header and these ordinals would be wrong.
     */
    @Test
    void testScorer_whenNothingIsFiltered_thenVisitsEveryEntryInPostingOrder() throws IOException {
        // given
        PostingScorer scorer = cluster().scorer(scanContext(), null);

        // when
        List<Integer> visited = drain(scorer);

        // then
        int[] actual = visited.stream().mapToInt(Integer::intValue).toArray();
        assertArrayEquals(ORDINALS, actual, "every entry must be scored, in the order the posting stores it");
    }

    @Test
    void testScorer_whenOrdinalsAreFiltered_thenVisitsOnlyThoseAccepted() throws IOException {
        // given — accept three ordinals spread across different blocks
        FixedBitSet accepted = new FixedBitSet(128);
        accepted.set(ORDINALS[0]);
        accepted.set(ORDINALS[5]);
        accepted.set(ORDINALS[9]);
        PostingScorer scorer = cluster().scorer(scanContext(), accepted);

        // when
        List<Integer> visited = drain(scorer);

        // then
        assertArrayEquals(new int[] { ORDINALS[0], ORDINALS[5], ORDINALS[9] }, visited.stream().mapToInt(Integer::intValue).toArray());
    }

    @Test
    void testScorer_thenEveryScoreIsAUsableSimilarity() throws IOException {
        // given
        PostingScorer scorer = cluster().scorer(scanContext(), null);

        // when / then
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            assertTrue(scorer.score() >= 0f, "a euclidean ADC similarity is non-negative, got " + scorer.score());
            assertFalse(Float.isNaN(scorer.score()), "score must not be NaN");
        }
    }

    /**
     * Scorers share the cluster's one reader, so a second scorer has to start from block 0 rather than resume
     * where the first stopped. That holds only because advance() re-seeks unconditionally — used sequentially,
     * as a query does, reuse is safe.
     */
    @Test
    void testScorer_whenCalledTwice_thenEachWalksTheWholePosting() throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();

        // when
        List<Integer> first = drain(cluster.scorer(scanContext(), null));
        List<Integer> second = drain(cluster.scorer(scanContext(), null));

        // then
        assertEquals(CLUSTER_SIZE, first.size());
        assertEquals(first, second, "a fresh scorer starts at the beginning of the posting");
    }

    // ---------------------------------------------------------------- prepareScan

    /**
     * The whole point of the two-step contract: a query goes in, and what comes back drives a real scan of this
     * cluster. Every other scan test hand-builds a context, so this is the only place the quantisation path itself is
     * exercised.
     */
    @Test
    void testPrepareScan_thenProducesAContextThatScansThisCluster() throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();

        // when
        ScanContext context = cluster.prepareScan(ScanParams.of(query()));
        List<Integer> visited = drain(cluster.scorer(context, null));

        // then
        assertArrayEquals(ORDINALS, visited.stream().mapToInt(Integer::intValue).toArray());
    }

    /** Every score from a prepared context has to be usable, not merely produced. */
    @Test
    void testPrepareScan_thenEveryScoreIsAUsableSimilarity() throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();

        // when
        PostingScorer scorer = cluster.scorer(cluster.prepareScan(ScanParams.of(query())), null);

        // then
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            assertTrue(scorer.score() >= 0f, "a euclidean ADC similarity is non-negative, got " + scorer.score());
            assertFalse(Float.isNaN(scorer.score()), "score must not be NaN");
        }
    }

    /**
     * The quantizer centres its input in place, and one query array is shared across every cluster a scan visits — so
     * preparing against this cluster must not disturb it. If it did, the damage would show as lost recall on the
     * clusters visited afterwards rather than as a failure here.
     */
    @Test
    void testPrepareScan_thenLeavesTheCallersQueryUntouched() throws IOException {
        // given
        float[] query = query();
        float[] before = query.clone();

        // when
        cluster().prepareScan(ScanParams.of(query));

        // then
        assertArrayEquals(before, query, "the query is shared across clusters; centring it in place would corrupt it");
    }

    /** The context carries the query it was prepared from, and a transposed buffer the kernels can read in full. */
    @Test
    void testPrepareScan_thenSizesTheTransposedQueryFromTheEncoding() throws IOException {
        // given
        float[] query = query();

        // when
        SQScanContext context = (SQScanContext) cluster().prepareScan(ScanParams.of(query));

        // then
        assertSame(query, context.query(), "the context reports the query it was prepared from");
        assertEquals(ENCODING.getQueryPackedLength(DIMENSION), context.transposed().length);
        assertEquals(ENCODING.getQueryBitsPerDim(), context.queryBitsPerDimension());
    }

    /**
     * The kernels are written for one query width and the transposition packs for it, so a different width would be
     * scored against a buffer laid out for another. {@link ScanParams} no longer constrains it, which is why this is
     * checked here.
     */
    @ParameterizedTest(name = "queryBits {0}")
    @ValueSource(ints = { 1, 2, 3, 8 })
    void testPrepareScan_whenQueryWidthIsNotTheEncodingsOwn_thenThrows(int queryBits) throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();
        ScanParams params = new ScanParams(query(), queryBits);

        // when
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> cluster.prepareScan(params));

        // then
        assertTrue(e.getMessage().contains("got: " + queryBits), e.getMessage());
    }

    /** Centring is relative to this cluster's centroid, which does not change — so it is read once, not per query. */
    @Test
    void testPrepareScan_whenPreparedRepeatedly_thenReadsTheCentroidOnce() throws IOException {
        // given
        CountingCentroid centroidSupplier = new CountingCentroid();
        ScalarQuantizedCluster cluster = cluster(countingInput(), centroidSupplier);

        // when
        cluster.prepareScan(ScanParams.of(query()));
        cluster.prepareScan(ScanParams.of(query()));

        // then
        assertEquals(1, centroidSupplier.calls, "the centroid is the same for every query against this cluster");
    }

    /** Two queries prepare to two contexts; nothing is cached between them. */
    @Test
    void testPrepareScan_thenPreparesEachQuerySeparately() throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();

        // when
        ScanContext first = cluster.prepareScan(ScanParams.of(query()));
        ScanContext second = cluster.prepareScan(ScanParams.of(query()));

        // then
        assertNotSame(first, second);
    }

    // ---------------------------------------------------------------- deferred reads

    /**
     * A scan obtains every cluster it might visit and scans only the nearest few, so obtaining one must cost
     * nothing at all — not the posting header, and not the centroid. Hinting does not count as scanning.
     */
    @Test
    void testConstructor_thenReadsNothing() throws IOException {
        // given
        CountingCentroid centroidSupplier = new CountingCentroid();
        CountingInput input = countingInput();

        // when
        ScalarQuantizedCluster cluster = cluster(input, centroidSupplier);
        cluster.prefetch(true);
        cluster.prefetch(false);

        // then
        assertEquals(0, input.reads(), "obtaining and hinting a cluster must not read its posting");
        assertEquals(0, centroidSupplier.calls, "nor its centroid");
        assertEquals(CENTROID_ORDINAL, cluster.ordinal(), "identity comes from the caller, not from the file");
        assertEquals(CLUSTER_SIZE, cluster.size());
    }

    /** Header and centroid are the same for every scan of one cluster, so both are paid for once. */
    @Test
    void testScorer_whenScannedRepeatedly_thenReadsTheHeaderAndCentroidOnce() throws IOException {
        // given
        CountingCentroid centroidSupplier = new CountingCentroid();
        CountingInput input = countingInput();
        ScalarQuantizedCluster cluster = cluster(input, centroidSupplier);

        // when
        drain(cluster.scorer(scanContext(), null));
        int headerReads = input.reads();
        drain(cluster.scorer(scanContext(), null));

        // then
        assertTrue(headerReads > 0, "the first scan has to read the header");
        assertEquals(headerReads, input.reads(), "a later scan must not read it again");
        assertEquals(1, centroidSupplier.calls);
    }

    // ---------------------------------------------------------------- accounting

    /**
     * A cluster's footprint depends on whether it was scanned: unscanned it is the reader's buffers, and a scan
     * adds the ordinals and the centroid. Reporting the scanned figure for an unscanned cluster would overstate a
     * scan that holds one of these per cluster it merely considered.
     */
    @Test
    void testRamBytesUsed_thenGrowsOnlyWhenTheClusterIsScanned() throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();
        long beforeScan = cluster.ramBytesUsed();

        // when
        drain(cluster.scorer(scanContext(), null));

        // then
        long afterScan = cluster.ramBytesUsed();
        assertTrue(beforeScan > 0, "the reader's buffers exist from the start");
        assertTrue(afterScan > beforeScan, "a scan adds the ordinals and the centroid");
        long ordinalsAndCentroid = (long) CLUSTER_SIZE * Integer.BYTES + (long) DIMENSION * Float.BYTES;
        assertTrue(afterScan - beforeScan >= ordinalsAndCentroid, "both must be counted, not just one");
    }

    // ---------------------------------------------------------------- prefetch

    @ParameterizedTest(name = "partial={0}")
    @ValueSource(booleans = { true, false })
    void testPrefetch_thenHintsWithoutReading(boolean partial) throws IOException {
        // given
        ScalarQuantizedCluster cluster = cluster();

        // when — a hint is advisory, so the assertion is that it is accepted and changes nothing
        cluster.prefetch(partial);

        // then
        assertArrayEquals(ORDINALS, drain(cluster.scorer(scanContext(), null)).stream().mapToInt(Integer::intValue).toArray());
    }

    // ---------------------------------------------------------------- helpers

    private static float expectedDistance(int position) {
        return 0.5f + position * 0.25f;
    }

    /**
     * A prepared query, built here rather than through {@code prepareScan}, which is still a stub. The values are
     * arbitrary but fixed: these tests are about which entries a scan visits, not what it scores them.
     */
    /** A unit query, which is what the quantizer asks for on the similarities that check. */
    private static float[] query() {
        float[] query = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            query[i] = (i % 7) * 0.25f - 0.5f;
        }
        normalise(query);
        return query;
    }

    private static SQScanContext scanContext() {
        float[] query = query();

        // The kernels read four query stripes per doc byte, so this length is what keeps them in bounds.
        byte[] transposed = new byte[ENCODING.getQueryPackedLength(DIMENSION)];
        for (int i = 0; i < transposed.length; i++) {
            transposed[i] = (byte) (0x33 + i);
        }
        return new SQScanContext(query, 4, 1.0f, transposed, -0.75f, 0.05f, 42f, 2.5f);
    }

    /**
     * Counts reads of the posting itself, so "no IO in the constructor" is asserted rather than assumed. Slices
     * are not wrapped, so block reads made through the blocks slice do not count — only header reads do.
     */
    private static final class CountingInput extends FilterIndexInput {

        /** Shared with every clone, since the cluster reads the header through one. */
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
        public void readInts(int[] ints, int offset, int length) throws IOException {
            reads[0]++;
            in.readInts(ints, offset, length);
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

    /** Counts how often the centroid is actually read, so laziness and reuse can be asserted rather than assumed. */
    private static final class CountingCentroid implements IOSupplier<Centroid> {

        private int calls;

        @Override
        public Centroid get() {
            calls++;
            return new Centroid(centroid(), 1.0f);
        }
    }

    private static float[] centroid() {
        float[] centroid = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            centroid[i] = (i % 3) * 0.1f + 0.05f;
        }
        normalise(centroid);
        return centroid;
    }

    private static void normalise(float[] vector) {
        double sumOfSquares = 0;
        for (float value : vector) {
            sumOfSquares += (double) value * value;
        }
        float norm = (float) Math.sqrt(sumOfSquares);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= norm;
        }
    }

    private static List<Integer> drain(PostingScorer scorer) throws IOException {
        List<Integer> ords = new ArrayList<>();
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            ords.add(scorer.ord());
            if (ords.size() > 50) {
                throw new AssertionError("advance() never returned false");
            }
        }
        return ords;
    }

    private CountingInput countingInput() throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        writePosting(directory);
        return new CountingInput(directory.openInput(FILE, IOContext.DEFAULT), new int[1]);
    }

    private ScalarQuantizedCluster cluster() throws IOException {
        return cluster(countingInput(), new CountingCentroid());
    }

    private static ScalarQuantizedCluster cluster(CountingInput posting, CountingCentroid centroidSupplier) throws IOException {
        return new ScalarQuantizedCluster(
            posting,
            CENTROID_ORDINAL,
            CLUSTER_SIZE,
            centroidSupplier,
            BLOCK_SIZE,
            DIMENSION,
            ENCODING,
            new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN),
            VectorSimilarityFunction.EUCLIDEAN
        );
    }

    /** The posting layout: ordinals, packed SOAR bits, ascending distances, then the unpadded blocks. */
    private static void writePosting(Directory directory) throws IOException {
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int ordinal : ORDINALS) {
                out.writeInt(ordinal);
            }

            byte[] soar = new byte[(CLUSTER_SIZE + 7) / 8];
            for (int position : SOAR_POSITIONS) {
                soar[position >> 3] |= (byte) (1 << (position & 7));
            }
            out.writeBytes(soar, 0, soar.length);

            for (int position = 0; position < CLUSTER_SIZE; position++) {
                out.writeInt(Float.floatToIntBits(expectedDistance(position)));
            }

            for (int first = 0; first < CLUSTER_SIZE; first += BLOCK_SIZE) {
                int count = Math.min(BLOCK_SIZE, CLUSTER_SIZE - first);
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(-1.0f - (first + i) * 0.1f));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits(1.0f + (first + i) * 0.1f));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(Float.floatToIntBits((first + i) * 0.05f));
                }
                for (int i = 0; i < count; i++) {
                    out.writeInt(first + i);
                }
                for (int i = 0; i < count; i++) {
                    for (int b = 0; b < PACKED_BYTES; b++) {
                        out.writeByte((byte) (0x5A + first + i + b));
                    }
                }
            }
        }
    }
}
