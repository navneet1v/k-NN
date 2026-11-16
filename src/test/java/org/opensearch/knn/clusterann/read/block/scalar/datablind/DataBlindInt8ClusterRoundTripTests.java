/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;
import org.opensearch.knn.clusterann.write.block.scalar.DataBlindInt8ClusterWriter;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Round-trip unit tests for the data-blind int8 read path: a real posting is written with
 * {@link DataBlindInt8ClusterWriter} and read back through {@link DataBlindInt8Cluster}, which exercises
 * {@link DataBlindInt8BlockReader}, {@link DataBlindInt8BlockScorer} and {@link DataBlindInt8ScanContext}
 * together. The write path is the one the production writer uses, so the on-disk layout the reader parses is
 * never hand-fabricated here.
 */
class DataBlindInt8ClusterRoundTripTests {

    private static final int DIMENSION = 16;
    private static final int BLOCK_SIZE = 4;
    private static final int CLUSTER_SIZE = 10;
    private static final int CENTROID_ORDINAL = 3;
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
    private static final String FILE = "posting";

    /** Global ordinals, ascending by distance from the centroid rather than by ordinal. */
    private static final int[] ORDINALS = { 40, 3, 91, 12, 55, 7, 88, 21, 64, 30 };

    /** The second cluster's ordinals. Seven of them, so its block count and partial tail differ from the first's. */
    private static final int[] OTHER_ORDINALS = { 5, 71, 18, 44, 2, 99, 63 };

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (final Directory directory : directories) {
            directory.close();
        }
    }

    @Test
    void testConstructor_thenReportsItsIdentityAndSizeWithoutReadingBlocks() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        assertEquals(CENTROID_ORDINAL, cluster.ordinal());
        assertEquals(CLUSTER_SIZE, cluster.size());
    }

    @Test
    void testPrepareScan_thenTheContextReportsItIsQueryScoped() throws IOException {
        final ScanContext context = cluster().prepareScan(ScanParams.of(query()));

        assertTrue(context.isQueryScoped(), "a centroid-free scheme produces a context valid for every cluster of the scan");
    }

    @Test
    void testPrepareScanThenScorer_whenNothingIsFiltered_thenVisitsEveryEntryInPostingOrder() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        final ScanContext context = cluster.prepareScan(ScanParams.of(query()));
        final List<Integer> visited = drain(cluster.scorer(context, null, buffers()));

        assertArrayEquals(
            ORDINALS,
            visited.stream().mapToInt(Integer::intValue).toArray(),
            "every entry must be scored, in the order the posting stores it"
        );
    }

    @Test
    void testScorer_whenOrdinalsAreFiltered_thenVisitsOnlyThoseAccepted() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();
        final FixedBitSet accepted = new FixedBitSet(128);
        accepted.set(ORDINALS[0]);
        accepted.set(ORDINALS[5]);
        accepted.set(ORDINALS[9]);

        final List<Integer> visited = drain(cluster.scorer(cluster.prepareScan(ScanParams.of(query())), accepted, buffers()));

        assertArrayEquals(new int[] { ORDINALS[0], ORDINALS[5], ORDINALS[9] }, visited.stream().mapToInt(Integer::intValue).toArray());
    }

    @Test
    void testScorer_thenEveryScoreIsAFiniteSimilarity() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        final PostingScorer scorer = cluster.scorer(cluster.prepareScan(ScanParams.of(query())), null, buffers());

        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            assertTrue(Float.isFinite(scorer.score()), "score must be finite, got " + scorer.score());
            assertTrue(scorer.score() >= 0f, "a normalised MIP similarity is non-negative, got " + scorer.score());
        }
    }

    @Test
    void testScorer_whenCalledTwice_thenEachWalksTheWholePosting() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        final List<Integer> first = drain(cluster.scorer(cluster.prepareScan(ScanParams.of(query())), null, buffers()));
        final List<Integer> second = drain(cluster.scorer(cluster.prepareScan(ScanParams.of(query())), null, buffers()));

        assertEquals(CLUSTER_SIZE, first.size());
        assertEquals(first, second, "a fresh scorer starts at the beginning of the posting");
    }

    @Test
    void testPrepareScan_whenQueryDimensionIsWrong_thenThrows() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        final IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> cluster.prepareScan(ScanParams.of(new float[DIMENSION + 1]))
        );
        assertTrue(e.getMessage().contains("dimensional"), e.getMessage());
    }

    @Test
    void testPrepareScan_whenScoringItsOwnStoredVector_thenThatVectorRanksHighest() throws IOException {
        // The query is member 0's own vector: with near-exact 8-bit codes it must score at least as high as every
        // other member, so the absolute estimate orders correctly.
        final float[][] members = members();
        final DataBlindInt8Cluster cluster = cluster();

        final PostingScorer scorer = cluster.scorer(cluster.prepareScan(ScanParams.of(members[0].clone())), null, buffers());

        float selfScore = Float.NaN;
        float best = Float.NEGATIVE_INFINITY;
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            final float score = scorer.score();
            best = Math.max(best, score);
            if (scorer.ord() == ORDINALS[0]) {
                selfScore = score;
            }
        }
        assertFalse(Float.isNaN(selfScore), "member 0 must have been scored");
        assertEquals(best, selfScore, 1e-3f, "a member scored against itself should rank highest");
    }

    /**
     * Everything a scan fills belongs to the scan's buffers, so a scanned cluster is no heavier than an unscanned
     * one. Counting the buffers per cluster instead would report one scan's arrays once per probed cluster.
     */
    @Test
    void testRamBytesUsed_whenTheClusterIsScanned_thenDoesNotGrow() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();
        final long beforeScan = cluster.ramBytesUsed();

        drain(cluster.scorer(cluster.prepareScan(ScanParams.of(query())), null, buffers()));

        assertTrue(beforeScan > 0, "a cluster has a shallow footprint of its own");
        assertEquals(
            beforeScan,
            cluster.ramBytesUsed(),
            "the ordinals and the block buffers are the scan's, so scanning must not add to the cluster"
        );
    }

    /**
     * The ordinals buffer is sized for the field's largest cluster when the scan's buffers are made and is final
     * thereafter, so scanning cannot reallocate it whatever order it meets cluster sizes in.
     */
    @Test
    void testScanBuffers_whenClustersOfDifferentSizesAreScanned_thenTheOrdinalsBufferNeverGrows() throws IOException {
        final DataBlindInt8ScanBuffers shared = buffers();
        final long atConstruction = shared.ramBytesUsed();

        assertTrue(
            atConstruction >= (long) CLUSTER_SIZE * Integer.BYTES,
            "the ordinals are accounted for from the start, not once a cluster has been read"
        );

        // Smaller first, then larger: growing into the largest would show up as a bigger footprint.
        final DataBlindInt8Cluster smaller = otherCluster();
        final DataBlindInt8Cluster larger = cluster();
        final ScanContext context = larger.prepareScan(ScanParams.of(query()));
        drain(smaller.scorer(context, null, shared));
        drain(larger.scorer(context, null, shared));

        assertEquals(
            atConstruction,
            shared.ramBytesUsed(),
            "a scan must not reallocate the ordinals buffer, whatever order it meets cluster sizes in"
        );
    }

    @Test
    void testScorer_whenGivenAForeignScanContext_thenThrows() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        // A context from another cluster family must be rejected rather than misread.
        final ScanContext foreign = new ScanContext() {
        };
        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> cluster.scorer(foreign, null, buffers()));
        assertTrue(e.getMessage().contains("data-blind int8 scan context"), e.getMessage());
    }

    @Test
    void testScorer_whenGivenANullScanContext_thenThrows() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();

        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> cluster.scorer(null, null, buffers()));
        assertTrue(e.getMessage().contains("null"), e.getMessage());
    }

    @Test
    void testScorer_underEuclidean_thenEveryScoreIsAFiniteSimilarity() throws IOException {
        // Exercises the Euclidean score transform, the counterpart to the fixture's inner-product metric.
        final DataBlindInt8Cluster cluster = cluster(VectorSimilarityFunction.EUCLIDEAN);

        final PostingScorer scorer = cluster.scorer(cluster.prepareScan(ScanParams.of(query())), null, buffers());

        int scored = 0;
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            assertTrue(Float.isFinite(scorer.score()), "score must be finite, got " + scorer.score());
            assertTrue(scorer.score() >= 0f, "a euclidean similarity is non-negative, got " + scorer.score());
            scored++;
        }
        assertEquals(CLUSTER_SIZE, scored, "every entry in the posting has to be scored");
    }

    @Test
    void testScorer_whenOneSetOfBuffersIsReusedAcrossClusters_thenEachScoresAsIfScannedAlone() throws IOException {
        final DataBlindInt8Cluster first = cluster();
        final DataBlindInt8Cluster second = otherCluster();

        // One set of buffers, so both clusters go through the same block reader; the second resets it onto its own.
        final ScanContext context = first.prepareScan(ScanParams.of(query()));
        final ScanBuffers shared = buffers();
        final List<String> sharedFirst = drainScored(first.scorer(context, null, shared));
        final List<String> sharedSecond = drainScored(second.scorer(context, null, shared));

        // The same two scans, each with buffers of its own, which is what the shared ones have to reproduce.
        final List<String> aloneFirst = drainScored(cluster().scorer(context, null, buffers()));
        final List<String> aloneSecond = drainScored(otherCluster().scorer(context, null, buffers()));

        assertEquals(aloneFirst, sharedFirst, "reusing the buffers must not change the first cluster's scores");
        assertEquals(aloneSecond, sharedSecond, "the second cluster must be read from its own blocks, not from what the first left behind");
    }

    /**
     * The filtered path looks ahead through the ordinals array, which is the scan's buffer and so is oversized for
     * every cluster but the largest. The larger cluster is scanned first to grow it, and the smaller must still
     * report only its own entries — the stale tail beyond its size must not reach the caller.
     */
    @Test
    void testScorer_whenASmallerClusterFollowsALargerOne_thenTheFilteredWalkStopsAtItsOwnEnd() throws IOException {
        final DataBlindInt8Cluster larger = cluster();                 // CLUSTER_SIZE entries
        final DataBlindInt8Cluster smaller = otherCluster();           // fewer, so the buffer is oversized for it
        final ScanBuffers shared = buffers();
        final ScanContext context = larger.prepareScan(ScanParams.of(query()));

        final FixedBitSet accepted = new FixedBitSet(128);
        accepted.set(OTHER_ORDINALS[0]);
        accepted.set(OTHER_ORDINALS[OTHER_ORDINALS.length - 1]);

        drain(larger.scorer(context, null, shared));                   // grows the buffer to the larger size
        final List<Integer> visited = drain(smaller.scorer(context, accepted, shared));

        assertArrayEquals(
            new int[] { OTHER_ORDINALS[0], OTHER_ORDINALS[OTHER_ORDINALS.length - 1] },
            visited.stream().mapToInt(Integer::intValue).toArray(),
            "only the smaller cluster's own accepted ordinals may be visited"
        );
    }

    @Test
    void testScorer_whenTheScanVisitsTwoClusters_thenOneSetOfBlockBuffersServesBoth() throws IOException {
        final DataBlindInt8Cluster first = cluster();
        final DataBlindInt8Cluster second = otherCluster();

        final ScanContext context = first.prepareScan(ScanParams.of(query()));
        final DataBlindInt8ScanBuffers shared = buffers();
        final byte[] codes = shared.reader().codes();
        drain(first.scorer(context, null, shared));
        drain(second.scorer(context, null, shared));

        assertSame(
            codes,
            shared.reader().codes(),
            "a probe must not allocate a block's worth of buffers; the scan's reader is reset onto it"
        );
    }

    @Test
    void testScorer_whenAnEarlierScorerIsLeftUndrained_thenItRefusesToScoreTheWrongCluster() throws IOException {
        final DataBlindInt8Cluster first = cluster();
        final DataBlindInt8Cluster second = otherCluster();
        final ScanContext context = first.prepareScan(ScanParams.of(query()));
        final ScanBuffers shared = buffers();

        final PostingScorer stale = first.scorer(context, null, shared);
        second.scorer(context, null, shared); // moves the shared reader off the first cluster

        final IllegalStateException e = assertThrows(IllegalStateException.class, () -> stale.advance(Float.NEGATIVE_INFINITY));
        assertTrue(e.getMessage().contains("another cluster"), e.getMessage());
    }

    @Test
    void testScorer_whenGivenAnotherFamilysBuffers_thenRejectsThemRatherThanMisreadThem() throws IOException {
        final DataBlindInt8Cluster cluster = cluster();
        final ScanContext context = cluster.prepareScan(ScanParams.of(query()));

        final IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> cluster.scorer(context, null, new ScanBuffers() {
            })
        );
        assertTrue(e.getMessage().contains("scan buffers"), e.getMessage());
    }

    @Test
    void testConstructor_whenThePostingIsTooShortForItsHeader_thenThrows() throws IOException {
        final Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        writePosting(directory, ORDINALS, members());
        final IndexInput posting = directory.openInput(FILE, IOContext.DEFAULT);

        // A header for twice the entries cannot fit in this posting, and saying so must not wait for a scan.
        final IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> new DataBlindInt8Cluster(
                posting,
                CENTROID_ORDINAL,
                CLUSTER_SIZE * 1000,
                DIMENSION,
                METRIC,
                JavaInt8ScoreComputer.INSTANCE
            )
        );
        assertTrue(e.getMessage().contains("too short"), e.getMessage());
    }

    @Test
    void testClusterOrdinals_whenAClusterExceedsTheFieldsLargest_thenThrows() {
        // The buffer is sized from the metadata and never grows, so a cluster bigger than the metadata promised is
        // a contradiction worth naming rather than an overrun inside the caller's read.
        final DataBlindInt8ScanBuffers buffers = new DataBlindInt8ScanBuffers(BLOCK_SIZE, DIMENSION, CLUSTER_SIZE);

        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> buffers.clusterOrdinals(CLUSTER_SIZE + 1));
        assertTrue(e.getMessage().contains("largest holds"), e.getMessage());
    }

    @Test
    void testReader_whenNotResetOntoACluster_thenRefusesToRead() {
        final DataBlindInt8BlockReader reader = new DataBlindInt8BlockReader(BLOCK_SIZE, DIMENSION);

        final IllegalStateException e = assertThrows(IllegalStateException.class, () -> reader.advance(0));
        assertTrue(e.getMessage().contains("reset"), e.getMessage());
    }

    // ---------------------------------------------------------------- helpers

    /** Buffers of this field's geometry, as {@code ClusterFactory#newScanBuffers()} makes in production. */
    private static DataBlindInt8ScanBuffers buffers() {
        return new DataBlindInt8ScanBuffers(BLOCK_SIZE, DIMENSION, Math.max(CLUSTER_SIZE, OTHER_ORDINALS.length));
    }

    /** A query of the fixture dimension; values arbitrary but fixed. */
    private static float[] query() {
        final float[] query = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            query[i] = (i % 5) * 0.3f - 0.6f;
        }
        return query;
    }

    /** The member vectors, one per ordinal; deterministic so scores are reproducible. */
    private static float[][] members() {
        final float[][] vectors = new float[CLUSTER_SIZE][DIMENSION];
        for (int m = 0; m < CLUSTER_SIZE; m++) {
            for (int i = 0; i < DIMENSION; i++) {
                vectors[m][i] = ((m + i) % 7) * 0.2f - 0.7f;
            }
        }
        return vectors;
    }

    private DataBlindInt8Cluster cluster() throws IOException {
        return cluster(METRIC);
    }

    private DataBlindInt8Cluster cluster(final VectorSimilarityFunction metric) throws IOException {
        final Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        writePosting(directory, ORDINALS, members());
        final IndexInput posting = directory.openInput(FILE, IOContext.DEFAULT);
        return new DataBlindInt8Cluster(posting, CENTROID_ORDINAL, CLUSTER_SIZE, DIMENSION, metric, JavaInt8ScoreComputer.INSTANCE);
    }

    /**
     * A second cluster, deliberately a different size and content from {@link #cluster()}. The size matters: at
     * {@link #BLOCK_SIZE} it gives a different block count and a different partial tail, so a reader reset onto it
     * has to re-derive its geometry rather than carry the first cluster's.
     */
    private DataBlindInt8Cluster otherCluster() throws IOException {
        final Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        writePosting(directory, OTHER_ORDINALS, otherMembers());
        final IndexInput posting = directory.openInput(FILE, IOContext.DEFAULT);
        return new DataBlindInt8Cluster(
            posting,
            CENTROID_ORDINAL + 1,
            OTHER_ORDINALS.length,
            DIMENSION,
            METRIC,
            JavaInt8ScoreComputer.INSTANCE
        );
    }

    /** Members of the second cluster; different values, so a buffer carried over from the first would show up. */
    private static float[][] otherMembers() {
        final float[][] vectors = new float[OTHER_ORDINALS.length][DIMENSION];
        for (int m = 0; m < OTHER_ORDINALS.length; m++) {
            for (int i = 0; i < DIMENSION; i++) {
                vectors[m][i] = ((m * 3 + i) % 11) * 0.17f - 0.9f;
            }
        }
        return vectors;
    }

    /** Writes a real cluster posting with the production writer, so the reader parses a genuine layout. */
    private static void writePosting(final Directory directory, final int[] ordinals, final float[][] members) throws IOException {
        final int size = ordinals.length;
        final float[] distances = new float[size];
        final boolean[] secondary = new boolean[size];
        for (int i = 0; i < size; i++) {
            distances[i] = 0.5f + i * 0.25f; // ascending, as the writer expects
        }
        for (int i = 1; i < size; i += 3) {
            secondary[i] = true;
        }

        // storageBlockBytes chosen so vectorsPerBlock == BLOCK_SIZE for this dimension.
        final int storageBlockBytes = BLOCK_SIZE * DataBlindInt8ClusterWriter.bytesPerVector(DIMENSION);
        final DataBlindInt8ClusterWriter writer = new DataBlindInt8ClusterWriter(storageBlockBytes, DIMENSION);

        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            writer.write(
                out,
                new ArrayFloatVectorValues(members),
                new ClusterWriter.ClusterMembers(ordinals.clone(), distances, secondary)
            );
        }
    }

    /** Ordinals and exact score bits, so a buffer carried over from another cluster cannot pass unnoticed. */
    private static List<String> drainScored(final PostingScorer scorer) throws IOException {
        final List<String> scored = new ArrayList<>();
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            scored.add(scorer.ord() + "=" + Float.floatToIntBits(scorer.score()));
            if (scored.size() > 50) {
                throw new AssertionError("advance() never returned false");
            }
        }
        return scored;
    }

    private static List<Integer> drain(final PostingScorer scorer) throws IOException {
        final List<Integer> ords = new ArrayList<>();
        while (scorer.advance(Float.NEGATIVE_INFINITY)) {
            ords.add(scorer.ord());
            if (ords.size() > 50) {
                throw new AssertionError("advance() never returned false");
            }
        }
        return ords;
    }

    /** Minimal in-memory {@link FloatVectorValues} over a fixed array, read by position. */
    private static final class ArrayFloatVectorValues extends FloatVectorValues {

        private final float[][] vectors;

        private ArrayFloatVectorValues(final float[][] vectors) {
            this.vectors = vectors;
        }

        @Override
        public float[] vectorValue(final int ord) {
            return vectors[ord];
        }

        @Override
        public int dimension() {
            return DIMENSION;
        }

        @Override
        public int size() {
            return vectors.length;
        }

        @Override
        public ArrayFloatVectorValues copy() {
            return this;
        }
    }
}
