/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.format.block.BlockPostingsPruner;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class BlockPostingScorerTests {

    private static final int BLOCK_SIZE = 4;

    /** No score is below this, so nothing is pruned. */
    private static final float ACCEPT_ALL = Float.NEGATIVE_INFINITY;

    /** 12 positions over 3 full blocks of 4. Shorter sequences are prefixes of this. */
    private static final int[] ORDINALS = { 30, 31, 12, 13, 40, 41, 22, 23, 50, 51, 60, 61 };

    /** Score per position. Block maxima are 0.40, 0.85 and 0.45 — distinct, so pruning is easy to aim. */
    private static final float[] SCORES = { 0.10f, 0.20f, 0.30f, 0.40f, 0.55f, 0.65f, 0.75f, 0.85f, 0.05f, 0.15f, 0.45f, 0.35f };

    /** Sequence lengths worth covering: partial-only, exact fits, and both sides of a block boundary. */
    @ParameterizedTest(name = "{0} vectors")
    @ValueSource(ints = { 1, 3, 4, 5, 8, 10, 12 })
    void testAdvance_whenNothingIsFilteredOrPruned_thenVisitsEveryPosition(int vectorCount) throws IOException {
        // given
        int[] ordinals = Arrays.copyOf(ORDINALS, vectorCount);
        float[] scores = Arrays.copyOf(SCORES, vectorCount);
        Scan scan = scanOver(ordinals, scores, null);

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        List<Integer> everyBlock = blocksUpTo(vectorCount);
        List<Hit> expectedHits = hitsAt(ordinals, scores, positionsUpTo(vectorCount));
        RecordingBlockReader expectedReader = readerThatSaw(vectorCount).advanced(everyBlock)
            .fetched(everyBlock)
            .read(everyBlock)
            .prefetched(blocksAfterFirst(vectorCount))
            .reader();

        assertEquals(expectedHits, hits);
        assertEquals(expectedReader, scan.reader);
    }

    /**
     * Each case gives the accepted positions, then the blocks the walk is expected to touch and the ones it is
     * expected to hint.
     *
     * <p>A block with nothing accepted is pruned before any IO, so it is never even positioned on — the walk
     * only ever advances to a block it then fetches, and a hint only ever names such a block.
     */
    private static Stream<Arguments> filters() {
        return Stream.of(
            Arguments.of("one whole block", new int[] { 4, 5, 6, 7 }, List.of(1), List.of()),
            Arguments.of("one position per block, each at a different offset", new int[] { 2, 5, 11 }, List.of(0, 1, 2), List.of(1, 2)),
            Arguments.of("only the first and last positions", new int[] { 0, 11 }, List.of(0, 2), List.of(2)),
            Arguments.of("nothing at all", new int[] {}, List.of(), List.of())
        );
    }

    /** A block with no accepted ordinal must be skipped whole — not filtered after scoring, and never hinted. */
    @ParameterizedTest(name = "accepting {0}")
    @MethodSource("filters")
    void testAdvance_whenOrdinalsAreFiltered_thenTouchesOnlyBlocksWithAnAcceptedOrdinal(
        String description,
        int[] acceptedPositions,
        List<Integer> expectedBlocks,
        List<Integer> expectedPrefetches
    ) throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, bitsAt(acceptedPositions));

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        List<Hit> expectedHits = hitsAt(ORDINALS, SCORES, acceptedPositions);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(expectedBlocks)
            .fetched(expectedBlocks)
            .read(expectedBlocks)
            .prefetched(expectedPrefetches)
            .reader();

        assertEquals(expectedHits, hits);
        assertEquals(expectedReader, scan.reader);
    }

    private static Stream<Arguments> thresholds() {
        return Stream.of(
            // Block maxima are 0.40, 0.85, 0.45; a surviving block yields all of its positions.
            Arguments.of(ACCEPT_ALL, positionsUpTo(12)),
            Arguments.of(0.42f, new int[] { 4, 5, 6, 7, 8, 9, 10, 11 }),
            Arguments.of(0.50f, new int[] { 4, 5, 6, 7 }),
            Arguments.of(0.90f, new int[] {})
        );
    }

    /** Pruning happens after scoring, so a dropped block still costs its load — only its hits are withheld. */
    @ParameterizedTest(name = "at minCompetitiveSimilarity {0}")
    @MethodSource("thresholds")
    void testAdvance_whenABlockCannotCompete_thenWithholdsItsHits(float minCompetitiveSimilarity, int[] expectedPositions)
        throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, null);

        // when
        List<Hit> hits = drain(scan, minCompetitiveSimilarity);

        // then
        List<Integer> everyBlock = blocksUpTo(ORDINALS.length);
        List<Hit> expectedHits = hitsAt(ORDINALS, SCORES, expectedPositions);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(everyBlock)
            .fetched(everyBlock)
            .read(everyBlock)
            .prefetched(blocksAfterFirst(ORDINALS.length))
            .reader();

        assertEquals(expectedHits, hits);
        assertEquals(expectedReader, scan.reader);
    }

    /** A pruner's SKIP has to cost nothing: the block is not positioned on, not hinted, not fetched. */
    @Test
    void testAdvance_whenAPrunerSkipsABlock_thenNeverTouchesIt() throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, null, skipping(1));

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        List<Integer> touched = List.of(0, 2);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(touched)
            .fetched(touched)
            .read(touched)
            .prefetched(List.of(2))
            .reader();

        assertEquals(hitsAt(ORDINALS, SCORES, 0, 1, 2, 3, 8, 9, 10, 11), hits);
        assertEquals(expectedReader, scan.reader);
    }

    /**
     * A TERMINATE found while looking ahead ends the walk, but the block already scored is still drained — the
     * decision is about the blocks after it, not about the candidates in hand.
     */
    @Test
    void testAdvance_whenAPrunerTerminatesAhead_thenDrainsTheCurrentBlockAndStops() throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, null, terminatingAt(2));

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        List<Integer> touched = List.of(0, 1);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(touched)
            .fetched(touched)
            .read(touched)
            .prefetched(List.of(1))
            .reader();

        assertEquals(hitsAt(ORDINALS, SCORES, 0, 1, 2, 3, 4, 5, 6, 7), hits);
        assertEquals(expectedReader, scan.reader);
    }

    @Test
    void testAdvance_whenAPrunerTerminatesAtTheFirstBlock_thenReadsNothing() throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, null, terminatingAt(0));

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        assertEquals(List.of(), hits);
        assertEquals(readerThatSaw(ORDINALS.length).reader(), scan.reader);
    }

    /**
     * The threshold rises as hits are collected, so a lookahead taken at a lower one can go stale: block 2 is
     * accepted while the bar is still low, and by the time the walk reaches it block 1's hits have raised the bar
     * past its bound. The walk trusts the lookahead rather than testing a block twice, so block 2 is still
     * fetched and scored — its hits are withheld by the max-score check instead. That is the trade: every block
     * is tested exactly once, at the price of the occasional fetch a re-test would have avoided.
     */
    @Test
    void testAdvance_whenTheThresholdRisesPastALookahead_thenStillFetchesItAndWithholdsItsHits() throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, null, blockMaxPruner(0.40f, 0.85f, 0.45f));

        // when
        List<Hit> hits = drainAgainstBestSoFar(scan);

        // then
        List<Integer> everyBlock = blocksUpTo(ORDINALS.length);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(everyBlock)
            .fetched(everyBlock)
            .read(everyBlock)
            .prefetched(List.of(1, 2))
            .reader();

        assertEquals(hitsAt(ORDINALS, SCORES, 0, 1, 2, 3, 4, 5, 6, 7), hits);
        assertEquals(expectedReader, scan.reader);
    }

    @Test
    void testAdvance_whenAlreadyExhausted_thenKeepsReturningFalse() throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, null);
        drain(scan, ACCEPT_ALL);

        // when
        boolean firstCall = scan.scorer.advance(ACCEPT_ALL);
        boolean secondCall = scan.scorer.advance(ACCEPT_ALL);

        // then
        assertFalse(firstCall, "advance() must keep returning false after exhaustion");
        assertFalse(secondCall, "advance() must keep returning false after exhaustion");
    }

    // ---------------------------------------------------------------- what the walk cost

    /** With nothing to prune on, every block is cleared and read, and no decision is anything else. */
    @Test
    void testCounters_whenThereIsNoPruner_thenEveryBlockIsScored() throws IOException {
        // given
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null);

        // when
        drain(scan, ACCEPT_ALL);

        // then
        int blocks = numBlocks(ORDINALS.length);
        assertEquals((long) blocks, ClusterANNQueryValue.PRUNER_SCORE.getValue());
        assertEquals((long) blocks, ClusterANNQueryValue.BLOCKS_FETCHED.getValue());
        assertEquals(0L, ClusterANNQueryValue.PRUNER_SKIP.getValue());
        assertEquals(0L, ClusterANNQueryValue.PRUNER_TERMINATE.getValue());
        assertEquals(0L, ClusterANNQueryValue.BLOCKS_TERMINATED.getValue());
    }

    /** A SKIP is counted as a skip and not as a fetch — the whole point of the decision. */
    @Test
    void testCounters_whenAPrunerSkipsABlock_thenCountsTheSkipAndNotAFetch() throws IOException {
        // given
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, skipping(1));

        // when
        drain(scan, ACCEPT_ALL);

        // then — blocks 0 and 2 scored, block 1 skipped
        assertEquals(2L, ClusterANNQueryValue.PRUNER_SCORE.getValue());
        assertEquals(2L, ClusterANNQueryValue.BLOCKS_FETCHED.getValue());
        assertEquals(1L, ClusterANNQueryValue.PRUNER_SKIP.getValue());
        assertEquals(0L, ClusterANNQueryValue.PRUNER_TERMINATE.getValue());
    }

    /**
     * A TERMINATE is counted once, and the tail it bought is counted as the blocks from the terminating one to the
     * end — those are never tested again, so this is the only chance to count them.
     */
    @Test
    void testCounters_whenAPrunerTerminates_thenCountsTheTailItAvoided() throws IOException {
        // given — 3 blocks of 4; terminate on reaching block 2, so block 2 is the tail
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, terminatingAt(2));

        // when
        drain(scan, ACCEPT_ALL);

        // then
        assertEquals(2L, ClusterANNQueryValue.BLOCKS_FETCHED.getValue(), "blocks 0 and 1 were read");
        assertEquals(1L, ClusterANNQueryValue.PRUNER_TERMINATE.getValue());
        assertEquals(1L, ClusterANNQueryValue.BLOCKS_TERMINATED.getValue(), "block 2 was never reached");
        assertEquals(0L, ClusterANNQueryValue.PRUNER_SKIP.getValue());
    }

    /** Terminating on the first block means the whole posting is the avoided tail, and nothing is read. */
    @Test
    void testCounters_whenAPrunerTerminatesAtTheFirstBlock_thenTheWholePostingIsTheTail() throws IOException {
        // given
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, terminatingAt(0));

        // when
        drain(scan, ACCEPT_ALL);

        // then
        assertEquals(0L, ClusterANNQueryValue.BLOCKS_FETCHED.getValue());
        assertEquals(1L, ClusterANNQueryValue.PRUNER_TERMINATE.getValue());
        assertEquals((long) numBlocks(ORDINALS.length), ClusterANNQueryValue.BLOCKS_TERMINATED.getValue());
    }

    /**
     * The invariant the counters are carried for: fetched, skipped and terminated partition every block of the
     * posting. Asserted over each pruner shape, since a decision counted in the wrong arm would still leave the
     * individual totals looking plausible.
     */
    @ParameterizedTest(name = "{0}")
    @MethodSource("prunerShapes")
    void testCounters_thenFetchedSkippedAndTerminatedAccountForEveryBlock(String description, BlockPostingsPruner pruner)
        throws IOException {
        // given
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, pruner);

        // when
        drain(scan, ACCEPT_ALL);

        // then
        long accounted = ClusterANNQueryValue.BLOCKS_FETCHED.getValue() + ClusterANNQueryValue.PRUNER_SKIP.getValue()
            + ClusterANNQueryValue.BLOCKS_TERMINATED.getValue();
        assertEquals((long) numBlocks(ORDINALS.length), accounted, "every block must be fetched, skipped or terminated past");
        assertEquals(
            ClusterANNQueryValue.BLOCKS_FETCHED.getValue(),
            ClusterANNQueryValue.PRUNER_SCORE.getValue(),
            "a cleared block is always fetched"
        );
    }

    private static Stream<Arguments> prunerShapes() {
        return Stream.of(
            Arguments.of("no pruner", BlockPostingsPruner.NONE),
            Arguments.of("skips the middle block", skipping(1)),
            Arguments.of("terminates partway", terminatingAt(2)),
            Arguments.of("terminates immediately", terminatingAt(0)),
            Arguments.of("block maxima", blockMaxPruner(0.40f, 0.85f, 0.45f))
        );
    }

    /** The counters are node-wide and never reset in production, so a test that asserts on them must start from zero. */
    private static void resetStats() {
        for (ClusterANNQueryValue value : ClusterANNQueryValue.values()) {
            value.set(0);
        }
    }

    // ---------------------------------------------------------------- the code-read gate

    /**
     * A block whose ceiling cannot reach the bar costs its corrections and nothing else: positioned on,
     * fetched, never read. That is the whole point of the gate — the codes are the bulk of a block.
     */
    @Test
    @Disabled("BlockPostingScorer.CODE_GATE_ENABLED is false on this branch; flip both together")
    void testAdvance_whenNoVectorInABlockCanReachTheBar_thenNeverReadsItsCodes() throws IOException {
        // given — every block's ceiling is below the bar the drain is run at
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, BlockPostingsPruner.NONE, 0.2f);

        // when
        List<Hit> hits = drain(scan, 0.5f);

        // then
        List<Integer> everyBlock = blocksUpTo(ORDINALS.length);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(everyBlock)
            .fetched(everyBlock)
            .prefetched(blocksAfterFirst(ORDINALS.length))
            .reader();

        assertEquals(List.of(), hits, "nothing could compete, so nothing is handed out");
        assertEquals(expectedReader, scan.reader, "every block fetched, none read");
        assertEquals((long) everyBlock.size(), ClusterANNQueryValue.CODE_READS_SKIPPED.getValue());
        assertEquals((long) everyBlock.size(), ClusterANNQueryValue.BLOCKS_FETCHED.getValue(), "a gated block still counts as fetched");
        assertEquals(0L, ClusterANNQueryValue.VECTORS_SCORED.getValue(), "a gated block scores nothing");
    }

    /** A ceiling above the bar decides nothing, so the walk reads and scores exactly as it would without one. */
    @Test
    @Disabled("BlockPostingScorer.CODE_GATE_ENABLED is false on this branch; flip both together")
    void testAdvance_whenTheCeilingClearsTheBar_thenReadsTheCodes() throws IOException {
        // given
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, BlockPostingsPruner.NONE, 1.0f);

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        assertEquals(hitsAt(ORDINALS, SCORES, positionsUpTo(ORDINALS.length)), hits);
        assertEquals(0L, ClusterANNQueryValue.CODE_READS_SKIPPED.getValue());
        assertEquals((long) ORDINALS.length, ClusterANNQueryValue.VECTORS_SCORED.getValue());
    }

    /**
     * The gate is a ceiling, so equality gates: the scan keeps a block only on a score strictly above the bar,
     * and a block whose best possible score merely ties it cannot produce one.
     */
    @Test
    @Disabled("BlockPostingScorer.CODE_GATE_ENABLED is false on this branch; flip both together")
    void testAdvance_whenTheCeilingExactlyTiesTheBar_thenStillSkipsTheCodes() throws IOException {
        // given
        resetStats();
        Scan scan = scanOver(ORDINALS, SCORES, null, BlockPostingsPruner.NONE, 0.5f);

        // when
        drain(scan, 0.5f);

        // then
        assertEquals((long) numBlocks(ORDINALS.length), ClusterANNQueryValue.CODE_READS_SKIPPED.getValue());
    }

    // ---------------------------------------------------------------- helpers

    private record Hit(int ord, float score) {
    }

    /** A scorer plus the reader it drives, so a test can assert on the results and on the I/O they cost. */
    private record Scan(PostingScorer scorer, RecordingBlockReader reader) {
    }

    private static Scan scanOver(int[] ordinals, float[] scores, Bits acceptedOrds) {
        return scanOver(ordinals, scores, acceptedOrds, BlockPostingsPruner.NONE);
    }

    private static Scan scanOver(int[] ordinals, float[] scores, Bits acceptedOrds, BlockPostingsPruner pruner) {
        return scanOver(ordinals, scores, acceptedOrds, pruner, Float.POSITIVE_INFINITY);
    }

    /** As above, with the scorer reporting {@code ceiling} for every block, to drive the code-read gate. */
    private static Scan scanOver(int[] ordinals, float[] scores, Bits acceptedOrds, BlockPostingsPruner pruner, float ceiling) {
        RecordingBlockReader reader = new RecordingBlockReader(ordinals.length, BLOCK_SIZE);
        TableScorer blockScorer = new TableScorer(reader, scores);
        blockScorer.ceiling = ceiling;
        return new Scan(new BlockPostingScorer(blockScorer, ordinals, acceptedOrds, pruner), reader);
    }

    /** Starts describing the reader a case expects to end up with: same geometry, and exactly these calls. */
    private static ExpectedReader readerThatSaw(int vectorCount) {
        return new ExpectedReader(new RecordingBlockReader(vectorCount, BLOCK_SIZE));
    }

    /**
     * Names each rung a case expects, so an expectation cannot be built by getting four positional lists in the
     * wrong order. Anything left unsaid is expected not to have happened at all.
     */
    private record ExpectedReader(RecordingBlockReader reader) {

        private ExpectedReader advanced(List<Integer> blocks) {
            reader.advances.addAll(blocks);
            return this;
        }

        private ExpectedReader fetched(List<Integer> blocks) {
            reader.fetches.addAll(blocks);
            return this;
        }

        private ExpectedReader read(List<Integer> blocks) {
            reader.reads.addAll(blocks);
            return this;
        }

        private ExpectedReader prefetched(List<Integer> blocks) {
            reader.prefetches.addAll(blocks);
            return this;
        }
    }

    /** Drives the scan to exhaustion, failing rather than hanging if it never says it is done. */
    private static List<Hit> drain(Scan scan, float minCompetitiveSimilarity) throws IOException {
        List<Hit> hits = new ArrayList<>();
        while (scan.scorer.advance(minCompetitiveSimilarity)) {
            hits.add(new Hit(scan.scorer.ord(), scan.scorer.score()));
            if (hits.size() > 50) {
                fail("advance() never returned false; got " + hits.size() + " hits from a sequence that cannot have that many");
            }
        }
        return hits;
    }

    /**
     * Drives the scan the way a top-1 collector would: every hit raises the threshold to the best score seen,
     * so pruning decisions are made against a threshold that moves.
     */
    private static List<Hit> drainAgainstBestSoFar(Scan scan) throws IOException {
        List<Hit> hits = new ArrayList<>();
        float best = ACCEPT_ALL;
        while (scan.scorer.advance(best)) {
            Hit hit = new Hit(scan.scorer.ord(), scan.scorer.score());
            hits.add(hit);
            best = Math.max(best, hit.score());
            if (hits.size() > 50) {
                fail("advance() never returned false; got " + hits.size() + " hits from a sequence that cannot have that many");
            }
        }
        return hits;
    }

    private static BlockPostingsPruner skipping(int skippedBlock) {
        return (block, minCompetitiveScore) -> block == skippedBlock
            ? BlockPostingsPruner.Decision.SKIP
            : BlockPostingsPruner.Decision.SCORE;
    }

    private static BlockPostingsPruner terminatingAt(int lastBlock) {
        return (block, minCompetitiveScore) -> block >= lastBlock
            ? BlockPostingsPruner.Decision.TERMINATE
            : BlockPostingsPruner.Decision.SCORE;
    }

    /** Stands in for a clip pruner: it knows each block's best possible score and skips the ones that can't win. */
    private static BlockPostingsPruner blockMaxPruner(float... blockMax) {
        return (block, minCompetitiveScore) -> minCompetitiveScore >= blockMax[block]
            ? BlockPostingsPruner.Decision.SKIP
            : BlockPostingsPruner.Decision.SCORE;
    }

    /** What a scan should report for {@code positions}, in position order. */
    private static List<Hit> hitsAt(int[] ordinals, float[] scores, int... positions) {
        List<Hit> hits = new ArrayList<>(positions.length);
        for (int pos : positions) {
            hits.add(new Hit(ordinals[pos], scores[pos]));
        }
        return hits;
    }

    private static int[] positionsUpTo(int vectorCount) {
        return IntStream.range(0, vectorCount).toArray();
    }

    private static int numBlocks(int vectorCount) {
        return (vectorCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    private static List<Integer> blocksUpTo(int vectorCount) {
        return IntStream.range(0, numBlocks(vectorCount)).boxed().toList();
    }

    /** Every block but the first — what a full walk is expected to hint, each one block ahead. */
    private static List<Integer> blocksAfterFirst(int vectorCount) {
        return IntStream.range(1, numBlocks(vectorCount)).boxed().toList();
    }

    /** The ordinals living at {@code positions}, as the filter a scan is given. */
    private static Bits bitsAt(int... positions) {
        FixedBitSet bits = new FixedBitSet(64);
        for (int pos : positions) {
            bits.set(ORDINALS[pos]);
        }
        return bits;
    }

    /** Cursor stand-in: owns the block geometry, remembers where it is, and records the calls the scan makes. */
    private static final class RecordingBlockReader implements BlockVectorFormat.Reader {
        private final List<Integer> advances = new ArrayList<>();
        private final List<Integer> fetches = new ArrayList<>();
        private final List<Integer> reads = new ArrayList<>();
        private final List<Integer> prefetches = new ArrayList<>();

        private final int vectorCount;
        private final int blockSize;

        private int currentBlock = -1;
        private boolean fetched;
        private boolean loaded;

        private RecordingBlockReader(int vectorCount, int blockSize) {
            this.vectorCount = vectorCount;
            this.blockSize = blockSize;
        }

        @Override
        public int numBlocks() {
            return (vectorCount + blockSize - 1) / blockSize;
        }

        @Override
        public int blockSize() {
            return blockSize;
        }

        @Override
        public boolean advance(int blockPos) {
            advances.add(blockPos);
            currentBlock = blockPos;
            fetched = false;
            loaded = false;
            return blockPos < numBlocks();
        }

        @Override
        public int blockVectorCount() {
            assertTrue(currentBlock >= 0, "blockVectorCount() before any advance()");
            return Math.min(blockSize, vectorCount - firstPosition());
        }

        /** Not part of {@link BlockVectorFormat.Reader} — the fake's own bookkeeping, for scoring the right rows. */
        private int firstPosition() {
            return currentBlock * blockSize;
        }

        @Override
        public void fetchBlock() {
            assertTrue(currentBlock >= 0, "fetchBlock() before any advance()");
            assertFalse(fetched, "fetchBlock() called twice for block " + currentBlock);
            fetches.add(currentBlock);
            fetched = true;
        }

        @Override
        public void readBlockVectors() {
            assertTrue(fetched, "readBlockVectors() without fetchBlock()");
            assertFalse(loaded, "readBlockVectors() called twice for block " + currentBlock);
            reads.add(currentBlock);
            loaded = true;
        }

        @Override
        public void prefetchBlock(int block) {
            prefetches.add(block);
        }

        /**
         * Equal when built over the same geometry and having seen the same calls, in the same order. Where the
         * cursor happens to have stopped ({@code currentBlock}, {@code fetched}, {@code loaded}) is left out:
         * that is an artifact of when the scan gave up, not something a case means to pin.
         */
        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof RecordingBlockReader other)) {
                return false;
            }
            return vectorCount == other.vectorCount
                && blockSize == other.blockSize
                && advances.equals(other.advances)
                && fetches.equals(other.fetches)
                && reads.equals(other.reads)
                && prefetches.equals(other.prefetches);
        }

        @Override
        public int hashCode() {
            return Objects.hash(vectorCount, blockSize, advances, fetches, reads, prefetches);
        }

        @Override
        public String toString() {
            return "reader[vectors="
                + vectorCount
                + ", blockSize="
                + blockSize
                + ", advances="
                + advances
                + ", fetches="
                + fetches
                + ", reads="
                + reads
                + ", prefetches="
                + prefetches
                + "]";
        }

        @Override
        public long ramBytesUsed() {
            return 0L;
        }
    }

    /** Scores from a fixed table indexed by absolute position, so expected scores need no arithmetic. */
    private static final class TableScorer implements BlockVectorScorer {
        private final RecordingBlockReader reader;
        private final float[] scoreByPosition;

        /**
         * What {@link #blockCeiling} reports. Infinity by default — "cannot bound" — so a case that is not
         * about the ceiling gate behaves as though there were none.
         */
        private float ceiling = Float.POSITIVE_INFINITY;

        private TableScorer(RecordingBlockReader reader, float[] scoreByPosition) {
            this.reader = reader;
            this.scoreByPosition = scoreByPosition;
        }

        @Override
        public BlockVectorFormat.Reader reader() {
            return reader;
        }

        @Override
        public float blockCeiling(FixedBitSet validPos) {
            assertTrue(reader.fetched, "blockCeiling() before fetchBlock()");
            assertFalse(reader.loaded, "blockCeiling() must be answerable without the codes");
            return ceiling;
        }

        @Override
        public float scoreBlock(FixedBitSet validPos, BlockCandidates out) {
            assertTrue(reader.loaded, "scoreBlock() without readBlockVectors()");
            int base = reader.firstPosition();
            int n = 0;
            float max = -Float.MAX_VALUE;
            for (int pos = 0; pos < validPos.length(); pos++) {
                if (!validPos.get(pos)) {
                    continue;
                }
                assertTrue(
                    base + pos < scoreByPosition.length,
                    "asked to score position " + pos + " of block " + reader.currentBlock + ", which is past the end of the sequence"
                );
                float score = scoreByPosition[base + pos];
                out.getPositions()[n] = pos;
                out.getScores()[n] = score;
                max = Math.max(max, score);
                n++;
            }
            out.setSize(n);
            assertEquals(validPos.cardinality(), n, "every requested position should be scored");
            return max;
        }
    }
}
