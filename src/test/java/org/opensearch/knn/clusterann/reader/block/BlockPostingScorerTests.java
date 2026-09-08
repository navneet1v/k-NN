/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader.block;

import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.reader.PostingScorer;

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
        RecordingBlockReader expectedReader = readerThatSaw(vectorCount).advanced(blocksWalked(vectorCount))
            .fetched(everyBlock)
            .read(everyBlock)
            .prefetched(blocksAfterFirst(vectorCount))
            .reader();

        assertEquals(expectedHits, hits);
        assertEquals(expectedReader, scan.reader);
    }

    /**
     * Each case gives the accepted positions and then, per rung: which blocks get positioned on, which get
     * fetched and decoded, and which get hinted.
     *
     * <p>Block 0 is always positioned on, and block 3 is the out-of-range index that ends the walk. Beyond
     * those, a block with nothing accepted is never even positioned on, because the lookahead skips straight
     * past it — and a hint only ever names a block that is then fetched.
     */
    private static Stream<Arguments> filters() {
        return Stream.of(
            Arguments.of("one whole block", new int[] { 4, 5, 6, 7 }, List.of(0, 1, 3), List.of(1), List.of(1)),
            Arguments.of(
                "one position per block, each at a different offset",
                new int[] { 2, 5, 11 },
                List.of(0, 1, 2, 3),
                List.of(0, 1, 2),
                List.of(1, 2)
            ),
            Arguments.of("only the first and last positions", new int[] { 0, 11 }, List.of(0, 2, 3), List.of(0, 2), List.of(2)),
            Arguments.of("nothing at all", new int[] {}, List.of(0, 3), List.of(), List.of())
        );
    }

    /** A block with no accepted ordinal must be skipped whole — not filtered after scoring, and never hinted. */
    @ParameterizedTest(name = "accepting {0}")
    @MethodSource("filters")
    void testAdvance_whenOrdinalsAreFiltered_thenTouchesOnlyBlocksWithAnAcceptedOrdinal(
        String description,
        int[] acceptedPositions,
        List<Integer> expectedAdvances,
        List<Integer> expectedBlocks,
        List<Integer> expectedPrefetches
    ) throws IOException {
        // given
        Scan scan = scanOver(ORDINALS, SCORES, bitsAt(acceptedPositions));

        // when
        List<Hit> hits = drain(scan, ACCEPT_ALL);

        // then
        List<Hit> expectedHits = hitsAt(ORDINALS, SCORES, acceptedPositions);
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(expectedAdvances)
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
        RecordingBlockReader expectedReader = readerThatSaw(ORDINALS.length).advanced(blocksWalked(ORDINALS.length))
            .fetched(everyBlock)
            .read(everyBlock)
            .prefetched(blocksAfterFirst(ORDINALS.length))
            .reader();

        assertEquals(expectedHits, hits);
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

    // ---------------------------------------------------------------- helpers

    private record Hit(int ord, float score) {
    }

    /** A scorer plus the reader it drives, so a test can assert on the results and on the I/O they cost. */
    private record Scan(PostingScorer scorer, RecordingBlockReader reader) {
    }

    private static Scan scanOver(int[] ordinals, float[] scores, Bits acceptedOrds) {
        RecordingBlockReader reader = new RecordingBlockReader(ordinals.length, BLOCK_SIZE);
        BlockVectorScorer blockScorer = new TableScorer(reader, scores);
        return new Scan(new BlockPostingScorer(blockScorer, ordinals, acceptedOrds), reader);
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

    /**
     * Every block a full walk positions on, plus the out-of-range index that ends it — positioning is free, so
     * a walk pays it for every block including the ones it then declines to fetch.
     */
    private static List<Integer> blocksWalked(int vectorCount) {
        return IntStream.rangeClosed(0, numBlocks(vectorCount)).boxed().toList();
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

        private TableScorer(RecordingBlockReader reader, float[] scoreByPosition) {
            this.reader = reader;
            this.scoreByPosition = scoreByPosition;
        }

        @Override
        public BlockVectorFormat.Reader reader() {
            return reader;
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
