/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.read.Centroid;
import org.opensearch.knn.clusterann.read.PostingScorer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * That a {@link ScalarQuantizedCluster} clips on its own header: the distances the pruner uses are read from the
 * posting by stride, so a wrong offset would stop the walk in the wrong place rather than fail outright.
 *
 * <p>The shells are far apart on purpose — block 0 at 0.5 from the centroid, block 1 at 10, block 2 at 20 — so
 * the bound each block carries is unmistakable and the threshold that separates them is wide.
 */
class SQClusterClipTests {

    private static final ScalarEncoding ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;
    private static final int DIMENSION = 64;
    private static final int PACKED_BYTES = ENCODING.getDocPackedLength(DIMENSION);
    private static final int BLOCK_SIZE = 4;
    private static final int CLUSTER_SIZE = 10;
    private static final int CENTROID_ORDINAL = 7;
    private static final String FILE = "posting";

    private static final int[] ORDINALS = { 40, 3, 91, 12, 55, 7, 88, 21, 64, 30 };

    /** Squared distance to the centroid per position, ascending: shells at 0.5, then 10, then 20. */
    private static final float[] SQUARED_DISTANCE = { 0.25f, 0.26f, 0.27f, 0.28f, 100f, 101f, 102f, 103f, 400f, 401f };

    /** {@code ‖q−c‖² = 0.01}, so the query sits 0.1 from the centroid — inside the first shell only. */
    private static final float QUERY_CENTROID_DISTANCE_SQ = 0.01f;

    /**
     * Between block 1's bound, {@code 1/(1+9.9²) ≈ 0.0101}, and block 0's, {@code 1/(1+0.4²) ≈ 0.862}: high
     * enough to clip the far shells, low enough that the near one still competes.
     */
    private static final float CLIPPING_THRESHOLD = 0.05f;

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    @Test
    void testScorer_whenTheFarShellsCannotCompete_thenStopsAtTheFirstBlock() throws IOException {
        // given
        PostingScorer scorer = cluster().scorer(scanContext(), null);

        // when
        List<Integer> visited = drain(scorer, CLIPPING_THRESHOLD);

        // then
        assertEquals(List.of(ORDINALS[0], ORDINALS[1], ORDINALS[2], ORDINALS[3]), visited);
    }

    /** The same posting and the same query, with a threshold nothing can be clipped at. */
    @Test
    void testScorer_whenEveryShellCanStillCompete_thenWalksTheWholePosting() throws IOException {
        // given
        PostingScorer scorer = cluster().scorer(scanContext(), null);

        // when
        List<Integer> visited = drain(scorer, 0f);

        // then
        List<Integer> everyOrdinal = new ArrayList<>();
        for (int ordinal : ORDINALS) {
            everyOrdinal.add(ordinal);
        }
        assertEquals(everyOrdinal, visited);
    }

    // ---------------------------------------------------------------- helpers

    private static List<Integer> drain(PostingScorer scorer, float minCompetitiveSimilarity) throws IOException {
        List<Integer> ords = new ArrayList<>();
        while (scorer.advance(minCompetitiveSimilarity)) {
            ords.add(scorer.ord());
            if (ords.size() > 50) {
                throw new AssertionError("advance() never returned false");
            }
        }
        return ords;
    }

    private static SQScanContext scanContext() {
        float[] query = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            query[i] = (i % 7) * 0.25f - 0.5f;
        }
        byte[] transposed = new byte[ENCODING.getQueryPackedLength(DIMENSION)];
        for (int i = 0; i < transposed.length; i++) {
            transposed[i] = (byte) (0x33 + i);
        }
        return new SQScanContext(query, 4, 1.0f, transposed, -0.75f, 0.05f, 42f, QUERY_CENTROID_DISTANCE_SQ);
    }

    private ScalarQuantizedCluster cluster() throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        writePosting(directory);
        return new ScalarQuantizedCluster(
            directory.openInput(FILE, IOContext.DEFAULT),
            CENTROID_ORDINAL,
            CLUSTER_SIZE,
            () -> new Centroid(new float[DIMENSION], 1.0f),
            BLOCK_SIZE,
            DIMENSION,
            ENCODING,
            new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN),
            VectorSimilarityFunction.EUCLIDEAN
        );
    }

    /** Ordinals, an empty SOAR bitset, the squared distances, then blocks whose codes only have to be scorable. */
    private static void writePosting(Directory directory) throws IOException {
        try (IndexOutput out = directory.createOutput(FILE, IOContext.DEFAULT)) {
            for (int ordinal : ORDINALS) {
                out.writeInt(ordinal);
            }
            for (long word : new FixedBitSet(CLUSTER_SIZE).getBits()) {
                out.writeLong(word);
            }
            for (float squaredDistance : SQUARED_DISTANCE) {
                out.writeInt(Float.floatToIntBits(squaredDistance));
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
