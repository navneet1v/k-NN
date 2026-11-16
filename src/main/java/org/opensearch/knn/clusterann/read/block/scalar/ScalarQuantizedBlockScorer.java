package org.opensearch.knn.clusterann.read.block.scalar;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.clusterann.read.ScoreComputer;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.Set;

import static org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding.DIBIT_QUERY_NIBBLE;
import static org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding.PACKED_NIBBLE;
import static org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

/**
 * {@link BlockVectorScorer} over scalar-quantized blocks that delegates the arithmetic to a {@link ScoreComputer}.
 *
 * <p>Per block it turns the accepted positions into a dense {@code int[]}, hands that list and the compute context
 * (the block's arrays plus the prepared query) to the computer in one call, and returns the block maximum the computer reports. Which
 * computer runs is the field's choice at reader open: the format's {@link JavaAdcScoreComputer} unless a
 * {@link org.opensearch.knn.clusterann.read.ScoreComputerFactory} supplied another.
 *
 * <p>Both sides being centered on one centroid is what ties an instance to a single cluster and a single query.
 */
@NotThreadSafe
public class ScalarQuantizedBlockScorer implements BlockVectorScorer {

    /** The code widths the format writes and every computer must score: 1 and 2 bit transposed, 4-bit packed nibbles. */
    static final Set<ScalarEncoding> VALID_ENCODINGS = Set.of(SINGLE_BIT_QUERY_NIBBLE, DIBIT_QUERY_NIBBLE, PACKED_NIBBLE);

    private final ScalarQuantizedBlockReader reader;
    private final SQComputeContext computeContext;
    private final ScoreComputer computer;
    private final int generation;

    /**
     * @param reader the cluster's block reader; must be the one whose buffers {@code computeContext} holds
     * @param computeContext the reader's block buffers and the query prepared for this cluster
     * @param computer scores each block's accepted positions
     * @throws IllegalArgumentException if the code width has no kernel, or the context holds another reader's buffers
     */
    public ScalarQuantizedBlockScorer(ScalarQuantizedBlockReader reader, SQComputeContext computeContext, ScoreComputer computer) {
        ScalarEncoding encoding = computeContext.encoding();
        if (!VALID_ENCODINGS.contains(encoding)) {
            throw new IllegalArgumentException(
                "Unsupported encoding "
                    + encoding
                    + " ("
                    + encoding.getDocBitsPerDim()
                    + " document bits); supported"
                    + " are 1 and 2 bit transposed codes and 4-bit packed nibbles"
            );
        }
        if (computeContext.codes() != reader.codes()) {
            throw new IllegalArgumentException("The compute context holds another reader's block buffers");
        }
        this.reader = reader;
        this.computeContext = computeContext;
        this.computer = computer;
        this.generation = reader.generation();
    }

    @Override
    public BlockVectorFormat.Reader reader() {
        return reader;
    }

    @Override
    public void ensureCurrentSequence() {
        if (reader.generation() != generation) {
            throw new IllegalStateException(
                "The scan's block reader has moved to another cluster; this scorer must be drained before the next" + " cluster is scanned"
            );
        }
    }

    @Override
    public float scoreBlock(final FixedBitSet validPos, final BlockCandidates out) {
        if (validPos == null || out == null) {
            throw new IllegalArgumentException("validPos and out must not be null");
        }
        ensureCurrentSequence();
        final int[] positions = out.getPositions();

        int count = 0;
        int index = validPos.nextSetBit(0);
        while (index != DocIdSetIterator.NO_MORE_DOCS) {
            positions[count++] = index;
            // nextSetBit asserts its argument is inside the set, so the last position needs the explicit stop.
            int next = index + 1;
            index = next < validPos.length() ? validPos.nextSetBit(next) : DocIdSetIterator.NO_MORE_DOCS;
        }
        out.setSize(count);
        // BlockPostingScorer skips a block whose accepted set is empty, so this is for other callers (tests included):
        // it keeps the contract of the interface without a call into the computer.
        if (count == 0) {
            return Float.NEGATIVE_INFINITY;
        }
        return computer.score(computeContext, positions, count, out.getScores());
    }
}
