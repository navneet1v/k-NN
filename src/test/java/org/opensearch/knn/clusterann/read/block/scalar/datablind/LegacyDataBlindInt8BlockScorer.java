/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.block.scalar.datablind;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.format.block.BlockVectorFormat;
import org.opensearch.knn.clusterann.format.block.BlockVectorScorer;
import org.opensearch.knn.quantization.DataBlindInt8Quantizer;
import org.opensearch.knn.quantization.Int8DotProduct;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * Verbatim copy of {@code DataBlindInt8BlockScorer} as it was before the ScoreComputer seam, visibility aside: the
 * parity reference for {@code JavaInt8ScoreComputer}.
 *
 * {@link BlockVectorScorer} that scores a query against data-blind int8 codes without decompressing them. The
 * query is encoded once (absolutely, no centroid) in the scan context; each vector's score is the unsigned
 * code dot {@code Σ(qᵤ·dᵤ)} corrected to a signed dot and run through the metric transform — all delegated to
 * {@link DataBlindInt8Quantizer#score}, so the scoring math lives in exactly one place and the write and read
 * sides cannot drift.
 *
 * <p>This is the absolute-scheme counterpart of
 * {@link org.opensearch.knn.clusterann.read.block.scalar.ScalarQuantizedBlockScorer}: there is no
 * four-term centroid expansion, because nothing is centred — the correction is purely the unsigned-to-signed offset
 * residual, which {@code score} applies.
 *
 * <p>One instance per query per cluster; not thread-safe.
 */
@NotThreadSafe
class LegacyDataBlindInt8BlockScorer implements BlockVectorScorer {

    private final DataBlindInt8BlockReader reader;
    private final DataBlindInt8ScanContext query;
    private final VectorSimilarityFunction sim;
    private final int dimension;

    LegacyDataBlindInt8BlockScorer(
        final DataBlindInt8BlockReader reader,
        final DataBlindInt8ScanContext query,
        final VectorSimilarityFunction sim
    ) {
        this.reader = reader;
        this.query = query;
        this.sim = sim;
        this.dimension = reader.dimension();
    }

    @Override
    public BlockVectorFormat.Reader reader() {
        return reader;
    }

    @Override
    public float scoreBlock(final FixedBitSet validPos, final BlockCandidates out) {
        if (validPos == null || out == null) {
            throw new IllegalArgumentException("validPos and out must not be null");
        }
        final int[] positions = out.getPositions();
        final float[] scores = out.getScores();

        final byte[] codes = reader.codes();
        final float[] scale = reader.scale();
        final int[] sum = reader.sum();
        final float[] sqNorm = reader.sqNorm();

        final byte[] queryCode = query.queryCode();
        final DataBlindInt8Quantizer.QueryTerms terms = query.terms();

        int scored = 0;
        int index = validPos.nextSetBit(0);
        float maxScore = Float.NEGATIVE_INFINITY;
        while (index != DocIdSetIterator.NO_MORE_DOCS) {
            final long unsignedDot = Int8DotProduct.unsignedDot(queryCode, codes, index * dimension, dimension);
            final float score = DataBlindInt8Quantizer.score(unsignedDot, terms, scale[index], sum[index], sqNorm[index], sim);

            positions[scored] = index;
            scores[scored] = score;
            scored++;
            if (score > maxScore) {
                maxScore = score;
            }

            final int next = index + 1;
            index = next < validPos.length() ? validPos.nextSetBit(next) : DocIdSetIterator.NO_MORE_DOCS;
        }

        out.setSize(scored);
        return maxScore;
    }
}
