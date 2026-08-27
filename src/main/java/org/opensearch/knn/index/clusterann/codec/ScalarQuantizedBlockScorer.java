/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Pluggable per-block scoring strategy over <b>scalar-quantized</b> block-columnar vectors — i.e.
 * packed codes plus {@link OptimizedScalarQuantizer} corrective terms ({@code lower/upper/add/sum}).
 * This is intentionally concrete to that quantization family: the corrective-term signature is
 * exactly what a scalar-quantized scorer needs, and there is no meaningful behavior shared with
 * non-quantized storage. A different storage family defines its own scorer; it is not forced through
 * this interface.
 *
 * <p>What is pluggable here is the <em>scoring math</em> over the same scalar-quantized blocks:
 * ADC (default; query precise, doc quantized), symmetric (SDC), reconstruct-and-score, … all consume
 * {@code codes + lower/upper/add/sum}. A scorer instance is bound to one query + one posting's
 * centroid via {@link Factory#create}; it is handed a block's columnar data as arguments and does
 * not reach into the values' buffers. Scores are similarities (higher = closer); collection cutoffs
 * stay with the caller.
 *
 * <p>This interface is <em>pure scoring</em>. Block-skip decisions live in the pruning strategies
 * ({@link PostingPruner} pre-I/O, {@link CorrectionsPruner} post-corrections), not here.
 */
public interface ScalarQuantizedBlockScorer extends BlockScorer {

    // Scoring is {@link BlockScorer#scoreBlock(int, Bits, float[])}: the scorer holds the block reader +
    // query context, so it reads codes/corrections itself rather than being handed columns. Instances are
    // built directly by the cluster, which owns both the reader and the query context they share.
}
