/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;

/**
 * Picks the {@link ClusterScan} implementation for one field's storage and stamps that field's layout facts
 * into every scan it builds.
 *
 * <p>Dispatch belongs here because it is a property of the index, not of a query: {@code .clam} records how
 * the field was written ({@code quantizerType}, {@code docBits}, metric, dimension), and that is fixed for
 * the segment's life. Per-query variation travels separately, as the {@link ScanParams} handed to
 * {@link #scan}.
 *
 * <p>Stateless and shared: one instance per field, built at reader open and reused across all queries and
 * threads. {@link #scan} allocates a fresh scan per call but shares no mutable state.
 *
 * <p>Today {@link QuantizerType#SCALAR} is the only family, so there is one branch; adding product or
 * binary quantization is an additive {@code switch} on {@code quantizerType} here, returning that family's
 * own {@link ClusterScan} — with no change to {@link Clusters}, {@link ClusterSearcher}, or the iterator
 * they drive.
 */
public final class ClusterFactory {

    private final ScalarQuantizedLayout layout;

    public ClusterFactory(QuantizerType quantizerType, byte docBits, int dimension, VectorSimilarityFunction sim) {
        if (quantizerType != QuantizerType.SCALAR) {
            throw new UnsupportedOperationException("ClusterANN supports only scalar quantization, not " + quantizerType);
        }
        if (sim != VectorSimilarityFunction.EUCLIDEAN && sim != VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            throw new UnsupportedOperationException("ClusterANN scoring supports only L2 and inner product, not " + sim);
        }
        this.layout = new ScalarQuantizedLayout(dimension, ScalarBitEncoding.fromDocBits(docBits), sim);
    }

    /**
     * A scan of this field for one query. Allocation only — it stamps in this field's layout facts and reads
     * nothing; postings and centroids are read lazily per posting, inside {@link ClusterScan#scorer}.
     *
     * @param postings a private {@code .clap} clone for this scan to read and slice through
     * @param centroidsBase the field's {@code .clac} base, copied per centroid read
     */
    ClusterScan scan(ScanParams params, IndexInput postings, CentroidVectorValues centroidsBase) {
        return new ScalarQuantizedScan(params, postings, centroidsBase, layout);
    }

    /**
     * Bytes of a posting of {@code count} vectors that a scan is near-certain to read. Asked of the family
     * because only it knows its per-vector stride, and answered without building anything so that hinting a
     * cluster stays free.
     */
    long guaranteedBytes(int count) {
        return layout.guaranteedBytes(count);
    }
}
