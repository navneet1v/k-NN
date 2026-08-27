/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Picks the {@link Cluster} implementation for one field's storage and stamps that field's layout facts
 * into each cluster it builds.
 *
 * <p>Dispatch belongs here because it is a property of the index, not of a query: {@code .clam} records
 * how the field was written ({@code quantizerType}, {@code docBits}, metric, dimension), and that is
 * fixed for the segment's life. Per-query variation travels separately, as {@link ScanParams} handed to
 * {@link Cluster#scorer}.
 *
 * <p>Stateless and shared: one instance per field, built at reader open and reused across all queries
 * and threads. {@link #create} allocates a fresh {@link Cluster} per call but shares no mutable state.
 *
 * <p>Today {@link QuantizerType#SCALAR} is the only family, so there is one branch; adding product or
 * binary quantization is an additive {@code switch} on {@code quantizerType} here, returning that
 * family's own {@link Cluster} — with no change to {@link Clusters}, {@link ClusterSearcher}, or the
 * iterator they drive.
 */
public final class ClusterFactory {

    private final int dimension;
    private final ScalarBitEncoding encoding;
    private final VectorSimilarityFunction sim;
    private final OptimizedScalarQuantizer quantizer;

    public ClusterFactory(QuantizerType quantizerType, byte docBits, int dimension, VectorSimilarityFunction sim) {
        if (quantizerType != QuantizerType.SCALAR) {
            throw new UnsupportedOperationException("ClusterANN supports only scalar quantization, not " + quantizerType);
        }
        if (sim != VectorSimilarityFunction.EUCLIDEAN && sim != VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            throw new UnsupportedOperationException("ClusterANN scoring supports only L2 and inner product, not " + sim);
        }
        this.dimension = dimension;
        this.encoding = ScalarBitEncoding.fromDocBits(docBits);
        this.sim = sim;
        this.quantizer = new OptimizedScalarQuantizer(sim);
    }

    /**
     * A cluster over the posting at {@code relativeOffset} (slice-relative) in {@code postings}, reading
     * its centroid through {@code centroids}. Allocation only — it stamps in this field's layout facts and
     * reads nothing; both the posting and the centroid are read lazily in {@link Cluster#scorer}.
     *
     * @param postings a private {@code .clap} clone for this cluster to read and hint through
     * @param centroid a capability to read this cluster's own geometry, invoked lazily
     */
    Cluster create(
        int ordinal,
        int count,
        IndexInput postings,
        long relativeOffset,
        long postingBytes,
        IOSupplier<Centroid> centroid
    ) {
        return new ScalarQuantizedCluster(
            ordinal, count, postings, relativeOffset, postingBytes, centroid, dimension, encoding, sim, quantizer);
    }
}
