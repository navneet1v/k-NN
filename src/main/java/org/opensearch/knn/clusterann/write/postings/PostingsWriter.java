/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.ClusteringResult;

import java.io.IOException;

/**
 * Serializes a field's postings to {@code .clap}. The ordinal&rarr;document map is reused from the flat
 * {@code .vem}, not written here. How members are quantized into code blocks, and any rotation applied
 * first, are an implementation's own concern (configured when it is constructed), not part of this interface.
 *
 * <p>The {@code .clap} {@link org.apache.lucene.store.IndexOutput} is owned by the caller (the codec
 * writer that opens it), so implementations write into it but never close it.
 */
public interface PostingsWriter {

    /**
     * Write the postings for one field and return where each cluster landed.
     *
     * @param out          the {@code .clap} output to write into; owned by the caller and never closed here
     * @param clusters     the write-ready clustering result (ordinals, assignments, per-cluster members)
     * @param vectors      random-access view of the field's vectors, addressed by ordinal
     * @param metric       the similarity function, used to order each cluster's members by distance
     * @return where each cluster's postings landed in {@code .clap}
     * @throws IOException if writing fails
     */
    PostingsRegions write(IndexOutput out, ClusteringResult clusters, FloatVectorValues vectors, VectorSimilarityFunction metric)
        throws IOException;
}
