package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.IOException;

/**
 * Partitions a field's vectors into clusters. Deliberately a black box.
 *
 * <p>Everything the format cares about is in the {@link ClusteringResult}: how many centroids there are, where they
 * sit, and which vectors belong to which. How they were arrived at — hierarchical k-means, a sample-then-refine
 * pass, something else entirely — changes nothing downstream, because the written bytes depend only on the result.
 *
 * <p>That is also why the result carries no quality measure. A worse clustering makes postings less selective and
 * pruning less effective, but it cannot make a segment unreadable, so the writer has no reason to inspect it.
 */
public interface Clustering {

    /**
     * Cluster {@code vectors}.
     *
     * @param similarity the metric distances are measured under, which decides what "nearest centroid" means
     * @param params the build parameters, of which {@link ClusterANNWriteParams#targetClusterSize()} and
     *     {@link ClusterANNWriteParams#soarLambda()} shape the outcome
     */
    ClusteringResult cluster(VectorSource vectors, VectorSimilarityFunction similarity, ClusterANNWriteParams params)
        throws IOException;
}
