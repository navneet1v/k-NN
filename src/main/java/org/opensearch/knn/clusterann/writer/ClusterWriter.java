package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.clusterann.reader.Centroid;

import java.io.IOException;

/**
 * Writes one cluster's posting — the twin of {@code Cluster}.
 *
 * <p>Deliberately narrower than the read side. A {@code Cluster} is addressable, because a scan's whole job is
 * choosing which clusters and which blocks <em>not</em> to read; a writer appends cluster after cluster, so it needs
 * no handle, no seek, and no per-cluster object to hold. That is the general shape of the asymmetry: the reader has
 * more structure than the writer because only the reader gets to skip.
 *
 * <p>Where the posting lands is the caller's business — it records the output's position before and after — so this
 * writes at the current pointer and reports nothing.
 */
public interface ClusterWriter {

    /**
     * Append {@code posting} at {@code clap}'s current position.
     *
     * @param reference the centroid this cluster's codes are quantized against, in the space the codes live in.
     *     Named for its role rather than as "the centroid" because the two are not always the same point: a
     *     cluster's own centroid decides membership and the geometric bound, while the reference is what residuals
     *     are measured from.
     * @param vectors the field's vectors, in the stored space, addressed by ordinal
     */
    void write(IndexOutput clap, Posting posting, Centroid reference, VectorSource vectors) throws IOException;
}
