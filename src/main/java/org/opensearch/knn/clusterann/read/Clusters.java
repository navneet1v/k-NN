/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.LongValues;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.read.orchestration.ClusterScan;
import org.opensearch.knn.clusterann.read.orchestration.DefaultClusterScan;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;

import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;
import java.io.IOException;
import java.io.UncheckedIOException;

import static org.opensearch.knn.clusterann.read.CentroidVectorValues.floatsPerCentroid;

/**
 * The clusters of one field: a persistent, query-independent handle. Intended to be built once per field and
 * reused. It has access to all the information of the clusters and acts as an abstraction to navigate and get
 * the information.
 *
 * <p>Immutable and thread-safe. It holds the {@code .clap} input, the immutable {@link ClusterANNFieldMeta}, and
 * <em>base</em> assignment and transformed-centroid readers over {@code .clac}.
 */
@ThreadSafe
public final class Clusters {

    private final ClusterANNFieldMeta fieldMeta;
    private final ClusterFactory clusterFactory;
    private final IndexInput ordToCentroidBase;
    private final CentroidVectorValues centroidsBase;
    private final Rotation rotation;

    /**
     * @param postingsSlice  this field's region of {@code .clap}, already cut at {@code clapOffset}; a cluster's own
     *     posting is sliced out of it per {@link #get}
     * @param centroidsSlice this field's region of {@code .clac}, already cut at {@code clacOffset}
     * @param rotationSlice  this field's region of {@code .clar}, or {@code null} when the field stores no rotation
     * @param fieldMeta      this field's entry from {@code .clam}
     * @param computer       scores the blocks of this field's clusters; the field's choice at reader open
     */
    public Clusters(
        final IndexInput postingsSlice,
        final IndexInput centroidsSlice,
        @Nullable final IndexInput rotationSlice,
        final ClusterANNFieldMeta fieldMeta,
        final ScoreComputer computer
    ) throws IOException {
        this.fieldMeta = fieldMeta;
        this.clusterFactory = new ClusterFactory(fieldMeta, postingsSlice, centroidsSlice, rotationSlice, computer);
        this.rotation = RotationFormats.read(fieldMeta.rotationId(), rotationSlice, fieldMeta.dimension());

        long ordToCentroidBytes = Math.multiplyExact((long) fieldMeta.vectorCount(), Integer.BYTES);
        this.ordToCentroidBase = centroidsSlice.slice("ord-to-centroid", 0L, ordToCentroidBytes);

        long centroidsBytes = (long) fieldMeta.centroidCount() * floatsPerCentroid(fieldMeta.dimension()) * Float.BYTES;
        this.centroidsBase = new CentroidVectorValues(
            centroidsSlice.slice("centroids", fieldMeta.clacCentroidsOffset(), centroidsBytes),
            fieldMeta.centroidCount(),
            fieldMeta.dimension()
        );
    }

    /** Whether every document in this segment has this vector field, making document ids equal vector ordinals. */
    public boolean hasDenseVectorValues() {
        return fieldMeta.ordToDoc().isDense();
    }

    /**
     * The rotation to put a query through before scoring it against this field's vectors. The identity for a field
     * stored unrotated, so a caller need not ask which it is.
     */
    public Rotation rotation() {
        return rotation;
    }

    public ClusterANNFieldMeta clusterMeta() {
        return fieldMeta;
    }

    /**
     * A private cursor over this field's centroids, in the space they were written in — the region a planner sweeps to
     * rank clusters against a query. Reads nothing until a centroid is asked for.
     *
     * <p>A copy per call, since the cursor carries a moving file pointer and a buffer it reuses, while this structure
     * is shared across concurrent searches. Cloning allocates but reads nothing.
     */
    public CentroidVectorValues centroids() throws IOException {
        return centroidsBase.copy();
    }

    /** Number of clusters in this field. */
    public int numClusters() {
        return fieldMeta.centroidCount();
    }

    /**
     * A query-private, read-only primary centroid assignment lookup over region 1 of this field's {@code .clac} data.
     * Creating it reads nothing; each assignment is read lazily through positional access.
     */
    public LongValues ordToCentroid() throws IOException {
        final RandomAccessInput assignments = ordToCentroidBase.randomAccessSlice(0L, ordToCentroidBase.length());
        return new LongValues() {
            @Override
            public long get(long ordinal) {
                try {
                    return assignments.readInt(ordinal * Integer.BYTES);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        };
    }

    /** Number of vectors in this field, across all clusters. */
    public int numVectors() {
        return fieldMeta.vectorCount();
    }

    /**
     * Number of vectors in the cluster with this centroid ordinal, primary and SOAR together.
     */
    public int clusterSize(int ordinal) {
        return fieldMeta.clusterSizes()[ordinal];
    }

    /**
     * Returns a scanner for one query.
     *
     * <p>Call this once per query, then call {@link ClusterScan#scorer} once for each cluster you want to search,
     * draining each scorer before asking for the next — the clusters of one scan are visited one at a time. What
     * that scanner shares across those clusters, and why, is {@link DefaultClusterScan}.
     */
    public ClusterScan scan() {
        return new DefaultClusterScan(clusterFactory.newScanBuffers());
    }

    /**
     * The cluster with this centroid ordinal. <b>Reads nothing</b> — it looks up layout facts and hands the cluster
     * private cursors to read through when it needs to. Both the posting and the cluster's own centroid are read
     * lazily inside {@link Cluster#scorer}.
     *
     * <p>That laziness is deliberate: it makes a walk's skip checks free, so a get costs no I/O.
     */
    public Cluster get(int ordinal) throws IOException {
        return clusterFactory.create(ordinal);
    }
}
