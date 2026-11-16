/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOSupplier;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.read.block.scalar.datablind.DataBlindInt8Cluster;
import org.opensearch.knn.clusterann.read.block.scalar.datablind.DataBlindInt8ScanBuffers;
import org.opensearch.knn.clusterann.read.block.scalar.SQScanBuffers;
import org.opensearch.knn.clusterann.read.block.scalar.ScalarQuantizedCluster;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;
import org.opensearch.knn.quantization.ScalarQuantizers;

import javax.annotation.Nullable;
import java.io.IOException;

import static org.opensearch.knn.clusterann.read.CentroidVectorValues.floatsPerCentroid;

/**
 * Creates the {@link Cluster} for a centroid in one field.
 *
 * <p>The field's metadata decides which implementation to create, so this is the only class in the
 * read path that knows the concrete types. Everything else works through {@link Cluster}.
 */
public final class ClusterFactory {

    private final IndexInput postings;
    private final IndexInput centroids;
    private final IndexInput rotation;
    private final ClusterANNFieldMeta fieldMeta;
    private final ScoreComputer computer;
    /** The region a scan reads centroids from, cloned per cluster. Rotated when the field is, raw when it is not. */
    private final CentroidVectorValues centroidsBase;

    /**
     * Entries in this field's largest cluster, so a scan's per-cluster buffers can be sized once to hold any of
     * them. One pass over the metadata here, rather than growing a buffer as a scan meets bigger clusters.
     */
    private final int maxClusterSize;

    /**
     * @param fieldMeta this field's entry from {@code .clam}
     * @param clap this field's region of {@code .clap}, already cut at {@code clapOffset}
     * @param clac this field's region of {@code .clac}, already cut at {@code clacOffset}
     * @param clar this field's region of {@code .clar}, or {@code null} when the field stores no rotation
     * @param computer scores the blocks of every cluster this factory creates
     */
    public ClusterFactory(
        ClusterANNFieldMeta fieldMeta,
        IndexInput clap,
        IndexInput clac,
        @Nullable IndexInput clar,
        ScoreComputer computer
    ) throws IOException {
        this.fieldMeta = fieldMeta;
        this.computer = computer;
        this.postings = clap;
        this.centroids = clac;
        this.rotation = clar;

        long centroidsOffset = fieldMeta.hasRotation() ? fieldMeta.clacRotatedCentroidsOffset() : fieldMeta.clacCentroidsOffset();
        long centroidsBytes = (long) fieldMeta.centroidCount() * floatsPerCentroid(fieldMeta.dimension()) * Float.BYTES;
        this.centroidsBase = new CentroidVectorValues(
            centroids.slice("centroids-rotated", centroidsOffset, centroidsBytes),
            fieldMeta.centroidCount(),
            fieldMeta.dimension()
        );

        int largest = 0;
        for (int clusterSize : fieldMeta.clusterSizes()) {
            largest = Math.max(largest, clusterSize);
        }
        this.maxClusterSize = largest;
    }

    /**
     * Creates the cluster for one centroid.
     *
     * <p>Does no reading. The centroid vector is loaded only if the cluster is scanned, so creating
     * clusters you may never visit is cheap.
     *
     * @param ordinal the centroid's ordinal within the field
     * @throws IllegalArgumentException if the field's metadata names a quantizer or code width this
     *     reader does not support
     */
    public Cluster create(int ordinal) throws IOException {
        return switch (fieldMeta.quantizerId()) {
            case ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ -> optimizedScalarCluster(ordinal);
            case ClusterANNFormatConstants.QUANTIZER_DATA_BLIND_INT8 -> dataBlindInt8Cluster(ordinal);
            default -> throw new IllegalArgumentException("Unsupported quantizerId: " + fieldMeta.quantizerId());
        };
    }

    /**
     * The scratch one scan of this field reuses across every cluster it probes.
     *
     * <p>Made here rather than by a cluster because the size depends only on the field's quantizer and geometry,
     * both of which this class already holds — so a scan can have its buffers before it has looked up a single
     * cluster, and the clusters need not agree among themselves on who sizes them.
     */
    public ScanBuffers newScanBuffers() {
        return switch (fieldMeta.quantizerId()) {
            case ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ -> new SQScanBuffers(
                fieldMeta.blockSize(),
                fieldMeta.dimension(),
                scalarEncoding(fieldMeta.docBits())
            );
            case ClusterANNFormatConstants.QUANTIZER_DATA_BLIND_INT8 -> new DataBlindInt8ScanBuffers(
                fieldMeta.blockSize(),
                fieldMeta.dimension(),
                maxClusterSize
            );
            default -> throw new IllegalArgumentException("Unsupported quantizerId: " + fieldMeta.quantizerId());
        };
    }

    private Cluster optimizedScalarCluster(int ordinal) throws IOException {
        IOSupplier<Centroid> centroid = () -> {
            CentroidVectorValues cursor = centroidsBase.copy();
            float[] vector = cursor.vectorValue(ordinal);
            return new Centroid(vector, cursor.norm());
        };

        return new ScalarQuantizedCluster(
            posting(this.fieldMeta, ordinal, postings),
            ordinal,
            fieldMeta.clusterSizes()[ordinal],
            centroid,
            fieldMeta.blockSize(),
            fieldMeta.dimension(),
            scalarEncoding(fieldMeta.docBits()),
            ScalarQuantizers.forSimilarity(fieldMeta.similarityFunction()),
            fieldMeta.similarityFunction(),
            computer
        );
    }

    private Cluster dataBlindInt8Cluster(int ordinal) throws IOException {
        // Absolute scheme: no centroid is read to score, and the query arrives already rotated by the reader,
        // so this cluster needs neither the centroid nor the rotation — only the posting and the metric.
        return new DataBlindInt8Cluster(
            posting(this.fieldMeta, ordinal, postings),
            ordinal,
            fieldMeta.clusterSizes()[ordinal],
            fieldMeta.dimension(),
            fieldMeta.similarityFunction(),
            computer
        );
    }

    /**
     * Slices this cluster's posting out of the field's {@code .clap} region.
     *
     * <p>{@code clap} is that region, not the whole file — the caller has already cut it at {@code clapOffset} —
     * and the metadata's per-centroid offsets are relative to it, so the offset is used as it is stored.
     */
    private static IndexInput posting(ClusterANNFieldMeta fieldMeta, int ordinal, IndexInput clap) throws IOException {
        long offset = fieldMeta.clapCentroidOffsets()[ordinal];
        int length = fieldMeta.centroidLengths()[ordinal];
        return clap.slice("cluster-" + ordinal, offset, length);
    }

    /** Maps the stored code width to its encoding, so the bit count in the metadata is the only source of truth. */
    private static ScalarEncoding scalarEncoding(int docBits) {
        try {
            return ScalarEncoding.fromNumBits(docBits);
        } catch (IllegalArgumentException e) {
            // Rethrown to name where the width came from: the field's metadata, not a caller's argument.
            throw new IllegalArgumentException("Unsupported docBits: " + docBits, e);
        }
    }
}
