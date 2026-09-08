/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.clusterann.reader.block.scalar.ScalarEncoding;
import org.opensearch.knn.clusterann.reader.block.scalar.ScalarQuantizedCluster;

import java.io.IOException;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * Creates the {@link Cluster} for a centroid in one field.
 *
 * <p>The field's metadata decides which implementation to create, so this is the only class in the
 * read path that knows the concrete types. Everything else works through {@link Cluster}.
 */
public final class ClusterFactory {

    /** {@code quantizerId} of a scalar-quantized field. */
    public static final int QUANTIZER_SQ = 0;

    /** One quantizer per similarity, shared by every cluster and query. Immutable, so sharing is safe. */
    private final Map<VectorSimilarityFunction, OptimizedScalarQuantizer> quantizers;
    private final IndexInput postings;
    private final IndexInput centroids;
    private final IndexInput rotation;
    private final ClusterANNFieldMeta fieldMeta;
    /** The region a scan reads centroids from, cloned per cluster. Rotated when the field is, raw when it is not. */
    private final CentroidVectorValues centroidsBase;

    /**
     * @param fieldMeta this field's entry from {@code .clam}
     * @param clap this field's region of {@code .clap}, already cut at {@code clapOffset}
     * @param clac this field's region of {@code .clac}, already cut at {@code clacOffset}
     * @param clar this field's region of {@code .clar}, or {@code null} when the field stores no rotation
     */
    public ClusterFactory(ClusterANNFieldMeta fieldMeta, IndexInput clap, IndexInput clac, @Nullable IndexInput clar) throws IOException {
        this.fieldMeta = fieldMeta;
        this.postings = clap;
        this.centroids = clac;
        this.rotation = clar;

        EnumMap<VectorSimilarityFunction, OptimizedScalarQuantizer> quantizers = new EnumMap<>(VectorSimilarityFunction.class);
        for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
            quantizers.put(similarity, new OptimizedScalarQuantizer(similarity));
        }
        this.quantizers = Collections.unmodifiableMap(quantizers);

        // The centroids a scan centres on: the rotated ones for a rotated field, since its codes live in that space,
        // and the only ones there are otherwise. Sliced rather than pointed at, so the cursor cannot reach a
        // neighbouring region and every read is bounds-checked against this field's centroids alone.
        long centroidsOffset = fieldMeta.hasRotation() ? fieldMeta.clacRotatedCentroidsOffset() : fieldMeta.clacCentroidsOffset();
        long centroidsBytes = (long) fieldMeta.centroidCount() * (fieldMeta.dimension() + 1) * Float.BYTES;
        this.centroidsBase = new CentroidVectorValues(
            centroids.slice("centroids", centroidsOffset, centroidsBytes),
            fieldMeta.centroidCount(),
            fieldMeta.dimension(),
            true
        );
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
        if (this.fieldMeta.quantizerId() != QUANTIZER_SQ) {
            throw new IllegalArgumentException("Unsupported quantizerId: " + fieldMeta.quantizerId());
        }

        IOSupplier<Centroid> centroid = () -> {
            CentroidVectorValues cursor = (CentroidVectorValues) centroidsBase.copy();
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
            this.quantizers.get(fieldMeta.similarityFunction()),
            fieldMeta.similarityFunction()
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
