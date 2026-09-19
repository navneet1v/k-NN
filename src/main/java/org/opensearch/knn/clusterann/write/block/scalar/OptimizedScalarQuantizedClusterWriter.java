/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.block.scalar;

import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter;

import java.io.IOException;

/**
 * {@link ClusterWriter} for the optimized-scalar-quantization family, one per cluster. Writes the cluster
 * header — the member ordinals, then the secondary-assignment bitset, then the parallel distances — and lays
 * the member vectors down after it as scalar-quantized code blocks, quantized against the cluster's centre by
 * an {@link OptimizedScalarQuantizedBlockWriter} it builds around that centre.
 *
 * <p>The output is supplied per {@link #write} call rather than held from construction: this writer keeps only
 * the per-cluster quantization config and centre, and touches the shared {@code .clap} output solely for the
 * duration of that one call.
 *
 * <p>The block byte format lives entirely inside the block writer; the header layout here is independent
 * of it.
 *
 * <pre>
 *   ordinals                    clusterSize x int              one int per member, in array order
 *   secondaryAssignmentBitSet   bits2words(clusterSize) x long bit i set =&gt; ordinals[i] is a spill member
 *   distances                   clusterSize x float            parallel to ordinals
 *   code blocks                 SoA blocks of blockSize        (see OptimizedScalarQuantizedBlockWriter)
 * </pre>
 */
public final class OptimizedScalarQuantizedClusterWriter implements ClusterWriter {

    private final int blockSize;
    private final int dimension;
    private final OptimizedScalarQuantizer quantizer;
    private final ScalarEncoding encoding;
    private final float[] centroid;

    /**
     * @param blockSize vectors per code block
     * @param dimension the field's vector dimension
     * @param metric    the field's similarity function, driving the scalar quantizer
     * @param docBits   bits per quantized coordinate, driving the scalar code width
     * @param centroid  this cluster's centre, already rotation-prepared, that members are quantized against
     */
    public OptimizedScalarQuantizedClusterWriter(
        final int blockSize,
        final int dimension,
        final VectorSimilarityFunction metric,
        final byte docBits,
        final float[] centroid
    ) {
        this.blockSize = blockSize;
        this.dimension = dimension;
        this.quantizer = new OptimizedScalarQuantizer(metric);
        this.encoding = ScalarEncoding.fromNumBits(docBits);
        this.centroid = centroid;
    }

    @Override
    public ClusterRegion write(final IndexOutput out, final FloatVectorValues vectors, final ClusterMembers members) throws IOException {
        if (vectors.size() != members.ordinals().length) {
            throw new IllegalArgumentException("vectors (" + vectors.size() + ") must match members (" + members.ordinals().length + ")");
        }
        // Clustering never emits a zero-member centroid — HierarchicalKMeans drops empty clusters and an empty
        // field yields no centroids at all — so an empty cluster here is a broken invariant, not a valid input.
        if (members.ordinals().length == 0) {
            throw new IllegalArgumentException("cluster has no members");
        }
        final long start = out.getFilePointer();

        writeHeader(out, members);
        new OptimizedScalarQuantizedBlockWriter(out, blockSize, dimension, quantizer, encoding, centroid).writeBlocks(vectors);

        return new ClusterRegion(start, out.getFilePointer() - start);
    }

    /** Writes the cluster header: ordinals, then the secondary-assignment bitset, then the parallel distances. */
    private static void writeHeader(final IndexOutput out, final ClusterMembers members) throws IOException {
        for (final int ordinal : members.ordinals()) {
            out.writeInt(ordinal);
        }
        writeSecondaryAssignmentBitSet(out, members.secondary());
        for (final float distance : members.distances()) {
            out.writeInt(Float.floatToIntBits(distance));
        }
    }

    /**
     * Writes the secondary-assignment flags as a {@link FixedBitSet} — {@code bits2words(clusterSize)} longs, the
     * layout Lucene uses to persist a bitset over a fixed entry count — with bit i set when {@code secondary[i]}
     * (i.e. a spill member). Reads back via {@code readLongs} into {@code new FixedBitSet(words, clusterSize)}.
     */
    private static void writeSecondaryAssignmentBitSet(final IndexOutput out, final boolean[] secondary) throws IOException {
        final FixedBitSet bitset = new FixedBitSet(secondary.length);
        for (int i = 0; i < secondary.length; i++) {
            if (secondary[i]) {
                bitset.set(i);
            }
        }
        for (final long word : bitset.getBits()) {
            out.writeLong(word);
        }
    }
}
