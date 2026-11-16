/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.block.scalar;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter;

import java.io.IOException;

/**
 * {@link ClusterWriter} for the data-blind int8 family, one per cluster. Writes the cluster header — the
 * member ordinals, then the secondary-assignment bitset, then the parallel distances — and lays the member
 * vectors down after it as data-blind int8 code blocks via a {@link DataBlindInt8BlockWriter}.
 *
 * <p>The header layout is identical to the optimized-scalar family and backend-independent; only the payload
 * (the code blocks) differs. Unlike the optimized-scalar writer this takes <em>no centroid</em>: the scheme is
 * absolute, so members are quantized in rotated space without any centre, and the centroid file is used only
 * for routing.
 *
 * <pre>
 *   ordinals                    clusterSize x int              one int per member, in array order
 *   secondaryAssignmentBitSet   bits2words(clusterSize) x long bit i set =&gt; ordinals[i] is a spill member
 *   distances                   clusterSize x float            parallel to ordinals
 *   code blocks                 SoA blocks of blockSize        (see DataBlindInt8BlockWriter)
 * </pre>
 */
public final class DataBlindInt8ClusterWriter implements ClusterWriter {

    /**
     * The only code width this backend encodes: one unsigned byte per dimension. The absolute {@code maxAbs/127}
     * grid is fixed at 8 bits, so {@code docBits} is not a free parameter here — it is validated against this at
     * the write seam ({@code ClusterWriterFactory}) so the width recorded in {@code .clam} cannot drift from the
     * bytes actually written.
     */
    public static final int DOC_BITS = 8;

    private final int blockSize;
    private final int dimension;

    /**
     * Bytes one vector occupies in a code block: three int corrections (scale, sum, sqNorm) plus one unsigned
     * byte per dimension. So a block of {@code n} vectors is {@code n * bytesPerVector(dim)} bytes.
     */
    public static int bytesPerVector(final int dimension) {
        return 3 * Integer.BYTES + dimension;
    }

    /**
     * Vectors per code block for a field this writer stores: as many whole vectors as one storage block holds,
     * and at least one, so a vector wider than a storage block gets a code block of its own.
     *
     * @param storageBlockBytes bytes the storage layer fetches in one block
     * @param dimension         the field's vector dimension
     */
    public static int vectorsPerBlock(final int storageBlockBytes, final int dimension) {
        return Math.max(1, storageBlockBytes / bytesPerVector(dimension));
    }

    /**
     * Creates a cluster writer for this field's shape.
     *
     * @param storageBlockBytes bytes the storage layer fetches in one block; code blocks hold
     *     {@link #vectorsPerBlock(int, int)} vectors
     * @param dimension the field's vector dimension
     */
    public DataBlindInt8ClusterWriter(final int storageBlockBytes, final int dimension) {
        this.blockSize = vectorsPerBlock(storageBlockBytes, dimension);
        this.dimension = dimension;
    }

    @Override
    public ClusterRegion write(final IndexOutput out, final FloatVectorValues vectors, final ClusterMembers members) throws IOException {
        if (vectors.size() != members.ordinals().length) {
            throw new IllegalArgumentException("vectors (" + vectors.size() + ") must match members (" + members.ordinals().length + ")");
        }
        if (members.ordinals().length == 0) {
            throw new IllegalArgumentException("cluster has no members");
        }
        final long start = out.getFilePointer();

        writeHeader(out, members);
        new DataBlindInt8BlockWriter(out, blockSize, dimension).writeBlocks(vectors);

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
