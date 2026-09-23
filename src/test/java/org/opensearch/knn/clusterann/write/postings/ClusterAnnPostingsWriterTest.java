/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.opensearch.knn.clusterann.read.block.scalar.ScalarEncoding;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer.QuantizationResult;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.write.QuantizationParams;
import org.opensearch.knn.clusterann.ClusteringResult;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_NONE;
import static org.opensearch.knn.clusterann.format.ClusterANNFormatConstants.ROTATION_RANDOM_GAUSSIAN;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end component test for the {@code .clap} write path: postings writer -&gt; cluster writer -&gt; real
 * block writer. Reads the whole file back and checks, per cluster, the header (ordinals nearest-first,
 * secondary bitset, distances) and every code block's struct-of-arrays bytes against an independent reference
 * quantization of the member at that position. Covers the unrotated path and the rotated (L2) path, where the
 * rotation must be applied to both the vectors and the centroid before quantization.
 */
class ClusterAnnPostingsWriterTest {

    private static final int DIMENSION = 2;
    private static final int BLOCK_SIZE = 2;
    private static final int DOC_BITS = 1;
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;
    private static final ScalarEncoding ENCODING = ScalarEncoding.fromNumBits(DOC_BITS);
    private static final int DISCRETE_DIMS = ENCODING.getDiscreteDimensions(DIMENSION);
    private static final int CODE_LENGTH = ENCODING.getDocPackedLength(DISCRETE_DIMS);

    private static final float[][] VECTORS = { { 0f, 0f }, { 1f, 0f }, { 0f, 1f }, { 1f, 1f } };
    private static final float[][] CENTROIDS = { { 0f, 0f }, { 1f, 1f } };

    /** primary: v0,v1 -> c0 ; v2,v3 -> c1. secondary: v0 -> c1 ; v3 -> c0. */
    private static ClusteringResult clustering() {
        return new ClusteringResult(
            CENTROIDS,
            new int[] { 0, 0, 1, 1 },
            new float[] { 0f, 1f, 1f, 0f },
            new int[] { 1, -1, -1, 0 },
            new float[] { 2f, Float.NaN, Float.NaN, 3f },
            2
        );
    }

    @Test
    void write_unrotated_framesAndQuantizesEveryCluster() throws IOException {
        final PostingsRegions layout;
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
                layout = new ClusterAnnPostingsWriter(BLOCK_SIZE, quantization(), RotationFormats.create(ROTATION_NONE, DIMENSION)).write(
                    out,
                    clustering(),
                    source(),
                    METRIC
                );
            }

            // Two clusters were written: two distinct, non-empty regions laid out back to back, together
            // spanning the whole .clap.
            assertEquals(0L, layout.clapOffset());
            assertEquals(2, layout.centroidOffsets().length, "one region per centroid");
            assertArrayEquals(new int[] { 3, 3 }, layout.clusterSizes());
            assertEquals(0L, layout.centroidOffsets()[0]);
            assertTrue(layout.centroidLengths()[0] > 0 && layout.centroidLengths()[1] > 0, "both clusters non-empty");
            assertEquals(
                layout.centroidOffsets()[0] + layout.centroidLengths()[0],
                layout.centroidOffsets()[1],
                "cluster 1 region immediately follows cluster 0"
            );
            assertEquals(
                layout.centroidLengths()[0] + layout.centroidLengths()[1],
                layout.clapLength(),
                "the two cluster regions span the whole .clap"
            );

            try (IndexInput in = dir.openInput("clap", IOContext.DEFAULT)) {
                // c0 sorted nearest-first: v0(0), v1(1), v3(3 secondary).
                verifyCluster(
                    in,
                    layout.centroidOffsets()[0],
                    new int[] { 0, 1, 3 },
                    new float[] { 0f, 1f, 3f },
                    new boolean[] { false, false, true },
                    CENTROIDS[0],
                    VECTORS
                );
                // c1 sorted nearest-first: v3(0), v2(1), v0(2 secondary).
                verifyCluster(
                    in,
                    layout.centroidOffsets()[1],
                    new int[] { 3, 2, 0 },
                    new float[] { 0f, 1f, 2f },
                    new boolean[] { false, false, true },
                    CENTROIDS[1],
                    VECTORS
                );
                assertEquals(in.length(), in.getFilePointer(), "no trailing bytes");
            }
        }
    }

    @Test
    void write_rotated_appliesRotationToVectorsAndCentroid() throws IOException {
        final Rotation rotation = RotationFormats.create(ROTATION_RANDOM_GAUSSIAN, DIMENSION);

        final PostingsRegions layout;
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
                layout = new ClusterAnnPostingsWriter(BLOCK_SIZE, quantization(), rotation).write(out, clustering(), source(), METRIC);
            }

            final float[][] rotatedVectors = new float[VECTORS.length][];
            for (int i = 0; i < VECTORS.length; i++) {
                rotatedVectors[i] = rotate(rotation, VECTORS[i]);
            }
            try (IndexInput in = dir.openInput("clap", IOContext.DEFAULT)) {
                verifyCluster(
                    in,
                    layout.centroidOffsets()[0],
                    new int[] { 0, 1, 3 },
                    new float[] { 0f, 1f, 3f },
                    new boolean[] { false, false, true },
                    rotate(rotation, CENTROIDS[0]),
                    rotatedVectors
                );
                verifyCluster(
                    in,
                    layout.centroidOffsets()[1],
                    new int[] { 3, 2, 0 },
                    new float[] { 0f, 1f, 2f },
                    new boolean[] { false, false, true },
                    rotate(rotation, CENTROIDS[1]),
                    rotatedVectors
                );
                assertEquals(in.length(), in.getFilePointer(), "no trailing bytes");
            }
        }
    }

    /** Reads one cluster region at {@code offset} and checks its header and every block against the reference. */
    private static void verifyCluster(
        final IndexInput in,
        final long offset,
        final int[] expectedOrdinals,
        final float[] expectedDistances,
        final boolean[] expectedSecondary,
        final float[] preparedCentroid,
        final float[][] preparedVectors
    ) throws IOException {
        final int n = expectedOrdinals.length;
        in.seek(offset);

        final int[] ordinals = new int[n];
        for (int i = 0; i < n; i++) {
            ordinals[i] = in.readInt();
        }
        assertArrayEquals(expectedOrdinals, ordinals, "ordinals nearest-first");

        final long[] words = new long[FixedBitSet.bits2words(n)];
        in.readLongs(words, 0, words.length);
        final FixedBitSet bitset = new FixedBitSet(words, n);
        for (int i = 0; i < n; i++) {
            assertEquals(expectedSecondary[i], bitset.get(i), "secondary bit " + i);
        }
        for (int i = 0; i < n; i++) {
            assertEquals(expectedDistances[i], Float.intBitsToFloat(in.readInt()), 0f, "distance " + i);
        }

        final OptimizedScalarQuantizer reference = new OptimizedScalarQuantizer(METRIC);
        for (int base = 0; base < n; base += BLOCK_SIZE) {
            final int len = Math.min(BLOCK_SIZE, n - base);
            final Expected[] expected = new Expected[len];
            for (int j = 0; j < len; j++) {
                expected[j] = quantize(reference, preparedVectors[ordinals[base + j]], preparedCentroid);
            }
            for (int j = 0; j < len; j++) {
                assertEquals(expected[j].lower, Float.intBitsToFloat(in.readInt()), "lower @" + (base + j));
            }
            for (int j = 0; j < len; j++) {
                assertEquals(expected[j].upper, Float.intBitsToFloat(in.readInt()), "upper @" + (base + j));
            }
            for (int j = 0; j < len; j++) {
                assertEquals(expected[j].add, Float.intBitsToFloat(in.readInt()), "add @" + (base + j));
            }
            for (int j = 0; j < len; j++) {
                assertEquals(expected[j].sum, in.readInt(), "sum @" + (base + j));
            }
            for (int j = 0; j < len; j++) {
                final byte[] code = new byte[CODE_LENGTH];
                in.readBytes(code, 0, CODE_LENGTH);
                assertArrayEquals(expected[j].code, code, "code @" + (base + j));
            }
        }
    }

    private static Expected quantize(final OptimizedScalarQuantizer q, final float[] vector, final float[] centroid) {
        final byte[] quantized = new byte[DISCRETE_DIMS];
        final QuantizationResult terms = q.scalarQuantize(vector.clone(), quantized, ENCODING.getBits(), centroid);
        final byte[] code = new byte[CODE_LENGTH];
        OptimizedScalarQuantizer.packAsBinary(quantized, code); // SINGLE_BIT_QUERY_NIBBLE
        return new Expected(
            terms.lowerInterval(),
            terms.upperInterval(),
            terms.additionalCorrection(),
            terms.quantizedComponentSum(),
            code
        );
    }

    private static float[] rotate(final Rotation rotation, final float[] vector) throws IOException {
        final float[] out = new float[vector.length];
        rotation.rotate(vector, out);
        return out;
    }

    private static FloatVectorValues source() {
        return FloatVectorValues.fromFloats(List.of(VECTORS[0], VECTORS[1], VECTORS[2], VECTORS[3]), DIMENSION);
    }

    private static QuantizationParams quantization() {
        return new QuantizationParams(QUANTIZER_OPTIMIZED_SQ, (byte) DOC_BITS);
    }

    private record Expected(float lower, float upper, float add, int sum, byte[] code) {
    }
}
