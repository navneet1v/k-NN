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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.opensearch.knn.clusterann.write.block.scalar.OptimizedScalarQuantizedClusterWriter;
import org.opensearch.knn.clusterann.write.postings.ClusterWriter.ClusterMembers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies {@link OptimizedScalarQuantizedClusterWriter} lays out one cluster's {@code .clap} region: the header
 * (ordinals, secondary bitset, distances) reflects the members given nearest-first, and block position
 * {@code i} is the field vector at the header's {@code ordinals[i]}. The caller supplies members ordered
 * nearest-first (a permutation with ascending distances) and a view bounded to the cluster in that order.
 * Swept across cluster size vs block size, verifying every block's struct-of-arrays bytes match an
 * independent reference quantization of the member at that position.
 */
class OptimizedScalarQuantizedClusterWriterTest {

    private static final int DIMENSION = 8;
    private static final int DOC_BITS = 1; // SINGLE_BIT_QUERY_NIBBLE
    private static final VectorSimilarityFunction METRIC = VectorSimilarityFunction.EUCLIDEAN;
    private static final ScalarEncoding ENCODING = ScalarEncoding.fromNumBits(DOC_BITS);
    private static final int DISCRETE_DIMS = ENCODING.getDiscreteDimensions(DIMENSION);
    private static final int CODE_LENGTH = ENCODING.getDocPackedLength(DISCRETE_DIMS);

    private static OptimizedScalarQuantizedClusterWriter writer(final int blockSize, final float[] centroid) {
        return new OptimizedScalarQuantizedClusterWriter(blockSize, DIMENSION, METRIC, (byte) DOC_BITS, centroid);
    }

    @ParameterizedTest(name = "count={0}, blockSize={1}")
    @CsvSource({ "1, 12", "11, 12", "12, 12", "13, 12", "24, 12", "25, 12", "36, 12", "16, 16", "36, 16", "7, 5" })
    void writesEveryBlockShapeThroughTheRealBlockWriter(final int count, final int blockSize) throws IOException {
        final Random rnd = new Random(42);
        final float[][] vectors = new float[count][DIMENSION];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < DIMENSION; j++) {
                vectors[i][j] = rnd.nextFloat();
            }
        }
        final float[] centroid = new float[DIMENSION];
        Arrays.fill(centroid, 0.5f);

        // Members already nearest-first (the orchestrator's job): a shuffled permutation of ordinals with
        // ascending distances 1..count parallel to it; every 5th ordinal is a spill member.
        final Integer[] shuffled = new Integer[count];
        for (int i = 0; i < count; i++) {
            shuffled[i] = i;
        }
        Collections.shuffle(Arrays.asList(shuffled), new Random(7));
        final int[] ordinals = new int[count];
        final float[] distances = new float[count];
        final boolean[] secondary = new boolean[count];
        for (int i = 0; i < count; i++) {
            ordinals[i] = shuffled[i];
            distances[i] = i + 1;
            secondary[i] = ordinals[i] % 5 == 0;
        }
        final ClusterMembers members = new ClusterMembers(ordinals, distances, secondary);

        final float[] preparedCentroid = centroid.clone();
        final FloatVectorValues source = FloatVectorValues.fromFloats(toList(vectors), DIMENSION);
        final FloatVectorValues view = new PreparedFloatVectorValues(source, ordinals);

        try (Directory dir = new ByteBuffersDirectory()) {
            final ClusterWriter.ClusterRegion region;
            try (IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
                region = writer(blockSize, preparedCentroid).write(out, view, members);
            }

            try (IndexInput in = dir.openInput("clap", IOContext.DEFAULT)) {
                assertEquals(0L, region.startOffset(), "region starts at the output position");
                assertEquals(in.length(), region.length(), "returned length is the region size");
                final int[] headerOrdinals = new int[count];
                for (int i = 0; i < count; i++) {
                    headerOrdinals[i] = in.readInt();
                }
                assertArrayEquals(ordinals, headerOrdinals, "ordinals written in member order");

                final long[] words = new long[FixedBitSet.bits2words(count)];
                in.readLongs(words, 0, words.length);
                final FixedBitSet bitset = new FixedBitSet(words, count);
                float previous = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < count; i++) {
                    final float d = Float.intBitsToFloat(in.readInt());
                    assertEquals(distances[i], d, 0f, "distance at " + i);
                    previous = d;
                    assertEquals(ordinals[i] % 5 == 0, bitset.get(i), "secondary bit at position " + i);
                }
                assertEquals(count, previous, 0f, "distances ascending to count");

                final OptimizedScalarQuantizer reference = new OptimizedScalarQuantizer(METRIC);
                for (int base = 0; base < count; base += blockSize) {
                    final int n = Math.min(blockSize, count - base);
                    final Expected[] expected = new Expected[n];
                    for (int j = 0; j < n; j++) {
                        expected[j] = quantize(reference, vectors[headerOrdinals[base + j]], preparedCentroid);
                    }
                    for (int j = 0; j < n; j++) {
                        assertEquals(expected[j].lower, Float.intBitsToFloat(in.readInt()), "lower @" + (base + j));
                    }
                    for (int j = 0; j < n; j++) {
                        assertEquals(expected[j].upper, Float.intBitsToFloat(in.readInt()), "upper @" + (base + j));
                    }
                    for (int j = 0; j < n; j++) {
                        assertEquals(expected[j].add, Float.intBitsToFloat(in.readInt()), "add @" + (base + j));
                    }
                    for (int j = 0; j < n; j++) {
                        assertEquals(expected[j].sum, in.readInt(), "sum @" + (base + j));
                    }
                    for (int j = 0; j < n; j++) {
                        final byte[] code = new byte[CODE_LENGTH];
                        in.readBytes(code, 0, CODE_LENGTH);
                        assertArrayEquals(expected[j].code, code, "code @" + (base + j));
                    }
                }
                assertEquals(in.length(), in.getFilePointer(), "no trailing bytes");
            }
        }
    }

    @Test
    void rejectsBoundedViewSizeNotMatchingMembers() throws IOException {
        final FloatVectorValues source = FloatVectorValues.fromFloats(List.of(new float[DIMENSION], new float[DIMENSION]), DIMENSION);
        final FloatVectorValues view = new PreparedFloatVectorValues(source, new int[] { 0, 1 }); // size 2
        final ClusterMembers members = new ClusterMembers(
            new int[] { 0, 1, 0 },
            new float[] { 1f, 2f, 3f },
            new boolean[] { false, false, true }
        );
        try (Directory dir = new ByteBuffersDirectory(); IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
            assertThrows(IllegalArgumentException.class, () -> writer(4, new float[DIMENSION]).write(out, view, members));
        }
    }

    @Test
    void emptyClusterThrows() throws IOException {
        // Clustering never emits a zero-member centroid (HierarchicalKMeans drops empties; an empty field
        // yields no centroids), so an empty cluster is a broken invariant and the writer rejects it.
        final ClusterMembers members = new ClusterMembers(new int[0], new float[0], new boolean[0]);
        final FloatVectorValues view = new PreparedFloatVectorValues(FloatVectorValues.fromFloats(List.of(), DIMENSION), new int[0]);

        try (Directory dir = new ByteBuffersDirectory(); IndexOutput out = dir.createOutput("clap", IOContext.DEFAULT)) {
            assertThrows(IllegalArgumentException.class, () -> writer(4, new float[DIMENSION]).write(out, view, members));
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

    private static List<float[]> toList(final float[][] vectors) {
        final List<float[]> list = new ArrayList<>(vectors.length);
        for (final float[] v : vectors) {
            list.add(v);
        }
        return list;
    }

    private record Expected(float lower, float upper, float add, int sum, byte[] code) {
    }
}
