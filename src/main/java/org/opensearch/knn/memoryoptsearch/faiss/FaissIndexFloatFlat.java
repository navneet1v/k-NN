/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Supplier;

/**
 * This index type represents the storage of flat float vectors in FAISS.
 * Each vector occupies a fixed size proportional to its dimension.
 * The total storage size is calculated as `4 * dimension * number_of_vectors`, where `4` is the size of a float.
 * Please refer to IndexFlatL2 and IndexFlatIp in <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexFlat.h">...</a>.
 */
@Log4j2
public class FaissIndexFloatFlat extends FaissIndex {
    // Flat format for L2 metric
    public static final String IXF2 = "IxF2";
    // Flat format for inner product metric
    public static final String IXFI = "IxFI";
    private static long BYTES_128 = 128 * 1024;

    private static final Map<String, Supplier<KNNVectorSimilarityFunction>> INDEX_TYPE_TO_INDEX_FLOAT_FLAT = Map.of(
        IXF2,
        () -> KNNVectorSimilarityFunction.EUCLIDEAN,
        IXFI,
        () -> KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
    );

    private FaissSection floatVectors;
    private long oneVectorByteSize;
    @Getter
    private final KNNVectorSimilarityFunction vectorSimilarityFunction;

    public FaissIndexFloatFlat(final String indexType) {
        super(indexType);

        vectorSimilarityFunction = INDEX_TYPE_TO_INDEX_FLOAT_FLAT.getOrDefault(indexType, () -> {
            throw new IllegalStateException("Faiss index float flat does not support the index type [" + indexType + "].");
        }).get();
    }

    /**
     * Partial load the flat float vector section which is dimension * sizeof(float) * total_number_of_vectors.
     * FYI FAISS - <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L537">...</a>
     *
     * @param input
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        readCommonHeader(input);
        oneVectorByteSize = (long) Float.BYTES * getDimension();
        floatVectors = new FaissSection(input, Float.BYTES);
        if (floatVectors.getSectionSize() != (getTotalNumberOfVectors() * oneVectorByteSize)) {
            throw new IllegalStateException(
                "Got an inconsistent bytes size of vector ["
                    + floatVectors.getSectionSize()
                    + "] "
                    + "when faissIndexFloatFlat.totalNumberOfVectors="
                    + getTotalNumberOfVectors()
                    + ", faissIndexFloatFlat.oneVectorByteSize="
                    + oneVectorByteSize
            );
        }
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public FloatVectorValues getFloatValues(final IndexInput indexInput) {
        @RequiredArgsConstructor
        class FloatVectorValuesImpl extends FloatVectorValues {
            final IndexInput indexInput;
            final float[] buffer = new float[dimension];

            @Override
            public float[] vectorValue(int internalVectorId) throws IOException {
                indexInput.seek(floatVectors.getBaseOffset() + internalVectorId * oneVectorByteSize);
                indexInput.readFloats(buffer, 0, buffer.length);
                return buffer;
            }

            public void prefetch(final int[] ordsToPrefetch, int numOrds) throws IOException {
                if (KNNFeatureFlags.isExactPrefetch()) {
                    prefetchExact(ordsToPrefetch, numOrds);
                } else {
                    prefetch128KB(ordsToPrefetch, numOrds);
                }
            }

            public void prefetch128KB(final int[] ordsToPrefetch, int numOrds) throws IOException {
                if (ordsToPrefetch == null || numOrds <= 1) return;

                // 1. calculate offset and prefetch immediately
                long[] offsets = new long[numOrds];
                for (int i = 0; i < numOrds; i++) {
                    offsets[i] = floatVectors.getBaseOffset() + ((long) ordsToPrefetch[i] * oneVectorByteSize);
                }
                Arrays.sort(offsets);
                int j = 0;
                for (int i = 1; i < numOrds; i++) {
                    if (offsets[i] - offsets[j] > BYTES_128) {
                        j++;
                        offsets[j] = offsets[i];
                    }
                }
                log.debug(
                    "Prefetching compressed ["
                        + j
                        + "] vectors but ords size ["
                        + ordsToPrefetch.length
                        + "], "
                        + "num of ords is ["
                        + numOrds
                        + " ] using 128KB prefetch size"
                );
                for (int i = 0; i <= j; i++) {
                    // prefetch with 128KB size, may be at some time we should make sure that the long value overflow
                    // doesn't happen
                    long length = Math.min(BYTES_128, indexInput.length() - offsets[i]);
                    indexInput.prefetch(offsets[i], length);
                    if (length != BYTES_128) {
                        break;
                    }
                }
            }

            public void prefetchExact(final int[] ordsToPrefetch, int numOrds) throws IOException {
                if (ordsToPrefetch == null || numOrds <= 1) return;

                // Calculate all offsets and sort
                long[] offsets = new long[numOrds];
                for (int i = 0; i < numOrds; i++) {
                    offsets[i] = floatVectors.getBaseOffset() + ((long) ordsToPrefetch[i] * oneVectorByteSize);
                }
                Arrays.sort(offsets);

                // Group vectors within 128KB ranges
                int[] groupStarts = new int[numOrds];
                int[] groupEnds = new int[numOrds];
                int groupCount = 0;

                groupStarts[0] = 0;
                int currentGroupStart = 0;

                for (int i = 1; i < numOrds; i++) {
                    if (offsets[i] - offsets[currentGroupStart] > BYTES_128) {
                        // Close current group
                        groupEnds[groupCount] = i - 1;
                        groupCount++;

                        // Start new group
                        groupStarts[groupCount] = i;
                        currentGroupStart = i;
                    }
                }
                // Close final group
                groupEnds[groupCount] = numOrds - 1;
                groupCount++;

                log.debug(
                    "Prefetching compressed ["
                        + groupCount
                        + "] vectors but ords size ["
                        + ordsToPrefetch.length
                        + "], "
                        + "num of ords is ["
                        + numOrds
                        + " ] using exact prefetch size"
                );

                // Prefetch each group with exact size
                for (int g = 0; g < groupCount; g++) {
                    long startOffset = offsets[groupStarts[g]];
                    long endOffset = offsets[groupEnds[g]] + oneVectorByteSize;
                    indexInput.prefetch(startOffset, endOffset - startOffset);
                }
            }

            @Override
            public FloatVectorValues copy() {
                return new FloatVectorValuesImpl(indexInput.clone());
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public int size() {
                return totalNumberOfVectors;
            }
        }

        return new FloatVectorValuesImpl(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support " + ByteVectorValues.class.getSimpleName());
    }
}
