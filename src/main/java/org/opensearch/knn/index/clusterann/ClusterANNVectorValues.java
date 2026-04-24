/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Random-access float vector values for ClusterANN algorithms.
 * Extends Lucene's {@link FloatVectorValues} so it works with all Lucene APIs
 * (OptimizedScalarQuantizer, RandomVectorScorer, etc.).
 *
 * <p>Three storage modes:
 * <ul>
 *   <li><b>On-heap</b> — wraps {@code List<float[]>}. Used during flush.</li>
 *   <li><b>Merged</b> — reads from source segment readers via ordinal mapping. No temp file.</li>
 *   <li><b>Off-heap</b> — reads from IndexInput. Fallback for very large merges.</li>
 * </ul>
 */
public final class ClusterANNVectorValues extends FloatVectorValues {

    private final VectorSupplier supplier;
    private final int[] docIds;
    private final int numVectors;
    private final int dimension;

    private ClusterANNVectorValues(VectorSupplier supplier, int[] docIds, int numVectors, int dimension) {
        this.supplier = supplier;
        this.docIds = docIds;
        this.numVectors = numVectors;
        this.dimension = dimension;
    }

    // ========== FloatVectorValues contract ==========

    @Override
    public float[] vectorValue(int ord) throws IOException {
        return supplier.vector(ord);
    }

    @Override
    public int size() {
        return numVectors;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public ClusterANNVectorValues copy() throws IOException {
        return new ClusterANNVectorValues(supplier.copy(), docIds, numVectors, dimension);
    }

    @Override
    public VectorEncoding getEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public int ordToDoc(int ord) {
        return docIds == null ? ord : docIds[ord];
    }

    // ========== Factory: flush (on-heap) ==========

    /** Create from on-heap list. Zero copy. */
    public static ClusterANNVectorValues fromList(List<float[]> vectors, int[] docIds, int dim) {
        return new ClusterANNVectorValues(new OnHeapSupplier(vectors), docIds, vectors.size(), dim);
    }

    /** Convenience: no doc ID mapping. */
    public static ClusterANNVectorValues fromList(List<float[]> vectors, int dim) {
        return fromList(vectors, null, dim);
    }

    // ========== Factory: merge (source segment readers) ==========

    private static final int TEMP_FILE_THRESHOLD = 10_000_000;

    /**
     * Create from merge state — reads vectors from source segment .vec files.
     * No temp file for merges under 10M vectors.
     *
     * @param reservoirOut pre-allocated array for reservoir sample (may be null)
     */
    public static ClusterANNVectorValues fromMergeState(MergeState mergeState, FieldInfo fieldInfo, float[][] reservoirOut)
        throws IOException {
        int dimension = fieldInfo.getVectorDimension();

        FloatVectorValues[] segmentValues = new FloatVectorValues[mergeState.knnVectorsReaders.length];
        int totalSize = 0;
        for (int s = 0; s < segmentValues.length; s++) {
            if (mergeState.knnVectorsReaders[s] != null) {
                segmentValues[s] = mergeState.knnVectorsReaders[s].getFloatVectorValues(fieldInfo.name);
                if (segmentValues[s] != null) {
                    totalSize += segmentValues[s].size();
                }
            }
        }

        if (totalSize == 0) {
            return fromList(List.of(), null, dimension);
        }

        if (totalSize > TEMP_FILE_THRESHOLD) {
            return fromMergeViaTempFile(mergeState, fieldInfo, segmentValues, totalSize, dimension, reservoirOut);
        }

        // Collect (newDocId, segIndex, localOrd) from source segments
        int[] newDocIds = new int[totalSize];
        int[] segIndex = new int[totalSize];
        int[] localOrd = new int[totalSize];
        int count = 0;

        Random rng = reservoirOut != null ? new Random(42L) : null;
        int reservoirSize = reservoirOut != null ? reservoirOut.length : 0;

        for (int s = 0; s < segmentValues.length; s++) {
            if (segmentValues[s] == null) continue;
            var iter = segmentValues[s].iterator();
            for (int doc = iter.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iter.nextDoc()) {
                int newDocId = mergeState.docMaps[s].get(doc);
                if (newDocId == -1) continue;

                newDocIds[count] = newDocId;
                segIndex[count] = s;
                localOrd[count] = iter.index();

                if (reservoirOut != null) {
                    float[] vec = segmentValues[s].vectorValue(iter.index());
                    if (count < reservoirSize) {
                        reservoirOut[count] = vec.clone();
                    } else if (rng.nextInt(count + 1) < reservoirSize) {
                        reservoirOut[rng.nextInt(reservoirSize)] = vec.clone();
                    }
                }
                count++;
            }
        }

        sortParallel(newDocIds, segIndex, localOrd, count);

        if (count < totalSize) {
            newDocIds = Arrays.copyOf(newDocIds, count);
            segIndex = Arrays.copyOf(segIndex, count);
            localOrd = Arrays.copyOf(localOrd, count);
        }

        return new ClusterANNVectorValues(
            new MergedSegmentSupplier(segmentValues, segIndex, localOrd, dimension),
            newDocIds,
            count,
            dimension
        );
    }

    // ========== Factory: off-heap (large merge fallback) ==========

    /** Create from IndexInput. Vectors stored as contiguous little-endian floats. */
    public static ClusterANNVectorValues fromIndexInput(IndexInput input, int[] docIds, int numVectors, int dimension) {
        return new ClusterANNVectorValues(new OffHeapSupplier(input, dimension), docIds, numVectors, dimension);
    }

    private static ClusterANNVectorValues fromMergeViaTempFile(
        MergeState mergeState,
        FieldInfo fieldInfo,
        FloatVectorValues[] segmentValues,
        int totalSize,
        int dimension,
        float[][] reservoirOut
    ) throws IOException {
        FloatVectorValues merged = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        org.apache.lucene.store.IndexOutput tempOut = mergeState.segmentInfo.dir.createTempOutput(
            mergeState.segmentInfo.name,
            "clann_merge",
            org.apache.lucene.store.IOContext.DEFAULT
        );
        int[] docIds = new int[totalSize];
        Random rng = reservoirOut != null ? new Random(42L) : null;
        int reservoirSize = reservoirOut != null ? reservoirOut.length : 0;
        int count = 0;
        try {
            var iterator = merged.iterator();
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                float[] vec = merged.vectorValue(iterator.index());
                for (int d = 0; d < dimension; d++) {
                    tempOut.writeInt(Float.floatToIntBits(vec[d]));
                }
                docIds[count] = doc;
                if (reservoirOut != null) {
                    if (count < reservoirSize) {
                        reservoirOut[count] = vec.clone();
                    } else if (rng.nextInt(count + 1) < reservoirSize) {
                        reservoirOut[rng.nextInt(reservoirSize)] = vec.clone();
                    }
                }
                count++;
            }
        } finally {
            tempOut.close();
        }
        if (count < totalSize) {
            docIds = Arrays.copyOf(docIds, count);
        }
        IndexInput tempIn = mergeState.segmentInfo.dir.openInput(tempOut.getName(), org.apache.lucene.store.IOContext.DEFAULT);
        return new ClusterANNVectorValues(new OffHeapSupplier(tempIn, dimension), docIds, count, dimension);
    }

    // ========== Helpers ==========

    /** Returns a copy of the vector (safe to store). */
    public float[] vectorValueCopy(int ord) throws IOException {
        return supplier.vector(ord).clone();
    }

    private static void sortParallel(int[] keys, int[] vals1, int[] vals2, int count) {
        long[] packed = new long[count];
        for (int i = 0; i < count; i++) {
            packed[i] = ((long) keys[i] << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        int[] tk = new int[count];
        int[] tv1 = new int[count];
        int[] tv2 = new int[count];
        for (int i = 0; i < count; i++) {
            int origIdx = (int) packed[i];
            tk[i] = keys[origIdx];
            tv1[i] = vals1[origIdx];
            tv2[i] = vals2[origIdx];
        }
        System.arraycopy(tk, 0, keys, 0, count);
        System.arraycopy(tv1, 0, vals1, 0, count);
        System.arraycopy(tv2, 0, vals2, 0, count);
    }

    // ========== Vector suppliers ==========

    private interface VectorSupplier {
        float[] vector(int ord) throws IOException;

        VectorSupplier copy() throws IOException;
    }

    private static final class OnHeapSupplier implements VectorSupplier {
        private final List<float[]> vectors;

        OnHeapSupplier(List<float[]> vectors) {
            this.vectors = vectors;
        }

        @Override
        public float[] vector(int ord) {
            return vectors.get(ord);
        }

        @Override
        public VectorSupplier copy() {
            return this;
        }
    }

    private static final class OffHeapSupplier implements VectorSupplier {
        private final IndexInput input;
        private final float[] buffer;
        private final long vectorByteSize;

        OffHeapSupplier(IndexInput input, int dims) {
            this.input = input;
            this.buffer = new float[dims];
            this.vectorByteSize = (long) dims * Float.BYTES;
        }

        @Override
        public float[] vector(int ord) throws IOException {
            input.seek(ord * vectorByteSize);
            input.readFloats(buffer, 0, buffer.length);
            return buffer;
        }

        @Override
        public VectorSupplier copy() {
            return new OffHeapSupplier(input.clone(), buffer.length);
        }
    }

    private static final class MergedSegmentSupplier implements VectorSupplier {
        private final FloatVectorValues[] segmentValues;
        private final int[] segIndex;
        private final int[] localOrd;
        private final int dims;

        MergedSegmentSupplier(FloatVectorValues[] segmentValues, int[] segIndex, int[] localOrd, int dims) {
            this.segmentValues = segmentValues;
            this.segIndex = segIndex;
            this.localOrd = localOrd;
            this.dims = dims;
        }

        @Override
        public float[] vector(int ord) throws IOException {
            // Clone result — source readers return shared buffers
            return segmentValues[segIndex[ord]].vectorValue(localOrd[ord]).clone();
        }

        @Override
        public VectorSupplier copy() throws IOException {
            FloatVectorValues[] cloned = new FloatVectorValues[segmentValues.length];
            for (int i = 0; i < segmentValues.length; i++) {
                if (segmentValues[i] != null) {
                    cloned[i] = segmentValues[i].copy();
                }
            }
            return new MergedSegmentSupplier(cloned, segIndex, localOrd, dims);
        }
    }
}
