/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.List;

/**
 * Random-access vector values for ClusterANN algorithms (k-means, IVF, SOAR).
 *
 * <p>Supports two storage modes:
 * <ul>
 *   <li><b>On-heap</b> — wraps a {@code List<float[]>} from {@code FlatFieldVectorsWriter.getVectors()}.
 *       Zero copy, O(1) random access. Used during flush.</li>
 *   <li><b>Off-heap</b> — reads from an {@link IndexInput} where vectors are stored as contiguous
 *       little-endian floats. Used during merge when vectors are already on disk.</li>
 * </ul>
 *
 * <p>Optionally carries an ord-to-doc mapping for segments with deleted docs where
 * vector ordinals don't match document IDs.
 *
 * <p>Usage:
 * <pre>{@code
 * // Flush (on-heap, zero copy):
 * ClusterANNVectorValues values = ClusterANNVectorValues.fromList(
 *     flatFieldWriter.getVectors(), null, dimension);
 *
 * // Flush with deleted docs:
 * ClusterANNVectorValues values = ClusterANNVectorValues.fromList(vectors, docIds, dimension);
 *
 * // Merge (off-heap):
 * ClusterANNVectorValues values = ClusterANNVectorValues.fromIndexInput(
 *     vectorInput, null, numVectors, dimension);
 *
 * // Thread-safe copy for parallel k-means:
 * ClusterANNVectorValues threadLocal = values.copy();
 * }</pre>
 */
public final class ClusterANNVectorValues {

    private final VectorSupplier vectorSupplier;
    private final int[] docIds;
    private final int numVectors;
    private final int dimension;

    private ClusterANNVectorValues(VectorSupplier vectorSupplier, int[] docIds, int numVectors, int dimension) {
        this.vectorSupplier = vectorSupplier;
        this.docIds = docIds;
        this.numVectors = numVectors;
        this.dimension = dimension;
    }

    /**
     * Create from an on-heap list. Zero copy — the list is used directly.
     *
     * @param vectors   list of vectors (typically from {@code FlatFieldVectorsWriter.getVectors()})
     * @param docIds    ord-to-doc mapping, or {@code null} if ord == docId
     * @param dimension vector dimensionality
     */
    public static ClusterANNVectorValues fromList(List<float[]> vectors, int[] docIds, int dimension) {
        return new ClusterANNVectorValues(new OnHeapSupplier(vectors, dimension), docIds, vectors.size(), dimension);
    }

    /** Convenience overload without doc ID mapping. */
    public static ClusterANNVectorValues fromList(List<float[]> vectors, int dimension) {
        return fromList(vectors, null, dimension);
    }

    /**
     * Create from an off-heap {@link IndexInput}. Vectors must be stored as contiguous
     * little-endian floats ({@code dimension * Float.BYTES} per vector).
     *
     * @param input      the index input positioned at the start of vector data
     * @param docIds     ord-to-doc mapping, or {@code null} if ord == docId
     * @param numVectors total number of vectors
     * @param dimension  vector dimensionality
     */
    public static ClusterANNVectorValues fromIndexInput(IndexInput input, int[] docIds, int numVectors, int dimension) {
        return new ClusterANNVectorValues(new OffHeapSupplier(input, dimension), docIds, numVectors, dimension);
    }

    /**
     * Get vector at the given ordinal.
     * <p>On-heap: O(1), returns shared array — caller must not modify.
     * <p>Off-heap: reads from disk into a reusable buffer — not thread-safe.
     */
    public float[] vectorValue(int ord) throws IOException {
        return vectorSupplier.vector(ord);
    }

    /** Number of vectors. */
    public int size() {
        return numVectors;
    }

    /** Vector dimensionality. */
    public int dimension() {
        return dimension;
    }

    /**
     * Map vector ordinal to document ID.
     *
     * @return the document ID for the given ordinal, or {@code ord} if no mapping exists
     */
    public int ordToDoc(int ord) {
        return docIds == null ? ord : docIds[ord];
    }

    /** Returns a copy of the vector at the given ordinal. */
    public float[] vectorValueCopy(int ord) throws IOException {
        return vectorSupplier.vector(ord).clone();
    }

    /**
     * Create a thread-safe copy. On-heap instances share the backing list (immutable).
     * Off-heap instances clone the {@link IndexInput} so each thread has its own file pointer.
     */
    public ClusterANNVectorValues copy() {
        return new ClusterANNVectorValues(vectorSupplier.copy(), docIds, numVectors, dimension);
    }

    // ========== Vector suppliers ==========

    private interface VectorSupplier {
        float[] vector(int ord) throws IOException;

        int dims();

        VectorSupplier copy();
    }

    private static final class OnHeapSupplier implements VectorSupplier {
        private final List<float[]> vectors;
        private final int dims;

        OnHeapSupplier(List<float[]> vectors, int dims) {
            this.vectors = vectors;
            this.dims = dims;
        }

        @Override
        public float[] vector(int ord) {
            return vectors.get(ord);
        }

        @Override
        public int dims() {
            return dims;
        }

        @Override
        public VectorSupplier copy() {
            return this; // on-heap is inherently thread-safe (shared read-only list)
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
        public int dims() {
            return buffer.length;
        }

        @Override
        public VectorSupplier copy() {
            return new OffHeapSupplier(input.clone(), buffer.length);
        }
    }
}
