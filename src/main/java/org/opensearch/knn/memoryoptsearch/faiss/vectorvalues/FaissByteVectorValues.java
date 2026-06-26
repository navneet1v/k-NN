/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.vectorvalues;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.packed.DirectMonotonicReader;

import java.io.IOException;

/**
 * A {@link ByteVectorValues} implementation that reads byte-encoded vectors directly from a FAISS
 * index section via an {@link IndexInput}.
 * <p>
 * Each vector is located by seeking to {@code internalVectorId * codeSize} within the backing
 * {@link IndexInput} and reading {@code codeSize} bytes. This is used for scalar-quantized vectors
 * where {@code codeSize} may differ from {@code dimension} (e.g., SQ8 uses 1 byte per dimension,
 * while SQfp16 uses 2 bytes per dimension).
 * <p>
 * A single reusable buffer is used across calls to {@link #vectorValue(int)}, so callers must
 * consume or copy the returned array before the next call.
 * <p>
 * Implements {@link HasIndexSlice} to expose the underlying {@link IndexInput} for prefetching
 * or direct access by native code.
 */
public class FaissByteVectorValues extends ByteVectorValues implements HasIndexSlice {
    private final IndexInput indexInput;
    private final byte[] buffer;
    private final int codeSize;
    private final int dimension;
    private final int totalNumberOfVectors;

    /**
     * @param indexInput           The {@link IndexInput} positioned at the start of the vector data section.
     * @param codeSize             The byte size of a single encoded vector (may differ from dimension for quantized vectors).
     * @param dimension            The logical vector dimension.
     * @param totalNumberOfVectors The total number of vectors in this section.
     */
    public FaissByteVectorValues(final IndexInput indexInput, int codeSize, int dimension, int totalNumberOfVectors) {
        this.indexInput = indexInput;
        this.codeSize = codeSize;
        this.dimension = dimension;
        this.totalNumberOfVectors = totalNumberOfVectors;
        this.buffer = new byte[codeSize];
    }

    /**
     * Return the vector value for the given vector ordinal which must be in [0, size() - 1],
     * otherwise IndexOutOfBoundsException is thrown. The returned array may be shared across calls.
     *
     * @return the vector value
     */
    @Override
    public byte[] vectorValue(int internalVectorId) throws IOException {
        final long offset = (long) internalVectorId * codeSize;
        indexInput.seek(offset);
        indexInput.readBytes(buffer, 0, codeSize);
        return buffer;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    /**
     * Returns the byte length of a single vector, which equals {@code codeSize}. Default lucene returns
     * dimension multiplied by float byte size, hence we need to override this method
     */
    @Override
    public int getVectorByteLength() {
        return codeSize;
    }

    @Override
    public int size() {
        return totalNumberOfVectors;
    }

    @Override
    public ByteVectorValues copy() {
        return new FaissByteVectorValues(indexInput.clone(), codeSize, dimension, totalNumberOfVectors);
    }

    /**
     * Returns an IndexInput from which to read this instance's values, or null if not available.
     */
    @Override
    public IndexInput getSlice() {
        return indexInput;
    }

    /**
     * A {@link ByteVectorValues} wrapper for sparse or nested cases that maps internal vector IDs
     * to Lucene document IDs via a {@link DirectMonotonicReader}.
     * <p>
     * Delegates vector reads to the wrapped {@link ByteVectorValues} and translates ordinals
     * in {@link #ordToDoc(int)} and {@link #getAcceptOrds(Bits)}.
     * <p>
     * This class does NOT implement {@link HasIndexSlice} because the underlying byte values
     * may apply reconstruction (e.g., scalar quantized vectors). For binary indices where raw
     * bytes are the final values, use {@link SparseBinaryVectorValuesImpl} instead.
     */
    public static class SparseByteVectorValuesImpl extends ByteVectorValues {

        private final ByteVectorValues byteVectorValues;
        private final DirectMonotonicReader idMappingReader;
        private final int oneVectorByteSize;

        public SparseByteVectorValuesImpl(final ByteVectorValues byteVectorValues, final DirectMonotonicReader idMappingReader, int oneVectorByteSize) {
            this.byteVectorValues = byteVectorValues;
            this.idMappingReader = idMappingReader;
            this.oneVectorByteSize = oneVectorByteSize;
        }


        @Override
        public byte[] vectorValue(int internalVectorId) throws IOException {
            return byteVectorValues.vectorValue(internalVectorId);
        }

        @Override
        public int dimension() {
            return byteVectorValues.dimension();
        }

        @Override
        public int ordToDoc(int internalVectorId) {
            return (int) idMappingReader.get(internalVectorId);
        }

        @Override
        public Bits getAcceptOrds(final Bits acceptDocs) {
            if (acceptDocs != null) {
                return new Bits() {
                    @Override
                    public boolean get(int internalVectorId) {
                        return acceptDocs.get((int) idMappingReader.get(internalVectorId));
                    }

                    @Override
                    public int length() {
                        return byteVectorValues.size();
                    }
                };
            }

            return null;
        }

        @Override
        public int size() {
            return byteVectorValues.size();
        }

        @Override
        public int getVectorByteLength() {
            return oneVectorByteSize;
        }

        @Override
        public ByteVectorValues copy() throws IOException {
            return new SparseByteVectorValuesImpl(byteVectorValues.copy(), idMappingReader, oneVectorByteSize);
        }
    }

    /**
     * A {@link ByteVectorValues} wrapper for sparse binary index cases that maps internal vector IDs
     * to Lucene document IDs via a {@link DirectMonotonicReader}.
     * <p>
     * This class implements {@link HasIndexSlice} because binary indices store raw bytes without
     * any reconstruction — direct memory segment access produces correct results.
     */
    public static class SparseBinaryVectorValuesImpl extends ByteVectorValues implements HasIndexSlice {

        private final ByteVectorValues byteVectorValues;
        private final DirectMonotonicReader idMappingReader;
        private final int oneVectorByteSize;

        public SparseBinaryVectorValuesImpl(final ByteVectorValues byteVectorValues, final DirectMonotonicReader idMappingReader, int oneVectorByteSize) {
            this.byteVectorValues = byteVectorValues;
            this.idMappingReader = idMappingReader;
            this.oneVectorByteSize = oneVectorByteSize;
        }


        @Override
        public byte[] vectorValue(int internalVectorId) throws IOException {
            return byteVectorValues.vectorValue(internalVectorId);
        }

        @Override
        public int dimension() {
            return byteVectorValues.dimension();
        }

        @Override
        public int ordToDoc(int internalVectorId) {
            return (int) idMappingReader.get(internalVectorId);
        }

        @Override
        public Bits getAcceptOrds(final Bits acceptDocs) {
            if (acceptDocs != null) {
                return new Bits() {
                    @Override
                    public boolean get(int internalVectorId) {
                        return acceptDocs.get((int) idMappingReader.get(internalVectorId));
                    }

                    @Override
                    public int length() {
                        return byteVectorValues.size();
                    }
                };
            }

            return null;
        }

        @Override
        public int size() {
            return byteVectorValues.size();
        }

        @Override
        public int getVectorByteLength() {
            return oneVectorByteSize;
        }

        @Override
        public ByteVectorValues copy() throws IOException {
            return new SparseBinaryVectorValuesImpl(byteVectorValues.copy(), idMappingReader, oneVectorByteSize);
        }

        @Override
        public IndexInput getSlice() {
            return ((HasIndexSlice) byteVectorValues).getSlice();
        }
    }

}
