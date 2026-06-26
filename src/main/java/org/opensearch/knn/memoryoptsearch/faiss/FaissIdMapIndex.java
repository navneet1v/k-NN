/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissByteVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues;

import java.io.IOException;

/**
 * A FAISS index with an ID mapping that maps the internal vector ID to a logical ID, along with the actual vector index.
 * It first delegates the vector search to its nested vector index, then transforms the vector ID into a logical index that is
 * understandable by upstream components. This is particularly useful when not all Lucene documents are indexed with a vector field.
 * For example, if 70% of the documents have a vector field and the remaining 30% do not, the FAISS vector index will still assign
 * increasing and continuous vector IDs starting from 0.
 * However, these IDs only cover the sparse 30% of Lucene documents, so an ID mapping is needed to convert the internal physical vector ID
 * into the corresponding Lucene document ID.
 * If the mapping is an identity mapping, where each `i` is mapped to itself, we omit storing it to save memory.
 */
public class FaissIdMapIndex extends FaissBinaryIndex implements FaissHNSWProvider {
    public static final String IXMP = "IxMp";
    public static final String IBMP = "IBMp";

    @Getter
    private FaissIndex nestedIndex;
    private FaissHNSWProvider hnswGetter;
    private DirectMonotonicReader idMappingReader;

    public FaissIdMapIndex(final String indexType) {
        super(indexType);
    }

    /**
     * Partially load id mapping and its nested index to which vector searching will be delegated.
     * Faiss deserialization code :
     * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1088">IndexIDMap.h</a>
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        if (indexType.equals(IXMP)) {
            readCommonHeader(input);
        } else {
            readBinaryCommonHeader(input);
        }
        final FaissIndex nestedIndex = FaissIndex.load(input);

        if (nestedIndex instanceof AbstractFaissHNSWIndex || nestedIndex instanceof FaissBinaryHnswIndex) {
            this.nestedIndex = nestedIndex;
            this.hnswGetter = (FaissHNSWProvider) nestedIndex;
        } else {
            throw new IllegalStateException("Invalid nested HNSW index type, got index type=" + nestedIndex.getIndexType());
        }

        final int numElements = Math.toIntExact(input.readLong());

        // This is a mapping table that maps internal vector id to Lucene document id.
        // In dense case where all documents having at least one KNN field which also has exactly one vector, we don't need this.
        // However, in a sparse case where not all documents having KNN field, we need this mapping table to convert inner vector id to
        // Lucene document id.
        // Another case is parent-child nested case. In which, this mapping table will map internal vector id to parent document id.
        // NOTE : If the mapping is an identity function that maps `i` to `i`, then the reader will be null.
        idMappingReader = MonotonicIntegerSequenceEncoder.encode(numElements, input);
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return nestedIndex.getVectorEncoding();
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        if (idMappingReader == null) {
            // Handle 'dense' case where all documents have at least one KNN field, which has exactly one vector.
            // No re-mapping is required.
            return nestedIndex.getFloatValues(indexInput);
        }

        // Re-mapping is required.
        return sparseFloatValues(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        if (idMappingReader == null) {
            // Handle 'dense' case where all documents have at least one KNN field, which has exactly one vector.
            // No re-mapping is required.
            return nestedIndex.getByteValues(indexInput);
        }

        // Re-mapping is required.
        final ByteVectorValues vectorValues = nestedIndex.getByteValues(indexInput);
        if (indexType.equals(IBMP)) {
            return new FaissByteVectorValues.SparseBinaryVectorValuesImpl(vectorValues, idMappingReader, codeSize);
        }
        return new FaissByteVectorValues.SparseByteVectorValuesImpl(vectorValues, idMappingReader, codeSize);
    }

    private FloatVectorValues sparseFloatValues(final IndexInput indexInput) throws IOException {
        final FloatVectorValues vectorValues = nestedIndex.getFloatValues(indexInput);
        return new FaissFloatVectorValues.SparseFloatVectorValuesImpl(vectorValues, idMappingReader);
    }

    @Override
    public FaissHNSW getFaissHnsw() {
        return hnswGetter.getFaissHnsw();
    }
}
