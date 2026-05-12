/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.fielddata.LeafFieldData;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.fielddata.SortedBinaryDocValues;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.search.DocValueFormat;

import java.io.IOException;

/**
 * Per-segment leaf field data for KNN vector fields. Provides access to vector values
 * for scripting ({@code doc['field']}) and {@code docvalue_fields} in search responses.
 *
 * <p><b>Lifecycle and threading:</b> A new instance is created per segment per search request
 * via {@link KNNVectorIndexFieldData#load(org.apache.lucene.index.LeafReaderContext)}.
 * Each search thread owns its own instance — there is no sharing across threads.
 * The internal {@link KNNVectorValues} iterator is therefore safe to use without synchronization.
 *
 * <p><b>Initialization:</b> {@code fieldInfo} and {@code vectorValues} are eagerly initialized
 * in the constructor to avoid per-document overhead on the hot path. Since each instance is
 * single-threaded and short-lived (scoped to one segment within one search request), eager
 * initialization has no contention cost.
 */
public class KNNVectorDVLeafFieldData implements LeafFieldData {

    private final LeafReader reader;
    private final String fieldName;
    private final VectorDataType vectorDataType;
    private final FieldInfo fieldInfo;
    private final KNNVectorValues<?> vectorValues;

    public KNNVectorDVLeafFieldData(LeafReader reader, String fieldName, VectorDataType vectorDataType) {
        this.reader = reader;
        this.fieldName = fieldName;
        this.vectorDataType = vectorDataType;
        this.fieldInfo = reader.getFieldInfos().fieldInfo(fieldName);
        if (this.fieldInfo == null) {
            throw new IllegalStateException("Field info not found for field: " + fieldName);
        }
        try {
            this.vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, Lucene.segmentReader(reader));
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load vector values for field: " + fieldName, e);
        }
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public long ramBytesUsed() {
        return 0; // unknown
    }

    @Override
    public ScriptDocValues<?> getScriptValues() {
        try {
            KnnVectorValues knnVectorValues;
            if (fieldInfo.hasVectorValues()) {
                switch (fieldInfo.getVectorEncoding()) {
                    case FLOAT32:
                        knnVectorValues = reader.getFloatVectorValues(fieldName);
                        break;
                    case BYTE:
                        knnVectorValues = reader.getByteVectorValues(fieldName);
                        break;
                    default:
                        throw new IllegalStateException("Unsupported Lucene vector encoding: " + fieldInfo.getVectorEncoding());
                }
                return KNNVectorScriptDocValues.create(knnVectorValues, fieldName, vectorDataType);
            }
            DocIdSetIterator values = DocValues.getBinary(reader, fieldName);
            return KNNVectorScriptDocValues.create(values, fieldName, vectorDataType);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load values for knn vector field: " + fieldName, e);
        }
    }

    @Override
    public SortedBinaryDocValues getBytesValues() {
        throw new UnsupportedOperationException("knn vector field '" + fieldName + "' doesn't support sorting");
    }

    /**
     * Returns a {@link DocValueFetcher.Leaf} that reads vector values directly from the
     * KNN vector index or binary doc values, bypassing {@code _source} entirely.
     *
     * <p>This powers {@code "docvalue_fields": ["my_vector_field"]} in search requests.
     * Unlike the DerivedSource path (which deserializes the entire {@code _source}, injects
     * the vector, and re-serializes), this reads the vector with a single seek — zero
     * {@code _source} parsing overhead.
     *
     * <p><b>Threading:</b> The returned {@link DocValueFetcher.Leaf} captures the
     * {@code vectorValues} iterator from this instance. Since this instance is created
     * per segment per search request and owned by a single thread, the iterator is never
     * shared across threads.
     *
     * <p><b>Return types:</b>
     * <ul>
     *   <li>{@code float[]} for FLOAT vectors — {@link org.opensearch.core.xcontent.XContentBuilder}
     *       has a registered writer that serializes this directly as a JSON numeric array.</li>
     *   <li>{@code short[]} for BYTE/BINARY vectors — converted from {@code byte[]} because
     *       {@code XContentBuilder} would serialize raw {@code byte[]} as base64 binary data
     *       instead of a numeric array.</li>
     * </ul>
     *
     * @param format the doc value format — determines output encoding (array or binary)
     * @return a leaf fetcher that yields vector values per document
     */
    @Override
    public DocValueFetcher.Leaf getLeafValueFetcher(final DocValueFormat format) {
        if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
            throw new UnsupportedOperationException(
                "docvalue_fields is not supported for [" + vectorDataType + "] vector field '" + fieldName + "'"
            );
        }

        final boolean binary = format instanceof KNNVectorDocValueFormat && ((KNNVectorDocValueFormat) format).isBinary();

        return new DocValueFetcher.Leaf() {
            private int count;

            @Override
            public boolean advanceExact(int docId) throws IOException {
                if (vectorValues.advance(docId) == docId) {
                    count = 1;
                    return true;
                }
                count = 0;
                return false;
            }

            @Override
            public int docValueCount() {
                return count;
            }

            @Override
            public Object nextValue() throws IOException {
                float[] vector = (float[]) vectorValues.getVector();
                if (binary) {
                    return KNNVectorDocValueFormat.encodeToBinary(vector);
                }
                return vector;
            }
        };
    }
}
