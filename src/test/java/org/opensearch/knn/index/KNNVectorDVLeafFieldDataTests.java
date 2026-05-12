/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.DocValueFormat;
import org.junit.Before;

import java.io.IOException;

public class KNNVectorDVLeafFieldDataTests extends KNNTestCase {

    private static final String MOCK_INDEX_FIELD_NAME = "test-index-field-name";
    private static final String MOCK_NUMERIC_INDEX_FIELD_NAME = "test-index-price";
    private static final float[] SAMPLE_VECTOR_DATA = new float[] { 1.0f, 2.0f };
    private LeafReaderContext leafReaderContext;
    private Directory directory;
    private DirectoryReader reader;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        createKNNVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        leafReaderContext = reader.getContext().leaves().get(0);
    }

    private void createKNNVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        knnDocument.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, SAMPLE_VECTOR_DATA, VectorSimilarityFunction.EUCLIDEAN));
        knnDocument.add(new NumericDocValuesField(MOCK_NUMERIC_INDEX_FIELD_NAME, 1000));
        writer.addDocument(knnDocument);
        writer.commit();
        writer.close();
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        reader.close();
        directory.close();
    }

    @SuppressWarnings("unchecked")
    public void testGetScriptValues() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        ScriptDocValues<float[]> scriptValues = (ScriptDocValues<float[]>) leafFieldData.getScriptValues();
        assertNotNull(scriptValues);
        assertTrue(scriptValues instanceof KNNVectorScriptDocValues);
    }

    @SuppressWarnings("unchecked")
    public void testGetScriptValuesWrongFieldName() {
        expectThrows(
            IllegalStateException.class,
            () -> new KNNVectorDVLeafFieldData(leafReaderContext.reader(), "invalid", VectorDataType.FLOAT)
        );
    }

    public void testGetScriptValuesWrongFieldType() {
        expectThrows(
            IllegalStateException.class,
            () -> new KNNVectorDVLeafFieldData(leafReaderContext.reader(), MOCK_NUMERIC_INDEX_FIELD_NAME, VectorDataType.FLOAT)
        );
    }

    public void testRamBytesUsed() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        assertEquals(0, leafFieldData.ramBytesUsed());
    }

    public void testGetBytesValues() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        expectThrows(UnsupportedOperationException.class, () -> leafFieldData.getBytesValues());
    }

    public void testGetLeafValueFetcher_floatVector_returnsCorrectValues() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);
        assertNotNull(leaf);

        assertTrue(leaf.advanceExact(0));
        assertEquals(1, leaf.docValueCount());
        Object value = leaf.nextValue();
        assertTrue(value instanceof float[]);
        float[] vector = (float[]) value;
        assertEquals(SAMPLE_VECTOR_DATA.length, vector.length);
        for (int i = 0; i < SAMPLE_VECTOR_DATA.length; i++) {
            assertEquals(SAMPLE_VECTOR_DATA[i], vector[i], 0.001f);
        }
    }

    public void testGetLeafValueFetcher_advanceExact_nonExistentDoc_returnsFalse() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);

        assertFalse(leaf.advanceExact(999));
        assertEquals(0, leaf.docValueCount());
    }

    public void testGetLeafValueFetcher_docValueCount_isOne() throws IOException {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);

        assertTrue(leaf.advanceExact(0));
        assertEquals(1, leaf.docValueCount());
    }

    public void testGetLeafValueFetcher_multipleDocuments_iteratesCorrectly() throws IOException {
        float[] vector1 = new float[] { 1.0f, 2.0f };
        float[] vector2 = new float[] { 3.0f, 4.0f };
        float[] vector3 = new float[] { 5.0f, 6.0f };

        Directory multiDocDir = newDirectory();
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(multiDocDir, conf);

        Document doc1 = new Document();
        doc1.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, vector1, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc1);

        Document doc2 = new Document();
        doc2.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, vector2, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc2);

        Document doc3 = new Document();
        doc3.add(new KnnFloatVectorField(MOCK_INDEX_FIELD_NAME, vector3, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc3);

        writer.commit();
        writer.close();

        DirectoryReader multiDocReader = DirectoryReader.open(multiDocDir);
        LeafReaderContext ctx = multiDocReader.getContext().leaves().get(0);

        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(ctx.reader(), MOCK_INDEX_FIELD_NAME, VectorDataType.FLOAT);
        DocValueFetcher.Leaf leaf = leafFieldData.getLeafValueFetcher(DocValueFormat.RAW);

        assertTrue(leaf.advanceExact(0));
        assertEquals(1, leaf.docValueCount());
        float[] result1 = (float[]) leaf.nextValue();
        assertArrayEquals(vector1, result1, 0.001f);

        assertTrue(leaf.advanceExact(1));
        assertEquals(1, leaf.docValueCount());
        float[] result2 = (float[]) leaf.nextValue();
        assertArrayEquals(vector2, result2, 0.001f);

        assertTrue(leaf.advanceExact(2));
        assertEquals(1, leaf.docValueCount());
        float[] result3 = (float[]) leaf.nextValue();
        assertArrayEquals(vector3, result3, 0.001f);

        multiDocReader.close();
        multiDocDir.close();
    }

    public void testGetLeafValueFetcher_byteVectorDataType_throwsUnsupportedOp() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.BYTE
        );
        UnsupportedOperationException ex = expectThrows(
            UnsupportedOperationException.class,
            () -> leafFieldData.getLeafValueFetcher(DocValueFormat.RAW)
        );
        assertTrue(ex.getMessage().contains("docvalue_fields is not supported"));
        assertTrue(ex.getMessage().contains("BYTE"));
    }

    public void testGetLeafValueFetcher_binaryVectorDataType_throwsUnsupportedOp() {
        KNNVectorDVLeafFieldData leafFieldData = new KNNVectorDVLeafFieldData(
            leafReaderContext.reader(),
            MOCK_INDEX_FIELD_NAME,
            VectorDataType.BINARY
        );
        UnsupportedOperationException ex = expectThrows(
            UnsupportedOperationException.class,
            () -> leafFieldData.getLeafValueFetcher(DocValueFormat.RAW)
        );
        assertTrue(ex.getMessage().contains("docvalue_fields is not supported"));
        assertTrue(ex.getMessage().contains("BINARY"));
    }
}
