/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040LocalityAwareSQVectorsFormat;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

/**
 * Milestone 0 proof: vectors written through {@link KNN1040LocalityAwareSQVectorsFormat} round-trip
 * with zero corruption despite the internal physical shuffle. The write path is driven exactly as
 * the Lucene indexing chain drives it ({@code addField -> addValue -> flush -> finish -> close}),
 * so this also guards the {@code flush()}/{@code finish()} contract.
 *
 * <p>The quantized codes are checked against an independent, deterministic re-quantization of the
 * same float vectors using the centroid persisted in the file. Because the reader is addressed by
 * <b>original</b> ordinal, a wrong physical-permutation map would surface here as a code mismatch.
 */
public class LocalityOrderedQuantizedVectorsRoundTripTests extends KNNTestCase {

    private static final String FIELD_NAME = "test_field";
    private static final String SEGMENT_NAME = "_0";

    @SneakyThrows
    public void testInnerProductRoundTrip() {
        doRoundTrip(128, 100, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    @SneakyThrows
    public void testEuclideanRoundTrip() {
        doRoundTrip(128, 100, VectorSimilarityFunction.EUCLIDEAN);
    }

    @SneakyThrows
    public void testNonMultipleOf8Dimension() {
        // 56 -> 7 packed bytes; exercises the remainder path in the record layout
        doRoundTrip(56, 64, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    @SneakyThrows
    private void doRoundTrip(int dimension, int numVectors, VectorSimilarityFunction similarity) {
        final float[][] vectors = generateRandomVectors(numVectors, dimension);
        final int maxDoc = numVectors;
        final byte[] segmentId = StringHelper.randomId();

        try (Directory directory = newDirectory()) {
            final FieldInfo fieldInfo = createFieldInfo(similarity, dimension);
            final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            final SegmentInfo segmentInfo = new SegmentInfo(
                directory,
                Version.LATEST,
                Version.LATEST,
                SEGMENT_NAME,
                maxDoc,
                false,
                false,
                null,
                Collections.emptyMap(),
                segmentId,
                Collections.emptyMap(),
                null
            );
            final SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                directory,
                segmentInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );

            final KNN1040LocalityAwareSQVectorsFormat format = new KNN1040LocalityAwareSQVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);

            // Step 1: write the vectors through the format, mirroring the indexing chain's lifecycle.
            try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
                @SuppressWarnings({ "unchecked", "rawtypes" })
                final FlatFieldVectorsWriter fieldWriter = writer.addField(fieldInfo);
                for (int i = 0; i < numVectors; i++) {
                    fieldWriter.addValue(i, vectors[i]);
                }
                writer.flush(maxDoc, null);
                writer.finish();
            }

            final SegmentReadState readState = new SegmentReadState(
                writeState.directory,
                writeState.segmentInfo,
                writeState.fieldInfos,
                writeState.context,
                writeState.segmentSuffix
            );

            // Step 2: read it back and assert the store round-trips exactly.
            try (FlatVectorsReader flatReader = format.fieldsReader(readState)) {
                final LocalityOrderedQuantizedVectorsReader reader = (LocalityOrderedQuantizedVectorsReader) flatReader;
                final QuantizedByteVectorValues readValues = (QuantizedByteVectorValues) reader.getQuantizedVectorValues(FIELD_NAME);
                assertEquals(numVectors, readValues.size());

                // full-precision access delegates to the raw .vec reader
                final FloatVectorValues rawFloats = reader.getFloatVectorValues(FIELD_NAME);
                assertEquals(numVectors, rawFloats.size());

                // centroidDp must be self-consistent with the persisted centroid
                final float[] centroid = readValues.getCentroid();
                assertEquals(VectorUtil.dotProduct(centroid, centroid), readValues.getCentroidDP(), 1e-4f);

                // Independent, deterministic re-quantization oracle (same centroid the writer used).
                final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarity);
                final byte bits = SINGLE_BIT_QUERY_NIBBLE.getBits();
                final byte[] scratch = new byte[SINGLE_BIT_QUERY_NIBBLE.getDiscreteDimensions(dimension)];
                final byte[] expectedPacked = new byte[SINGLE_BIT_QUERY_NIBBLE.getDocPackedLength(dimension)];

                for (int ord = 0; ord < numVectors; ord++) {
                    final float[] input = prepareForQuantize(vectors[ord], similarity);
                    final OptimizedScalarQuantizer.QuantizationResult expected =
                        quantizer.scalarQuantize(input, scratch, bits, centroid);
                    OptimizedScalarQuantizer.packAsBinary(scratch, expectedPacked);

                    final byte[] actualCode = Arrays.copyOf(readValues.vectorValue(ord), expectedPacked.length);
                    assertArrayEquals("code mismatch for ordinal " + ord, expectedPacked, actualCode);

                    final OptimizedScalarQuantizer.QuantizationResult actual = readValues.getCorrectiveTerms(ord);
                    assertEquals(expected.lowerInterval(), actual.lowerInterval(), 0.0f);
                    assertEquals(expected.upperInterval(), actual.upperInterval(), 0.0f);
                    assertEquals(expected.additionalCorrection(), actual.additionalCorrection(), 0.0f);
                    assertEquals(expected.quantizedComponentSum(), actual.quantizedComponentSum());

                    // docs were added dense (docId == ordinal); docId is resolved via the raw reader
                    assertEquals("ordToDoc must equal ordinal for dense docs", ord, readValues.ordToDoc(ord));
                }

                // Iterator must yield docIDs ascending, index() must map back to the current docID,
                // and cover every vector.
                final KnnVectorValues.DocIndexIterator it = readValues.iterator();
                int expectedDoc = 0;
                for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                    assertEquals("docIDs must be dense-ascending", expectedDoc, doc);
                    assertEquals("index() -> docId must match ordToDoc", doc, readValues.ordToDoc(it.index()));
                    expectedDoc++;
                }
                assertEquals("iterator must visit every vector", numVectors, expectedDoc);

                // advance(target) must land on the target doc (docs are dense).
                final KnnVectorValues.DocIndexIterator adv = readValues.iterator();
                final int target = numVectors / 2;
                assertEquals("advance should land on the target doc", target, adv.advance(target));
                assertEquals(target, adv.docID());
                assertEquals("advanced index() must map back to the landed docId", target, readValues.ordToDoc(adv.index()));
            }
        }
    }

    /** Reproduces the exact vector the writer hands to the quantizer (normalized for COSINE). */
    private static float[] prepareForQuantize(float[] vector, VectorSimilarityFunction similarity) {
        final float[] copy = Arrays.copyOf(vector, vector.length);
        if (similarity == VectorSimilarityFunction.COSINE) {
            VectorUtil.l2normalize(copy);
        }
        return copy;
    }

    private static FieldInfo createFieldInfo(VectorSimilarityFunction similarityFunction, int dimension) {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0,
            0,
            0,
            dimension,
            VectorEncoding.FLOAT32,
            similarityFunction,
            false,
            false
        );
    }

    private static float[][] generateRandomVectors(int numVectors, int dimension) {
        final Random random = new Random(42);
        final float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }

    // ---------------------------------------------------------------------------------------------
    // Parity with Lucene's own scalar-quantized codec: the quantized record for a given doc must be
    // byte-for-byte identical between Lucene104ScalarQuantizedVectorsFormat and our locality format
    // (our physical shuffle must be invisible when the values are addressed by ordinal / walked by
    // ascending doc).
    // ---------------------------------------------------------------------------------------------

    @SneakyThrows
    public void testQuantizedParityWithLuceneInnerProduct() {
        doParityWithLucene(128, 100, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    @SneakyThrows
    public void testQuantizedParityWithLuceneEuclidean() {
        doParityWithLucene(128, 100, VectorSimilarityFunction.EUCLIDEAN);
    }

    @SneakyThrows
    public void testQuantizedParityWithLuceneCosine() {
        doParityWithLucene(128, 100, VectorSimilarityFunction.COSINE);
    }

    @SneakyThrows
    public void testQuantizedParityWithLuceneNonMultipleOf8() {
        doParityWithLucene(56, 64, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    @SneakyThrows
    private void doParityWithLucene(int dimension, int numVectors, VectorSimilarityFunction similarity) {
        final float[][] vectors = generateRandomVectors(numVectors, dimension);
        final FieldInfo fieldInfo = createFieldInfo(similarity, dimension);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        final FlatVectorsFormat luceneFormat = new Lucene104ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        final FlatVectorsFormat localityFormat = new KNN1040LocalityAwareSQVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);

        // Separate directories: both formats write a raw .vec for the same segment, which would
        // otherwise collide on the same file name.
        try (Directory luceneDir = newDirectory(); Directory localityDir = newDirectory()) {
            final SegmentWriteState luceneWs = newWriteState(luceneDir, fieldInfos, numVectors);
            final SegmentWriteState localityWs = newWriteState(localityDir, fieldInfos, numVectors);

            writeVectors(luceneFormat, luceneWs, fieldInfo, vectors);
            writeVectors(localityFormat, localityWs, fieldInfo, vectors);

            try (
                FlatVectorsReader luceneReader = luceneFormat.fieldsReader(toReadState(luceneWs));
                FlatVectorsReader localityReader = localityFormat.fieldsReader(toReadState(localityWs))
            ) {
                final QuantizedByteVectorValues luceneQ = (QuantizedByteVectorValues) ((QuantizedVectorsReader) luceneReader)
                    .getQuantizedVectorValues(FIELD_NAME);
                final QuantizedByteVectorValues localityQ = (QuantizedByteVectorValues) ((LocalityOrderedQuantizedVectorsReader) localityReader)
                    .getQuantizedVectorValues(FIELD_NAME);

                assertEquals("vector count mismatch", luceneQ.size(), localityQ.size());
                assertArrayEquals("centroid mismatch", luceneQ.getCentroid(), localityQ.getCentroid(), 0.0f);
                assertEquals("centroidDP mismatch", luceneQ.getCentroidDP(), localityQ.getCentroidDP(), 0.0f);

                // Walk both readers in ascending doc order and compare the quantized record per doc.
                final KnnVectorValues.DocIndexIterator luceneIt = luceneQ.iterator();
                final KnnVectorValues.DocIndexIterator localityIt = localityQ.iterator();
                int seen = 0;
                int luceneDoc = luceneIt.nextDoc();
                int localityDoc = localityIt.nextDoc();
                while (luceneDoc != DocIdSetIterator.NO_MORE_DOCS) {
                    assertEquals("doc order mismatch", luceneDoc, localityDoc);

                    final int luceneOrd = luceneIt.index();
                    final int localityOrd = localityIt.index();

                    final byte[] luceneCode = luceneQ.vectorValue(luceneOrd);
                    final byte[] localityCode = localityQ.vectorValue(localityOrd);
                    assertArrayEquals(
                        "quantized code mismatch at doc " + luceneDoc,
                        Arrays.copyOf(luceneCode, luceneCode.length),
                        Arrays.copyOf(localityCode, localityCode.length)
                    );

                    final OptimizedScalarQuantizer.QuantizationResult luceneCorr = luceneQ.getCorrectiveTerms(luceneOrd);
                    final OptimizedScalarQuantizer.QuantizationResult localityCorr = localityQ.getCorrectiveTerms(localityOrd);
                    assertEquals(luceneCorr.lowerInterval(), localityCorr.lowerInterval(), 0.0f);
                    assertEquals(luceneCorr.upperInterval(), localityCorr.upperInterval(), 0.0f);
                    assertEquals(luceneCorr.additionalCorrection(), localityCorr.additionalCorrection(), 0.0f);
                    assertEquals(luceneCorr.quantizedComponentSum(), localityCorr.quantizedComponentSum());

                    assertEquals("docId mismatch", luceneQ.ordToDoc(luceneOrd), localityQ.ordToDoc(localityOrd));

                    luceneDoc = luceneIt.nextDoc();
                    localityDoc = localityIt.nextDoc();
                    seen++;
                }
                assertEquals("locality reader must be exhausted too", DocIdSetIterator.NO_MORE_DOCS, localityDoc);
                assertEquals("must compare every doc", numVectors, seen);
            }
        }
    }

    private static SegmentWriteState newWriteState(Directory directory, FieldInfos fieldInfos, int maxDoc) {
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
        return new SegmentWriteState(InfoStream.NO_OUTPUT, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);
    }

    private static SegmentReadState toReadState(SegmentWriteState writeState) {
        return new SegmentReadState(
            writeState.directory,
            writeState.segmentInfo,
            writeState.fieldInfos,
            writeState.context,
            writeState.segmentSuffix
        );
    }

    private static void writeVectors(FlatVectorsFormat format, SegmentWriteState writeState, FieldInfo fieldInfo, float[][] vectors)
        throws IOException {
        try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
            @SuppressWarnings({ "unchecked", "rawtypes" })
            final FlatFieldVectorsWriter fieldWriter = writer.addField(fieldInfo);
            for (int i = 0; i < vectors.length; i++) {
                fieldWriter.addValue(i, vectors[i]);
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Score parity with Lucene's scalar-quantized codec: for a query, the score our locality reader
    // computes for a given ordinal must match Lucene104's, whether our reader takes the native SIMD
    // path (mmap-backed slice) or the Java fallback (heap-backed slice). This guards the
    // PhysicalOrdinalTranslatingScorer + physical-addressed-view fix on BOTH scorer paths.
    // ---------------------------------------------------------------------------------------------

    private static final int NUM_QUERIES = 5;

    @SneakyThrows
    public void testScoreParitySimdInnerProduct() {
        doScoreParity(128, 100, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, true);
    }

    @SneakyThrows
    public void testScoreParitySimdEuclidean() {
        doScoreParity(128, 100, VectorSimilarityFunction.EUCLIDEAN, true);
    }

    @SneakyThrows
    public void testScoreParityNonSimdInnerProduct() {
        doScoreParity(128, 100, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, false);
    }

    @SneakyThrows
    public void testScoreParityNonSimdEuclidean() {
        doScoreParity(128, 100, VectorSimilarityFunction.EUCLIDEAN, false);
    }

    @SneakyThrows
    private void doScoreParity(int dimension, int numVectors, VectorSimilarityFunction similarity, boolean useMmap) {
        final float[][] vectors = generateRandomVectors(numVectors, dimension);
        final float[][] queries = generateRandomVectors(NUM_QUERIES, dimension, 7L);
        final FieldInfo fieldInfo = createFieldInfo(similarity, dimension);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        final FlatVectorsFormat luceneFormat = new Lucene104ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        final FlatVectorsFormat localityFormat = new KNN1040LocalityAwareSQVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);

        // The locality reader's directory backing decides the path: mmap -> native SIMD (slice is
        // MemorySegment-backed), heap (ByteBuffers) -> Java fallback.
        try (Directory luceneDir = newBackingDirectory(useMmap); Directory localityDir = newBackingDirectory(useMmap)) {
            final SegmentWriteState luceneWs = newWriteState(luceneDir, fieldInfos, numVectors);
            final SegmentWriteState localityWs = newWriteState(localityDir, fieldInfos, numVectors);

            writeVectors(luceneFormat, luceneWs, fieldInfo, vectors);
            writeVectors(localityFormat, localityWs, fieldInfo, vectors);

            try (
                FlatVectorsReader luceneReader = luceneFormat.fieldsReader(toReadState(luceneWs));
                FlatVectorsReader localityReader = localityFormat.fieldsReader(toReadState(localityWs))
            ) {
                // Assert the intended path is actually exercised: the locality reader's record slice
                // must be address-extractable (SIMD) iff we used an mmap directory.
                final IndexInput slice = ((QuantizedByteVectorValues) ((LocalityOrderedQuantizedVectorsReader) localityReader)
                    .getQuantizedVectorValues(FIELD_NAME)).getSlice();
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(slice, 0, slice.length());
                if (useMmap) {
                    assertNotNull("expected an mmap (SIMD-capable) slice", addressAndSize);
                } else {
                    assertNull("expected a heap (non-SIMD) slice", addressAndSize);
                }

                // SIMD vs Java may differ by rounding; Java-vs-Java (non-mmap) is exact.
                for (final float[] query : queries) {
                    final RandomVectorScorer luceneScorer = luceneReader.getRandomVectorScorer(FIELD_NAME, query);
                    final RandomVectorScorer localityScorer = localityReader.getRandomVectorScorer(FIELD_NAME, query);
                    for (int ord = 0; ord < numVectors; ord++) {
                        final float expected = luceneScorer.score(ord);
                        final float actual = localityScorer.score(ord);
                        final float delta = useMmap ? Math.max(1e-4f, Math.abs(expected) * 1e-4f) : 0.0f;
                        assertEquals("score mismatch at ord=" + ord + " (mmap=" + useMmap + ")", expected, actual, delta);
                    }
                }
            }
        }
    }

    private Directory newBackingDirectory(boolean useMmap) throws IOException {
        return useMmap ? new MMapDirectory(createTempDir()) : new ByteBuffersDirectory();
    }

    private static float[][] generateRandomVectors(int numVectors, int dimension, long seed) {
        final Random random = new Random(seed);
        final float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }
}
