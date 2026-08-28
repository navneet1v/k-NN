/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * End-to-end tests for {@link Faiss1040SQHNSWReorderedKnnVectorsFormat}: ingest real vectors, build the
 * native Faiss SQ HNSW graph (single-layer + BFS ordering), write the reordered locality store, and
 * verify (a) the on-disk file layout — the reordered store is written and the redundant ordinal-order
 * SQ quantized files are deleted — and (b) the vectors round-trip through the reader.
 *
 * <p>Driven directly against the writer/reader (no {@code IndexWriter}) so the native JNI build runs,
 * mirroring {@link Faiss1040ScalarQuantizedKnnVectorsWriterTests}.
 */
@Log4j2
public class Faiss1040SQHNSWReorderedKnnVectorsFormatTests extends KNNTestCase {

    private static final int DIMENSION = 128;
    private static final String FIELD_NAME = "test_field";
    private static final String PARAMETERS_JSON = "{"
        + "\"index_description\":\"BHNSW16,Flat\","
        + "\"spaceType\":\"innerproduct\","
        + "\"name\":\"hnsw\","
        + "\"data_type\":\"float\","
        + "\"parameters\":{"
        + "\"ef_search\":256,"
        + "\"ef_construction\":256,"
        + "\"m\":16,"
        + "\"encoder\":{\"name\":\"sq\",\"bits\":1}"
        + "}"
        + "}";

    public void testFormatName_thenSuccess() {
        assertEquals(
            Faiss1040SQHNSWReorderedKnnVectorsFormat.class.getSimpleName(),
            new Faiss1040SQHNSWReorderedKnnVectorsFormat().getName()
        );
    }

    /**
     * End-to-end ANN search: index vectors, then run {@code reader.search(...)} with a
     * {@link TopKnnCollector}. Uses each query = an indexed vector, so the query's own doc is its exact
     * nearest neighbor; a working hub-seeded, single-layer graph search over the quantized data should
     * return that doc in the top-k for the large majority of queries. This is the only test that
     * exercises the full search path (memory-optimized searcher + hub entry point + graph traversal +
     * reordered-store scoring).
     */
    @SneakyThrows
    public void testSearch_whenQueryIsIndexedVector_thenSelfRecallIsHigh() {
        final int numVectors = 200;
        final int k = 10;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);

        try (Directory directory = newDirectory()) {
            final SegmentWriteState writeState = createWriteState(directory, "_0", numVectors);
            final FieldInfo fi = createRealFieldInfo();
            final Faiss1040SQHNSWReorderedKnnVectorsFormat format = new Faiss1040SQHNSWReorderedKnnVectorsFormat();

            try (KnnVectorsWriter knnWriter = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                for (int i = 0; i < numVectors; i++) {
                    fw.addValue(i, vectors[i]);
                }
                knnWriter.flush(numVectors, null);
                knnWriter.finish();
            }

            // The search path resolves the .faiss engine file via SegmentInfo.files(); an IndexWriter
            // would populate this, so we do it manually for the direct-writer test.
            writeState.segmentInfo.setFiles(
                Arrays.stream(directory.listAll()).filter(f -> f.startsWith(writeState.segmentInfo.name)).collect(Collectors.toSet())
            );

            final SegmentReadState readState = new SegmentReadState(
                directory,
                writeState.segmentInfo,
                new FieldInfos(new FieldInfo[] { fi }),
                IOContext.DEFAULT,
                FIELD_NAME
            );
            try (KnnVectorsReader knnReader = format.fieldsReader(readState)) {
                final int numQueries = 20;
                int selfHits = 0;
                for (int q = 0; q < numQueries; q++) {
                    final int queryDoc = q * (numVectors / numQueries);
                    final float[] query = vectors[queryDoc];

                    final TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
                    knnReader.search(FIELD_NAME, query, collector, AcceptDocs.fromLiveDocs(null, numVectors));

                    final TopDocs topDocs = collector.topDocs();
                    assertTrue("search returned no results for query doc " + queryDoc, topDocs.scoreDocs.length > 0);
                    assertTrue("search returned more than k results", topDocs.scoreDocs.length <= k);
                    for (final ScoreDoc sd : topDocs.scoreDocs) {
                        if (sd.doc == queryDoc) {
                            selfHits++;
                            break;
                        }
                    }
                }
                assertTrue("self-recall too low: " + selfHits + "/" + numQueries, selfHits >= (int) (numQueries * 0.8));
            }
        }
    }

    /**
     * Ingest real vectors, flush, and assert the segment holds the reordered locality store
     * ({@code .veqlo}/{@code .vemlo}) and the raw vectors ({@code .vec}), while the ordinal-order SQ
     * quantized files ({@code .veq}/{@code .vemq}) have been deleted as redundant.
     */
    @SneakyThrows
    public void testFlush_thenDeletesRedundantSqQuantizedFiles() {
        final int numVectors = 345;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);

        try (Directory directory = newDirectory()) {
            final SegmentWriteState writeState = createWriteState(directory, "_0", numVectors);
            final FieldInfo fi = createRealFieldInfo();
            final Faiss1040SQHNSWReorderedKnnVectorsFormat format = new Faiss1040SQHNSWReorderedKnnVectorsFormat();

            try (KnnVectorsWriter knnWriter = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                for (int i = 0; i < numVectors; i++) {
                    fw.addValue(i, vectors[i]);
                }
                knnWriter.flush(numVectors, null);
                knnWriter.finish();
            }

            // Reordered locality store is written.
            assertFalse("expected a .veqlo file", filesWithExtension(directory, "veqlo").isEmpty());
            assertFalse("expected a .vemlo file", filesWithExtension(directory, "vemlo").isEmpty());
            // Raw vectors are kept for rescoring / merge re-quantization.
            assertFalse("expected a raw .vec file", filesWithExtension(directory, "vec").isEmpty());
            // The redundant ordinal-order SQ quantized files must be gone.
            assertEquals("ordinal-order .veq should be deleted", Collections.emptyList(), filesWithExtension(directory, "veq"));
            assertEquals("ordinal-order .vemq should be deleted", Collections.emptyList(), filesWithExtension(directory, "vemq"));
        }
    }

    /**
     * Ingest real vectors, flush, then read the vectors back through the format's reader and verify
     * they round-trip unchanged.
     */
    @SneakyThrows
    public void testFlush_whenRealVectorsIngested_thenVectorsRoundTrip() {
        final int numVectors = 345;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);

        try (Directory directory = newDirectory()) {
            final SegmentWriteState writeState = createWriteState(directory, "_0", numVectors);
            final FieldInfo fi = createRealFieldInfo();
            final Faiss1040SQHNSWReorderedKnnVectorsFormat format = new Faiss1040SQHNSWReorderedKnnVectorsFormat();

            try (KnnVectorsWriter knnWriter = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                for (int i = 0; i < numVectors; i++) {
                    fw.addValue(i, vectors[i]);
                }
                knnWriter.flush(numVectors, null);
                knnWriter.finish();
            }

            final SegmentReadState readState = new SegmentReadState(
                directory,
                writeState.segmentInfo,
                new FieldInfos(new FieldInfo[] { fi }),
                IOContext.DEFAULT,
                FIELD_NAME
            );
            try (KnnVectorsReader knnReader = format.fieldsReader(readState)) {
                final FloatVectorValues fvv = knnReader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(fvv);
                assertEquals(numVectors, fvv.size());
                assertEquals(DIMENSION, fvv.dimension());

                final KnnVectorValues.DocIndexIterator it = fvv.iterator();
                int count = 0;
                while (it.nextDoc() != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                    assertArrayEquals(vectors[count], fvv.vectorValue(it.index()), 0.0f);
                    count++;
                }
                assertEquals(numVectors, count);
            }
        }
    }

    /**
     * The metadata file's hub section (appended at the very end of {@value org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter#METADATA_EXTENSION})
     * round-trips: the reader returns the same hub ordinals the JNI build produced, and a matching-length
     * quantized-record buffer (recordSize bytes each). We access the internal reader via the format's
     * {@link Faiss1040SQHNSWReorderedKnnVectorsFormat#fieldsReader}: unwrap the wrapping
     * {@code Faiss1040SQHNSWReorderedReader} → {@code LocalityOrderedQuantizedVectorsReader} to reach the
     * hub accessors.
     */
    @SneakyThrows
    public void testFlush_thenHubSectionRoundTripsInMetadata() {
        final int numVectors = 345;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);

        try (Directory directory = newDirectory()) {
            final SegmentWriteState writeState = createWriteState(directory, "_0", numVectors);
            final FieldInfo fi = createRealFieldInfo();
            final Faiss1040SQHNSWReorderedKnnVectorsFormat format = new Faiss1040SQHNSWReorderedKnnVectorsFormat();

            try (KnnVectorsWriter knnWriter = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                for (int i = 0; i < numVectors; i++) {
                    fw.addValue(i, vectors[i]);
                }
                knnWriter.flush(numVectors, null);
                knnWriter.finish();
            }

            final SegmentReadState readState = new SegmentReadState(
                directory,
                writeState.segmentInfo,
                new FieldInfos(new FieldInfo[] { fi }),
                IOContext.DEFAULT,
                FIELD_NAME
            );
            try (KnnVectorsReader knnReader = format.fieldsReader(readState)) {
                // Reach the locality reader (the format wraps it inside Faiss1040SQHNSWReorderedReader).
                final java.lang.reflect.Field delegateField = knnReader.getClass().getSuperclass().getDeclaredField("flatVectorsReader");
                delegateField.setAccessible(true);
                final org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsReader localityReader =
                    (org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsReader) delegateField.get(knnReader);

                final int[] hubs = localityReader.getHubOrdinals(FIELD_NAME);

                // Native JNI populates up to ~32 hubs (capped by BuildStrategy); every entry must be a
                // valid original ordinal in [0, numVectors).
                assertTrue("expected some hubs, got " + hubs.length, hubs.length > 0);
                assertTrue("hubs.length <= 32, got " + hubs.length, hubs.length <= 32);
                for (final int hub : hubs) {
                    assertTrue("hub ordinal out of range: " + hub, hub >= 0 && hub < numVectors);
                }
                // No duplicates (hubs are top-K distinct nodes).
                final java.util.Set<Integer> distinct = new java.util.HashSet<>();
                for (final int hub : hubs) {
                    distinct.add(hub);
                }
                assertEquals("hubs must be distinct", hubs.length, distinct.size());

                // Selecting the closest hub for a query scores the stored hub records and returns one of
                // the hub ordinals (validates the metadata hub-record slice is scoreable).
                final int best = localityReader.selectBestHubOrdinal(FIELD_NAME, vectors[0]);
                assertTrue("selected hub must be one of the stored hubs", distinct.contains(best));
            }
        }
    }

    /**
     * A segment where only <b>some</b> docs carry the vector field (a sparse field). {@code maxDoc} is
     * larger than the number of vectors: we add values for every other doc id, leaving gaps. The graph
     * and reordered store are built over the vectors only (ordinals {@code 0..numVectors-1}), and the
     * reader serves just those docs — mapping each ordinal back to its original doc id via
     * {@code ordToDoc}. Verifies the vectors round-trip against their correct doc ids and the redundant
     * SQ quantized files are still deleted.
     */
    @SneakyThrows
    public void testFlush_whenSomeDocsHaveNoVector_thenSparseFieldRoundTrips() {
        final int maxDoc = 300;
        final int step = 2;                       // only even doc ids get a vector
        final int numVectors = (maxDoc + step - 1) / step;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);
        final Map<Integer, float[]> vectorByDoc = new HashMap<>();

        try (Directory directory = newDirectory()) {
            final SegmentWriteState writeState = createWriteState(directory, "_0", maxDoc);
            final FieldInfo fi = createRealFieldInfo();
            final Faiss1040SQHNSWReorderedKnnVectorsFormat format = new Faiss1040SQHNSWReorderedKnnVectorsFormat();

            try (KnnVectorsWriter knnWriter = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                int ord = 0;
                for (int docId = 0; docId < maxDoc; docId += step) {
                    fw.addValue(docId, vectors[ord]);
                    vectorByDoc.put(docId, vectors[ord]);
                    ord++;
                }
                assertEquals(numVectors, ord);
                knnWriter.flush(maxDoc, null);
                knnWriter.finish();
            }

            // Redundant SQ quantized files are deleted even for a sparse field.
            assertFalse("expected a .veqlo file", filesWithExtension(directory, "veqlo").isEmpty());
            assertEquals("ordinal-order .veq should be deleted", Collections.emptyList(), filesWithExtension(directory, "veq"));
            assertEquals("ordinal-order .vemq should be deleted", Collections.emptyList(), filesWithExtension(directory, "vemq"));

            final SegmentReadState readState = new SegmentReadState(
                directory,
                writeState.segmentInfo,
                new FieldInfos(new FieldInfo[] { fi }),
                IOContext.DEFAULT,
                FIELD_NAME
            );
            try (KnnVectorsReader knnReader = format.fieldsReader(readState)) {
                final FloatVectorValues fvv = knnReader.getFloatVectorValues(FIELD_NAME);
                assertNotNull(fvv);
                // Only the docs that actually have a vector are present.
                assertEquals(numVectors, fvv.size());
                assertEquals(DIMENSION, fvv.dimension());

                final KnnVectorValues.DocIndexIterator it = fvv.iterator();
                int count = 0;
                while (it.nextDoc() != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                    final int docId = it.docID();
                    // Every returned doc id must be one we actually wrote a vector for...
                    assertTrue("unexpected doc id " + docId, vectorByDoc.containsKey(docId));
                    // ...and its vector must round-trip (ordinal -> doc mapping preserved through the reorder).
                    assertArrayEquals(vectorByDoc.get(docId), fvv.vectorValue(it.index()), 0.0f);
                    count++;
                }
                assertEquals(numVectors, count);
            }
        }
    }

    /**
     * Merge 3 segments of 150 vectors each into one, then verify all 450 vectors round-trip and that the
     * merged segment again holds the reordered store with the redundant SQ quantized files deleted.
     * Exercises {@code mergeOneField} + the native build on the merged set + re-quantization from raw
     * {@code .vec} (old segments have no {@code .veq}).
     */
    @SneakyThrows
    public void testMergeOneField_when3SegmentsMerged_thenAllVectorsReadableAndSqFilesDeleted() {
        final int numSegments = 3;
        final int vectorsPerSegment = 150;
        final int totalVectors = numSegments * vectorsPerSegment;

        final float[][][] segmentVectors = new float[numSegments][][];
        for (int s = 0; s < numSegments; s++) {
            segmentVectors[s] = generateRandomVectors(vectorsPerSegment, DIMENSION);
        }

        final FieldInfo fi = createRealFieldInfo();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fi });
        final Faiss1040SQHNSWReorderedKnnVectorsFormat format = new Faiss1040SQHNSWReorderedKnnVectorsFormat();

        try (Directory directory = newDirectory()) {
            // Step 1: create the input segments via flush.
            final SegmentInfo[] segInfos = new SegmentInfo[numSegments];
            final KnnVectorsReader[] readers = new KnnVectorsReader[numSegments];
            for (int s = 0; s < numSegments; s++) {
                segInfos[s] = createSegmentInfo(directory, "_" + s, vectorsPerSegment);
                final SegmentWriteState ws = new SegmentWriteState(
                    InfoStream.NO_OUTPUT,
                    directory,
                    segInfos[s],
                    fieldInfos,
                    null,
                    IOContext.DEFAULT,
                    FIELD_NAME
                );
                try (KnnVectorsWriter knnWriter = format.fieldsWriter(ws)) {
                    @SuppressWarnings("unchecked")
                    final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                    for (int i = 0; i < vectorsPerSegment; i++) {
                        fw.addValue(i, segmentVectors[s][i]);
                    }
                    knnWriter.flush(vectorsPerSegment, null);
                    knnWriter.finish();
                }
                readers[s] = format.fieldsReader(new SegmentReadState(directory, segInfos[s], fieldInfos, IOContext.DEFAULT, FIELD_NAME));
            }

            // Step 2: build the MergeState over the input readers.
            final MergeState.DocMap[] docMaps = new MergeState.DocMap[numSegments];
            final int[] maxDocs = new int[numSegments];
            final FieldInfos[] perSegFieldInfos = new FieldInfos[numSegments];
            int docBase = 0;
            for (int s = 0; s < numSegments; s++) {
                final int base = docBase;
                docMaps[s] = docID -> base + docID;
                maxDocs[s] = vectorsPerSegment;
                perSegFieldInfos[s] = fieldInfos;
                docBase += vectorsPerSegment;
            }

            final SegmentInfo mergedSegInfo = createSegmentInfo(directory, "_merged", totalVectors);
            final MergeState mergeState = new MergeState(
                docMaps,
                mergedSegInfo,
                fieldInfos,
                null,
                null,
                null,
                null,
                perSegFieldInfos,
                new org.apache.lucene.util.Bits[numSegments],
                null,
                null,
                readers,
                maxDocs,
                InfoStream.NO_OUTPUT,
                Runnable::run,
                false,
                null
            );

            // Step 3: merge into the merged segment.
            final SegmentWriteState mergedWriteState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                directory,
                mergedSegInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT,
                FIELD_NAME
            );
            try (KnnVectorsWriter mergedWriter = format.fieldsWriter(mergedWriteState)) {
                mergedWriter.mergeOneField(fi, mergeState);
                mergedWriter.finish();
            }
            for (final KnnVectorsReader reader : readers) {
                reader.close();
            }

            // Merged segment: reordered store present, redundant SQ quantized files deleted.
            final List<String> mergedVeqlo = filesWithSegmentAndExtension(directory, "_merged", "veqlo");
            final List<String> mergedVemlo = filesWithSegmentAndExtension(directory, "_merged", "vemlo");
            assertFalse("merged segment should have a .veqlo", mergedVeqlo.isEmpty());
            assertFalse("merged segment should have a .vemlo", mergedVemlo.isEmpty());
            assertEquals(
                "merged .veq should be deleted",
                Collections.emptyList(),
                filesWithSegmentAndExtension(directory, "_merged", "veq")
            );
            assertEquals(
                "merged .vemq should be deleted",
                Collections.emptyList(),
                filesWithSegmentAndExtension(directory, "_merged", "vemq")
            );

            // Step 4: all 450 vectors round-trip through the merged reader, in concatenated segment order.
            final float[][] allVectors = new float[totalVectors][];
            for (int s = 0; s < numSegments; s++) {
                System.arraycopy(segmentVectors[s], 0, allVectors, s * vectorsPerSegment, vectorsPerSegment);
            }
            verifyVectorsReadable(format, directory, mergedSegInfo, fi, allVectors, totalVectors);
        }
    }

    /**
     * Index the <b>same</b> vectors into the reordered format and the baseline
     * {@link Faiss1040ScalarQuantizedKnnVectorsFormat}, then compare results rank-by-rank.
     *
     * <p>Searches are <b>exhaustive</b> ({@code k == numVectors}): at {@code k >= maxOrd} the
     * memory-optimized searcher scores every vector and skips graph traversal entirely, so the two
     * formats' different graphs (baseline multi-layer HNSW vs our single-layer CAGRA) and different
     * entry points (descent vs hub seed) do not come into play. That isolates the only thing our change
     * touches — storage layout + scoring — which must be identical: the reorder is score-preserving, so
     * every doc must come back at the same rank with the same score.
     */
    @SneakyThrows
    public void testSearchParity_reorderedVsBaseline_whenExhaustive_thenScoresAndOrderMatch() {
        final int numVectors = 200;
        final int k = numVectors; // exhaustive
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);
        final FieldInfo fi = createRealFieldInfo();

        try (Directory reorderedDir = newDirectory(); Directory baselineDir = newDirectory()) {
            final SegmentWriteState reorderedWs = indexIntoDirectory(
                reorderedDir,
                new Faiss1040SQHNSWReorderedKnnVectorsFormat(),
                fi,
                vectors
            );
            final SegmentWriteState baselineWs = indexIntoDirectory(
                baselineDir,
                new Faiss1040ScalarQuantizedKnnVectorsFormat(),
                fi,
                vectors
            );

            try (
                KnnVectorsReader reorderedReader = new Faiss1040SQHNSWReorderedKnnVectorsFormat().fieldsReader(
                    readStateFor(reorderedDir, reorderedWs, fi)
                );
                KnnVectorsReader baselineReader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(
                    readStateFor(baselineDir, baselineWs, fi)
                )
            ) {
                // Use a spread of the indexed vectors as queries; each query is identical across formats.
                for (int qi = 0; qi < numVectors; qi += 20) {
                    final float[] query = vectors[qi];
                    final TopDocs baseline = searchAll(baselineReader, query, k, numVectors);
                    final TopDocs reordered = searchAll(reorderedReader, query, k, numVectors);

                    assertEquals("result count differs for query " + qi, baseline.scoreDocs.length, reordered.scoreDocs.length);
                    for (int rank = 0; rank < baseline.scoreDocs.length; rank++) {
                        assertEquals(
                            "doc at rank " + rank + " differs for query " + qi,
                            baseline.scoreDocs[rank].doc,
                            reordered.scoreDocs[rank].doc
                        );
                        assertEquals(
                            "score at rank " + rank + " (doc " + baseline.scoreDocs[rank].doc + ") differs for query " + qi,
                            baseline.scoreDocs[rank].score,
                            reordered.scoreDocs[rank].score,
                            0.0f
                        );
                    }
                }
            }
        }
    }

    /**
     * Recall@10 after a merge, for both formats. Indexes the same vectors as {@code numSegments}
     * separate segments, merges them into one, then measures how well the k=10 ANN (graph) search
     * recovers the exhaustive (score-exact) top-10. Ground truth is the format's own exhaustive search
     * ({@code k == numVectors}, which scores every vector), so this isolates <b>graph-search recall</b>
     * from quantization error. Logs the recall for the reordered (hub-seeded CAGRA) and baseline
     * (multi-layer HNSW) formats.
     */
    @SneakyThrows
    public void testRecallAtK10_afterMerge_reorderedVsBaseline() {
        final int numVectors = 300;
        final int numSegments = 3;
        final int k = 10;
        final int numQueries = 50;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);
        final float[][] queries = generateQueries(numQueries, DIMENSION, 7L);
        final FieldInfo fi = createRealFieldInfo();

        try (Directory reorderedDir = newDirectory(); Directory baselineDir = newDirectory()) {
            final SegmentWriteState reorderedWs = mergeIntoOneSegment(
                reorderedDir,
                new Faiss1040SQHNSWReorderedKnnVectorsFormat(),
                fi,
                vectors,
                numSegments
            );
            final SegmentWriteState baselineWs = mergeIntoOneSegment(
                baselineDir,
                new Faiss1040ScalarQuantizedKnnVectorsFormat(),
                fi,
                vectors,
                numSegments
            );

            try (
                KnnVectorsReader reorderedReader = new Faiss1040SQHNSWReorderedKnnVectorsFormat().fieldsReader(
                    readStateFor(reorderedDir, reorderedWs, fi)
                );
                KnnVectorsReader baselineReader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(
                    readStateFor(baselineDir, baselineWs, fi)
                )
            ) {
                final double reorderedRecall = recallAtK(reorderedReader, queries, k, numVectors);
                final double baselineRecall = recallAtK(baselineReader, queries, k, numVectors);

                log.info(
                    "RECALL@{} after merge [{} docs, {} segments merged, {} queries]: "
                        + "reordered(hub-seeded CAGRA)={}, baseline(multi-layer HNSW)={}",
                    k,
                    numVectors,
                    numSegments,
                    numQueries,
                    String.format(java.util.Locale.ROOT, "%.4f", reorderedRecall),
                    String.format(java.util.Locale.ROOT, "%.4f", baselineRecall)
                );

                assertTrue("reordered recall@" + k + " too low: " + reorderedRecall, reorderedRecall >= 0.3);
                assertTrue("baseline recall@" + k + " too low: " + baselineRecall, baselineRecall >= 0.3);
            }
        }
    }

    /**
     * Validates <b>scores</b> after a merge (not just doc overlap). For each query, over the merged
     * segment:
     * <ol>
     *   <li><b>Cross-format:</b> every doc gets the identical score in the reordered and baseline
     *       formats (exhaustive scan) — scoring is doc-intrinsic, so the reorder/merge must not change
     *       any score.</li>
     *   <li><b>Within-format:</b> the score the k=10 ANN (graph) search returns for a doc equals that
     *       doc's true exhaustive score — the graph path reports correct scores, it just may visit a
     *       different subset of docs.</li>
     * </ol>
     */
    @SneakyThrows
    public void testScores_afterMerge_matchExhaustiveWithinAndAcrossFormats() {
        final int numVectors = 300;
        final int numSegments = 3;
        final int k = 10;
        final int numQueries = 20;
        final float[][] vectors = generateRandomVectors(numVectors, DIMENSION);
        final float[][] queries = generateQueries(numQueries, DIMENSION, 11L);
        final FieldInfo fi = createRealFieldInfo();

        // Guard against a degenerate test: every ingested vector must be distinct, otherwise the
        // score-equality checks below would pass trivially on duplicate data.
        assertAllVectorsDistinct(vectors);

        try (Directory reorderedDir = newDirectory(); Directory baselineDir = newDirectory()) {
            final SegmentWriteState reorderedWs = mergeIntoOneSegment(
                reorderedDir,
                new Faiss1040SQHNSWReorderedKnnVectorsFormat(),
                fi,
                vectors,
                numSegments
            );
            final SegmentWriteState baselineWs = mergeIntoOneSegment(
                baselineDir,
                new Faiss1040ScalarQuantizedKnnVectorsFormat(),
                fi,
                vectors,
                numSegments
            );

            try (
                KnnVectorsReader reorderedReader = new Faiss1040SQHNSWReorderedKnnVectorsFormat().fieldsReader(
                    readStateFor(reorderedDir, reorderedWs, fi)
                );
                KnnVectorsReader baselineReader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(
                    readStateFor(baselineDir, baselineWs, fi)
                )
            ) {
                for (final float[] query : queries) {
                    final Map<Integer, Float> reorderedExhaustive = scoreByDoc(searchAll(reorderedReader, query, numVectors, numVectors));
                    final Map<Integer, Float> baselineExhaustive = scoreByDoc(searchAll(baselineReader, query, numVectors, numVectors));

                    // Guard: the query must actually differentiate docs — most docs should get distinct
                    // scores. If vectors were duplicated (or scoring were degenerate) this would collapse.
                    final long distinctScores = reorderedExhaustive.values().stream().distinct().count();
                    assertTrue(
                        "scores are not differentiated across docs (distinct=" + distinctScores + " of " + numVectors + ")",
                        distinctScores >= numVectors * 0.9
                    );

                    // (1) cross-format: identical score for every doc.
                    assertEquals("scored doc count differs", baselineExhaustive.size(), reorderedExhaustive.size());
                    for (final Map.Entry<Integer, Float> entry : baselineExhaustive.entrySet()) {
                        final float baselineScore = entry.getValue();
                        final Float reorderedScore = reorderedExhaustive.get(entry.getKey());
                        assertNotNull("doc " + entry.getKey() + " missing from reordered scores", reorderedScore);
                        assertEquals("cross-format score differs for doc " + entry.getKey(), baselineScore, reorderedScore, 0.0f);
                    }

                    // (2) within-format: the ANN-returned score is the doc's true (exhaustive) score.
                    for (final ScoreDoc sd : searchAll(reorderedReader, query, k, numVectors).scoreDocs) {
                        assertEquals("reordered ANN score wrong for doc " + sd.doc, reorderedExhaustive.get(sd.doc), sd.score, 0.0f);
                    }
                    for (final ScoreDoc sd : searchAll(baselineReader, query, k, numVectors).scoreDocs) {
                        assertEquals("baseline ANN score wrong for doc " + sd.doc, baselineExhaustive.get(sd.doc), sd.score, 0.0f);
                    }
                }
            }
        }
    }

    // ===================== helpers =====================

    /** Fails if any two vectors are identical — ensures score-parity tests aren't run on duplicate data. */
    private void assertAllVectorsDistinct(final float[][] vectors) {
        final Set<String> seen = new java.util.HashSet<>();
        for (int i = 0; i < vectors.length; i++) {
            assertTrue("duplicate vector at index " + i, seen.add(Arrays.toString(vectors[i])));
        }
    }

    private Map<Integer, Float> scoreByDoc(final TopDocs topDocs) {
        final Map<Integer, Float> byDoc = new HashMap<>();
        for (final ScoreDoc sd : topDocs.scoreDocs) {
            byDoc.put(sd.doc, sd.score);
        }
        return byDoc;
    }

    /** recall@k = mean over queries of |annTopK ∩ exhaustiveTopK| / k, using the format's own exhaustive search as truth. */
    private double recallAtK(final KnnVectorsReader reader, final float[][] queries, final int k, final int numVectors) throws IOException {
        int found = 0;
        for (final float[] query : queries) {
            final Set<Integer> truth = topDocIds(searchAll(reader, query, numVectors, numVectors), k);
            final TopDocs ann = searchAll(reader, query, k, numVectors);
            for (final ScoreDoc sd : ann.scoreDocs) {
                if (truth.contains(sd.doc)) {
                    found++;
                }
            }
        }
        return (double) found / ((long) queries.length * k);
    }

    private Set<Integer> topDocIds(final TopDocs topDocs, final int limit) {
        final Set<Integer> ids = new java.util.HashSet<>();
        for (int i = 0; i < Math.min(limit, topDocs.scoreDocs.length); i++) {
            ids.add(topDocs.scoreDocs[i].doc);
        }
        return ids;
    }

    private float[][] generateQueries(final int numQueries, final int dimension, final long seed) {
        final Random rng = new Random(seed);
        final float[][] queries = new float[numQueries][dimension];
        for (int i = 0; i < numQueries; i++) {
            for (int j = 0; j < dimension; j++) {
                queries[i][j] = rng.nextFloat() * 2 - 1;
            }
        }
        return queries;
    }

    /**
     * Indexes {@code allVectors} as {@code numSegments} equal segments, merges them into one segment via
     * {@code mergeOneField}, and returns the merged {@link SegmentWriteState} with its file set populated.
     */
    private SegmentWriteState mergeIntoOneSegment(
        final Directory dir,
        final org.apache.lucene.codecs.KnnVectorsFormat format,
        final FieldInfo fi,
        final float[][] allVectors,
        final int numSegments
    ) throws IOException {
        final int total = allVectors.length;
        final int perSegment = total / numSegments;
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fi });

        final SegmentInfo[] segInfos = new SegmentInfo[numSegments];
        final KnnVectorsReader[] readers = new KnnVectorsReader[numSegments];
        for (int s = 0; s < numSegments; s++) {
            segInfos[s] = createSegmentInfo(dir, "_" + s, perSegment);
            final SegmentWriteState ws = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                dir,
                segInfos[s],
                fieldInfos,
                null,
                IOContext.DEFAULT,
                FIELD_NAME
            );
            try (KnnVectorsWriter knnWriter = format.fieldsWriter(ws)) {
                @SuppressWarnings("unchecked")
                final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
                for (int i = 0; i < perSegment; i++) {
                    fw.addValue(i, allVectors[s * perSegment + i]);
                }
                knnWriter.flush(perSegment, null);
                knnWriter.finish();
            }
            readers[s] = format.fieldsReader(new SegmentReadState(dir, segInfos[s], fieldInfos, IOContext.DEFAULT, FIELD_NAME));
        }

        final MergeState.DocMap[] docMaps = new MergeState.DocMap[numSegments];
        final int[] maxDocs = new int[numSegments];
        final FieldInfos[] perSegFieldInfos = new FieldInfos[numSegments];
        int docBase = 0;
        for (int s = 0; s < numSegments; s++) {
            final int base = docBase;
            docMaps[s] = docID -> base + docID;
            maxDocs[s] = perSegment;
            perSegFieldInfos[s] = fieldInfos;
            docBase += perSegment;
        }

        final SegmentInfo mergedSegInfo = createSegmentInfo(dir, "_merged", total);
        final MergeState mergeState = new MergeState(
            docMaps,
            mergedSegInfo,
            fieldInfos,
            null,
            null,
            null,
            null,
            perSegFieldInfos,
            new org.apache.lucene.util.Bits[numSegments],
            null,
            null,
            readers,
            maxDocs,
            InfoStream.NO_OUTPUT,
            Runnable::run,
            false,
            null
        );
        final SegmentWriteState mergedWs = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            dir,
            mergedSegInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT,
            FIELD_NAME
        );
        try (KnnVectorsWriter mergedWriter = format.fieldsWriter(mergedWs)) {
            mergedWriter.mergeOneField(fi, mergeState);
            mergedWriter.finish();
        }
        for (final KnnVectorsReader reader : readers) {
            reader.close();
        }

        mergedWs.segmentInfo.setFiles(
            Arrays.stream(dir.listAll()).filter(f -> f.startsWith(mergedSegInfo.name)).collect(Collectors.toSet())
        );
        return mergedWs;
    }

    /** Writes {@code vectors} into {@code dir} via {@code format} and populates the segment's file set. */
    private SegmentWriteState indexIntoDirectory(
        final Directory dir,
        final org.apache.lucene.codecs.KnnVectorsFormat format,
        final FieldInfo fi,
        final float[][] vectors
    ) throws IOException {
        final SegmentWriteState ws = createWriteState(dir, "_0", vectors.length);
        try (KnnVectorsWriter knnWriter = format.fieldsWriter(ws)) {
            @SuppressWarnings("unchecked")
            final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) knnWriter.addField(fi);
            for (int i = 0; i < vectors.length; i++) {
                fw.addValue(i, vectors[i]);
            }
            knnWriter.flush(vectors.length, null);
            knnWriter.finish();
        }
        ws.segmentInfo.setFiles(Arrays.stream(dir.listAll()).filter(f -> f.startsWith(ws.segmentInfo.name)).collect(Collectors.toSet()));
        return ws;
    }

    private SegmentReadState readStateFor(final Directory dir, final SegmentWriteState ws, final FieldInfo fi) {
        return new SegmentReadState(dir, ws.segmentInfo, new FieldInfos(new FieldInfo[] { fi }), IOContext.DEFAULT, FIELD_NAME);
    }

    private TopDocs searchAll(final KnnVectorsReader reader, final float[] query, final int k, final int maxDoc) throws IOException {
        final TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
        reader.search(FIELD_NAME, query, collector, AcceptDocs.fromLiveDocs(null, maxDoc));
        return collector.topDocs();
    }

    /** Opens a reader on the segment and asserts every vector matches, in iteration order. */
    @SneakyThrows
    private void verifyVectorsReadable(
        final Faiss1040SQHNSWReorderedKnnVectorsFormat format,
        final Directory directory,
        final SegmentInfo segmentInfo,
        final FieldInfo fi,
        final float[][] expectedVectors,
        final int expectedCount
    ) {
        final SegmentReadState readState = new SegmentReadState(
            directory,
            segmentInfo,
            new FieldInfos(new FieldInfo[] { fi }),
            IOContext.DEFAULT,
            FIELD_NAME
        );
        try (KnnVectorsReader knnReader = format.fieldsReader(readState)) {
            final FloatVectorValues fvv = knnReader.getFloatVectorValues(FIELD_NAME);
            assertNotNull(fvv);
            assertEquals(expectedCount, fvv.size());
            assertEquals(DIMENSION, fvv.dimension());
            final KnnVectorValues.DocIndexIterator it = fvv.iterator();
            int count = 0;
            while (it.nextDoc() != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                assertArrayEquals(expectedVectors[count], fvv.vectorValue(it.index()), 0.0f);
                count++;
            }
            assertEquals(expectedCount, count);
        }
    }

    private List<String> filesWithSegmentAndExtension(final Directory directory, final String segmentName, final String extension)
        throws IOException {
        return Arrays.stream(directory.listAll())
            .filter(f -> f.startsWith(segmentName) && f.endsWith("." + extension))
            .collect(Collectors.toList());
    }

    private List<String> filesWithExtension(final Directory directory, final String extension) throws IOException {
        return Arrays.stream(directory.listAll()).filter(f -> f.endsWith("." + extension)).collect(Collectors.toList());
    }

    private FieldInfo createRealFieldInfo() {
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
            Map.of(
                KNNVectorFieldMapper.KNN_FIELD,
                "true",
                KNNConstants.PARAMETERS,
                PARAMETERS_JSON,
                KNNConstants.VECTOR_DATA_TYPE_FIELD,
                "float",
                KNNConstants.KNN_ENGINE,
                "faiss",
                KNNConstants.SQ_CONFIG,
                "bits=1"
            ),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            false,
            false
        );
    }

    private SegmentInfo createSegmentInfo(final Directory directory, final String segName, final int maxDoc) {
        return new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            segName,
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
    }

    private SegmentWriteState createWriteState(final Directory directory, final String segName, final int maxDoc) {
        final FieldInfo fi = createRealFieldInfo();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fi });
        final SegmentInfo segInfo = createSegmentInfo(directory, segName, maxDoc);
        return new SegmentWriteState(InfoStream.NO_OUTPUT, directory, segInfo, fieldInfos, null, IOContext.DEFAULT, FIELD_NAME);
    }

    private float[][] generateRandomVectors(final int numVectors, final int dimension) {
        final Random rng = new Random(42);
        final float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = rng.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }
}
