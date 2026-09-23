/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import static org.mockito.Mockito.mock;

/**
 * Drives {@link ClusterANNQuery} and {@link ClusterANNRescoreQuery} against a real multi-segment index written by the
 * ClusterANN format, so the whole path is exercised: the query layer, the codec's reader, and — for the rescore — the
 * exact pass over the stored full-precision vectors.
 *
 * <p>Doc ids are insertion ordinals here: documents are added in order and nothing merges or deletes, so a hit's doc id
 * indexes straight back into {@code vectors} and brute force needs no side table.
 */
public class ClusterANNQueryTests extends LuceneTestCase {

    private static final String FIELD = "cluster_vec";
    private static final String PARITY_FIELD = "parity";
    private static final int DIM = 16;
    private static final int NUM_DOCS = 600;
    private static final int DOCS_PER_SEGMENT = 200;

    private static final float[][] vectors = new float[NUM_DOCS][DIM];

    private Directory directory;
    private DirectoryReader reader;
    private IndexSearcher searcher;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = newDirectory();
        final Random random = new Random(42);
        final IndexWriterConfig config = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false);
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                for (int d = 0; d < DIM; d++) {
                    vectors[i][d] = random.nextFloat();
                }
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(PARITY_FIELD, i % 2 == 0 ? "even" : "odd", Field.Store.NO));
                writer.addDocument(doc);
                if (i % DOCS_PER_SEGMENT == DOCS_PER_SEGMENT - 1) {
                    writer.flush();
                }
            }
        }
        reader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(reader);
        assertEquals(NUM_DOCS / DOCS_PER_SEGMENT, reader.leaves().size());
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        super.tearDown();
    }

    /** The bare scan: k hits, in descending score order, merged across every segment. */
    public void testApproximateScanReturnsTopK() throws IOException {
        final int k = 10;
        final TopDocs hits = searcher.search(new ClusterANNQuery(FIELD, vectors[7], k, null, null, 0), k);

        assertEquals(k, hits.scoreDocs.length);
        for (int i = 1; i < hits.scoreDocs.length; i++) {
            assertTrue("hits must be in descending score order", hits.scoreDocs[i - 1].score >= hits.scoreDocs[i].score);
        }
        // The query vector is a document's own vector, so that document is its own nearest neighbour and the scan must
        // find it: this is the one hit whose rank does not depend on the quantized ordering being any good.
        assertEquals(7, hits.scoreDocs[0].doc);
    }

    /** A segment holding no accepted document contributes nothing, and hits stay inside the filter. */
    public void testFilterRestrictsHitsToMatchingDocuments() throws IOException {
        final int k = 10;
        final Query filter = new TermQuery(new Term(PARITY_FIELD, "even"));
        final TopDocs hits = searcher.search(new ClusterANNQuery(FIELD, vectors[7], k, filter, null, 0), k);

        assertEquals(k, hits.scoreDocs.length);
        for (final ScoreDoc scoreDoc : hits.scoreDocs) {
            assertEquals("filtered scan returned an odd document", 0, scoreDoc.doc % 2);
        }
    }

    /** Nothing accepted anywhere is an empty result, not a failure. */
    public void testFilterMatchingNothingYieldsNoHits() throws IOException {
        final Query filter = new TermQuery(new Term(PARITY_FIELD, "neither"));
        final TopDocs hits = searcher.search(new ClusterANNQuery(FIELD, vectors[7], 10, filter, null, 0), 10);
        assertEquals(0, hits.scoreDocs.length);
    }

    /**
     * The rescore's scores are the exact ones. The scan ranks by a quantized estimate; after rescoring, every returned
     * score must equal what the stored full-precision vector scores against the query.
     */
    public void testRescoreReturnsExactScores() throws IOException {
        final int k = 10;
        final int firstPassK = 50;
        final float[] query = randomQueryVector();

        final TopDocs hits = searcher.search(rescoreQuery(query, firstPassK, k, null), k);

        assertEquals(k, hits.scoreDocs.length);
        for (final ScoreDoc scoreDoc : hits.scoreDocs) {
            final float exact = VectorSimilarityFunction.EUCLIDEAN.compare(query, vectors[scoreDoc.doc]);
            assertEquals("doc " + scoreDoc.doc + " was not scored exactly", exact, scoreDoc.score, 1e-5f);
        }
        for (int i = 1; i < hits.scoreDocs.length; i++) {
            assertTrue("rescored hits must be in descending score order", hits.scoreDocs[i - 1].score >= hits.scoreDocs[i].score);
        }
    }

    /** Oversampling then rescoring is at least as accurate as the scan alone — which is the only reason to pay for it. */
    public void testRescoreIsAtLeastAsAccurateAsTheScan() throws IOException {
        final int k = 10;
        final int firstPassK = 100;
        final float[] query = randomQueryVector();
        final int[] trueNeighbours = bruteForceTopK(query, k);

        final TopDocs scanned = searcher.search(new ClusterANNQuery(FIELD, query, k, null, null, 0), k);
        final TopDocs rescored = searcher.search(rescoreQuery(query, firstPassK, k, null), k);

        final int scanRecall = overlap(scanned, trueNeighbours);
        final int rescoreRecall = overlap(rescored, trueNeighbours);
        assertTrue(
            "rescoring lost ground: scan found " + scanRecall + " of " + k + ", rescore found " + rescoreRecall,
            rescoreRecall >= scanRecall
        );
    }

    /** {@code firstPassK} is what feeds the rescore, so a value below k would cap the result before it is scored. */
    public void testRescoreRejectsFirstPassKBelowK() {
        final IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> rescoreQuery(vectors[0], 5, 10, null));
        assertTrue(e.getMessage(), e.getMessage().contains("must be at least k=10"));
    }

    /** No candidate survives the first pass, so there is nothing to rescore and the weight matches nothing. */
    public void testRescoreOverEmptyCandidateSetYieldsNoHits() throws IOException {
        final Query filter = new TermQuery(new Term(PARITY_FIELD, "neither"));
        final TopDocs hits = searcher.search(rescoreQuery(vectors[7], 50, 10, filter), 10);
        assertEquals(0, hits.scoreDocs.length);
    }

    private Query rescoreQuery(final float[] query, final int firstPassK, final int k, final Query filter) {
        final Query inner = new ClusterANNQuery(FIELD, query, firstPassK, filter, null, 0);
        return new ClusterANNRescoreQuery(inner, FIELD, firstPassK, k, query, null, () -> new ExactSearcher(mock(ModelDao.class)));
    }

    private static float[] randomQueryVector() {
        final Random random = new Random(7);
        final float[] query = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            query[d] = random.nextFloat();
        }
        return query;
    }

    /** The true top-k doc ids, scored against the vectors as written. */
    private static int[] bruteForceTopK(final float[] query, final int k) {
        final Integer[] docs = new Integer[NUM_DOCS];
        for (int i = 0; i < NUM_DOCS; i++) {
            docs[i] = i;
        }
        Arrays.sort(docs, Comparator.comparingDouble(doc -> -VectorSimilarityFunction.EUCLIDEAN.compare(query, vectors[doc])));
        final int[] topK = new int[k];
        for (int i = 0; i < k; i++) {
            topK[i] = docs[i];
        }
        return topK;
    }

    private static int overlap(final TopDocs hits, final int[] expected) {
        int found = 0;
        for (final ScoreDoc scoreDoc : hits.scoreDocs) {
            for (final int doc : expected) {
                if (doc == scoreDoc.doc) {
                    found++;
                    break;
                }
            }
        }
        return found;
    }
}
