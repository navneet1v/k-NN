/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

/**
 * End-to-end (codec-level) roundtrip for the ClusterANN read path: build a real {@code .clam/.clap/.clac}
 * index with {@link ClusterANN1040KnnVectorsWriter}, then search it with {@link ClusterANN1040KnnVectorsReader}
 * and measure recall against exact brute force — for both L2 (EUCLIDEAN) and inner product (MIP).
 *
 * <p>Drives the writer/reader directly (no cluster/codec-SPI needed): search() is uniform-ADC, so the
 * flat vectors reader is never touched during search and can be mocked. This exercises the full scan
 * stack — {@code OptimizedProbeScheduler → CentroidProbePlanner → ClusterANNCentroidScanner →
 * ScalarQuantizedPosting → PostingPruner chain → ADCBlockScorer}.
 */
// The IVF build uses parallel streams (ForkJoinPool.commonPool); those shared JVM workers linger and
// trip the suite-scope thread-leak check, so disable it for this codec-level roundtrip.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class ClusterANN1040SearchRoundTripTests extends KNNTestCase {

    private static final int DIM = 128;
    private static final int NUM_VECTORS = 100_000; // ~195 ADC centroids (estimateCentroids)
    private static final int NUM_CLUSTERS = 200;     // ≈ one real cluster per ADC centroid
    private static final int DOC_BITS = 4;
    private static final int K = 10;
    private static final int NUM_QUERIES = 25;
    private static final int WARMUP_QUERIES = 25; // untimed, to warm JIT + page cache before timing
    private static final long SEED = 42L;
    private static final byte[] SEGMENT_ID = new byte[16]; // shared between write + read headers
    private static final String FIELD = "test_field";
    // Cluster spread: tight clusters are trivially separable (easy); wide clusters overlap so the true
    // top-10 straddles several clusters — a realistic stress on nprobe + CLIP + quantization.
    private static final float TIGHT_STDDEV = 0.15f;
    private static final float WIDE_STDDEV = 2.0f;

    private float[][] centers;  // cluster centers, captured by clusteredVectors() for independent queries
    private float clusterStddev; // set per run from the query mode

    /** How query vectors are generated for a recall run. */
    private enum QueryMode {
        /** Query is a near-duplicate of an indexed vector (easy — a trivial distance-~0 top-1). */
        PERTURBED,
        /** Query is a fresh sample from a cluster's distribution — no near-duplicate; all NN are genuine. */
        INDEPENDENT
    }

    // --- Easy case: perturbed near-duplicate queries. High recall expected. ---

    public void testRecall_L2_perturbed() throws Exception {
        assertRecallAtLeast(VectorSimilarityFunction.EUCLIDEAN, QueryMode.PERTURBED, 0.80);
    }

    public void testRecall_InnerProduct_perturbed() throws Exception {
        assertRecallAtLeast(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, QueryMode.PERTURBED, 0.80);
    }

    // --- Realistic case: independent queries (no near-duplicate). Lower, honest recall. ---

    public void testRecall_L2_independent() throws Exception {
        assertRecallAtLeast(VectorSimilarityFunction.EUCLIDEAN, QueryMode.INDEPENDENT, 0.60);
    }

    public void testRecall_InnerProduct_independent() throws Exception {
        assertRecallAtLeast(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, QueryMode.INDEPENDENT, 0.60);
    }

    private void assertRecallAtLeast(VectorSimilarityFunction simFunc, QueryMode mode, double minRecall) throws Exception {
        // Perturbed = easy tight clusters; independent = realistic overlapping clusters.
        clusterStddev = mode == QueryMode.PERTURBED ? TIGHT_STDDEV : WIDE_STDDEV;
        List<float[]> vectors = clusteredVectors();
        Random rng = new Random(SEED + simFunc.ordinal());

        Path tmp = createTempDir("clusterann-roundtrip-" + simFunc.name());
        try (Directory dir = new MMapDirectory(tmp)) {
            writeIndex(dir, vectors, simFunc);

            FlatVectorsReader flatReader = mock(FlatVectorsReader.class); // unused by search()
            SegmentReadState readState = createReadState(dir, simFunc);
            try (ClusterANN1040KnnVectorsReader reader = new ClusterANN1040KnnVectorsReader(flatReader, readState)) {

                // Warm up JIT + page cache so the timed pass is representative (not measuring cold start).
                for (int w = 0; w < WARMUP_QUERIES; w++) {
                    search(reader, mode == QueryMode.PERTURBED
                        ? perturb(vectors.get(rng.nextInt(NUM_VECTORS)), rng, 0.05f)
                        : independentQuery(rng));
                }

                double recallSum = 0;
                double relaxedSum = 0; // fraction of returned docs within the TRUE top-(5*K)
                long visitedSum = 0;   // total vectors ADC-scored across queries (CLIP work metric)
                long searchNanos = 0;  // wall-clock spent inside search() only (excludes brute-force ranking)
                for (int q = 0; q < NUM_QUERIES; q++) {
                    float[] query = mode == QueryMode.PERTURBED
                        ? perturb(vectors.get(rng.nextInt(NUM_VECTORS)), rng, 0.05f)
                        : independentQuery(rng);

                    // Exact ranking, computed once per query (distances precomputed, not recomputed in the sort).
                    int[] ranked = rankedBySimilarity(vectors, query, simFunc);
                    Set<Integer> expected = prefixSet(ranked, K);
                    Set<Integer> relaxed = prefixSet(ranked, 5 * K);
                    long t0 = System.nanoTime();
                    Set<Integer> actual = search(reader, query);
                    searchNanos += System.nanoTime() - t0;
                    visitedSum += lastVisited;

                    Set<Integer> hit = new HashSet<>(expected);
                    hit.retainAll(actual);
                    recallSum += (double) hit.size() / K;

                    Set<Integer> relaxedHit = new HashSet<>(relaxed);
                    relaxedHit.retainAll(actual);
                    relaxedSum += actual.isEmpty() ? 0 : (double) relaxedHit.size() / actual.size();
                }
                double recall = recallSum / NUM_QUERIES;
                double relaxedRecall = relaxedSum / NUM_QUERIES;
                double avgVisited = (double) visitedSum / NUM_QUERIES;
                double avgSearchMs = searchNanos / 1e6 / NUM_QUERIES;
                logger.info("[ClusterANN roundtrip] {} {} recall@{} = {} ; relaxed = {} ; avgScored/query = {} ({}% of {}) ; avgSearch = {} ms/query ({}-bit)",
                    simFunc, mode, K, recall, relaxedRecall, avgVisited,
                    String.format("%.1f", 100.0 * avgVisited / NUM_VECTORS), NUM_VECTORS,
                    String.format("%.3f", avgSearchMs), DOC_BITS);
                assertTrue(
                    simFunc + " recall@" + K + " = " + recall + " below floor " + minRecall,
                    recall >= minRecall
                );
            }
        }
    }

    private long lastVisited; // vectors ADC-scored during the most recent search() (work metric)

    private Set<Integer> search(ClusterANN1040KnnVectorsReader reader, float[] query) throws Exception {
        TopKnnCollector collector = new TopKnnCollector(K, Integer.MAX_VALUE);
        reader.search(FIELD, query, collector, AcceptDocs.fromLiveDocs(null, NUM_VECTORS));
        lastVisited = collector.visitedCount(); // read before topDocs() (which is destructive)
        TopDocs td = collector.topDocs();
        Set<Integer> docs = new HashSet<>();
        for (ScoreDoc sd : td.scoreDocs) {
            docs.add(sd.doc);
        }
        return docs;
    }

    // ===== data =====

    /** Tight, well-separated Gaussian clusters — clear nearest neighbors so recall is a real signal. */
    private List<float[]> clusteredVectors() {
        Random rng = new Random(SEED);
        centers = new float[NUM_CLUSTERS][DIM];
        for (int c = 0; c < NUM_CLUSTERS; c++) {
            for (int d = 0; d < DIM; d++) {
                centers[c][d] = rng.nextFloat() * 10f;
            }
        }
        List<float[]> vectors = new ArrayList<>(NUM_VECTORS);
        for (int i = 0; i < NUM_VECTORS; i++) {
            vectors.add(sampleCluster(centers[i % NUM_CLUSTERS], rng));
        }
        return vectors;
    }

    /** A fresh sample from a cluster's Gaussian distribution (same process the indexed vectors use). */
    private float[] sampleCluster(float[] center, Random rng) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = center[d] + (float) rng.nextGaussian() * clusterStddev;
        }
        return v;
    }

    /** Independent query: a fresh cluster sample — in-distribution but not an indexed vector. */
    private float[] independentQuery(Random rng) {
        return sampleCluster(centers[rng.nextInt(NUM_CLUSTERS)], rng);
    }

    private float[] perturb(float[] base, Random rng, float scale) {
        float[] v = new float[base.length];
        for (int d = 0; d < base.length; d++) {
            v[d] = base[d] + (float) rng.nextGaussian() * scale;
        }
        return v;
    }

    /** Doc ids ranked by the field's similarity (higher = closer), best-first. Distances precomputed once. */
    private int[] rankedBySimilarity(List<float[]> vectors, float[] query, VectorSimilarityFunction sim) {
        int n = vectors.size();
        float[] score = new float[n];
        for (int i = 0; i < n; i++) {
            score[i] = sim.compare(query, vectors.get(i)); // one distance computation per vector
        }
        Integer[] ids = new Integer[n];
        for (int i = 0; i < n; i++) {
            ids[i] = i;
        }
        Arrays.sort(ids, (a, b) -> Float.compare(score[b], score[a])); // cheap: reads precomputed scores
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = ids[i];
        }
        return out;
    }

    private Set<Integer> prefixSet(int[] ranked, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < k && i < ranked.length; i++) {
            set.add(ranked[i]);
        }
        return set;
    }

    // ===== write / read plumbing (direct, no codec SPI) =====

    @SuppressWarnings("unchecked")
    private void writeIndex(Directory dir, List<float[]> vectors, VectorSimilarityFunction simFunc) throws Exception {
        SegmentWriteState writeState = createWriteState(dir, simFunc);

        FlatVectorsWriter flatWriter = mock(FlatVectorsWriter.class);
        FlatFieldVectorsWriter<float[]> fieldWriter = mock(FlatFieldVectorsWriter.class);
        doReturn(fieldWriter).when(flatWriter).addField(any());
        doReturn(vectors).when(fieldWriter).getVectors();
        DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        for (int i = 0; i < vectors.size(); i++) {
            docsWithField.add(i);
        }
        doReturn(docsWithField).when(fieldWriter).getDocsWithFieldSet();

        try (ClusterANN1040KnnVectorsWriter writer = new ClusterANN1040KnnVectorsWriter(writeState, flatWriter, DOC_BITS)) {
            writer.addField(fieldInfo(simFunc));
            writer.flush(vectors.size(), null);
            writer.finish();
        }
    }

    private SegmentWriteState createWriteState(Directory dir, VectorSimilarityFunction simFunc) {
        SegmentInfo segInfo = segmentInfo(dir);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo(simFunc) });
        return new SegmentWriteState(InfoStream.NO_OUTPUT, dir, segInfo, fieldInfos, null, IOContext.DEFAULT);
    }

    private SegmentReadState createReadState(Directory dir, VectorSimilarityFunction simFunc) {
        SegmentInfo segInfo = segmentInfo(dir);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo(simFunc) });
        return new SegmentReadState(dir, segInfo, fieldInfos, IOContext.DEFAULT, "");
    }

    private SegmentInfo segmentInfo(Directory dir) {
        return new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "seg0",
            NUM_VECTORS,
            false,
            false,
            null,
            Collections.emptyMap(),
            SEGMENT_ID,
            Collections.emptyMap(),
            null
        );
    }

    private FieldInfo fieldInfo(VectorSimilarityFunction simFunc) {
        return new FieldInfo(
            FIELD,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            org.apache.lucene.index.DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0,
            0,
            0,
            DIM,
            VectorEncoding.FLOAT32,
            simFunc,
            false,
            false
        );
    }
}
