/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.NamedThreadFactory;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;
import org.opensearch.knn.index.codec.clusterann.component.baseline.Baselines;
import org.opensearch.knn.index.codec.clusterann.component.corpus.Corpora;
import org.opensearch.knn.index.codec.clusterann.component.corpus.Corpus;
import org.opensearch.knn.index.codec.clusterann.component.corpus.Hdf5Corpus;
import org.opensearch.knn.index.codec.clusterann.component.profile.FlightRecording;
import org.opensearch.knn.index.codec.clusterann.component.suite.Scenario;
import org.opensearch.knn.index.codec.clusterann.component.suite.Suite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Indexes a corpus through a {@code KnnVectorsFormat}, reopens the segment and searches it, reporting recall against
 * the corpus's own ground truth plus p50/p90 latency. Each scenario in the suite file becomes one case; the format is
 * a name resolved through SPI, so this test depends on no format's types. See README.md for how to run it.
 */

@LuceneTestCase.SuppressSysoutChecks(bugUrl = "The measurements are the point of this test")
@ThreadLeakFilters(defaultFilters = true, filters = { FlightRecording.JfrThreadFilter.class })
public class KNNVectorFormatsComponentTest extends LuceneTestCase {
    private static final Logger log = LogManager.getLogger(KNNVectorFormatsComponentTest.class);

    private static final String VECTOR_FIELD = "float_field";

    private static final String ROW_FIELD = "row";

    private static final double RECALL_TOLERANCE = 0.01;

    private static final double LATENCY_HEADROOM = 2.0;

    private final Scenario scenario;
    private final Corpus corpus;
    private final int k;

    public KNNVectorFormatsComponentTest(final Scenario scenario) {
        this.scenario = scenario;
        this.corpus = Corpora.of(scenario.corpus());
        this.k = scenario.search().k();
    }

    @ParametersFactory(argumentFormatting = "%s")
    public static Iterable<Object[]> scenarios() {
        return Suite.load().stream().map(scenario -> new Object[] { scenario }).toList();
    }

    static {
        Configurator.setLevel(KNNVectorFormatsComponentTest.class.getName(), Level.INFO);
        Configurator.setLevel(FlightRecording.class.getName(), Level.INFO);
        Configurator.setLevel(Suite.class.getName(), Level.INFO);

        Configurator.setLevel(Hdf5Corpus.class.getName(), Level.INFO);
    }

    private IndexWriterConfig config() {
        return new IndexWriterConfig().setCodec(codec())
            .setMaxBufferedDocs(scenario.index().docsPerSegment())
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE)

            .setUseCompoundFile(false);
    }

    private Codec codec() {
        final String name = scenario.codec().name();

        final Integer docBits = docBitsFrom(scenario.codec().params());
        if (docBits != null) {
            if (!ClusterANN1030TestCodec.CODEC_NAME.equals(name)) {
                throw new IllegalArgumentException(
                    "Scenario \""
                        + scenario.name()
                        + "\" sets quantization.docBits, which only "
                        + ClusterANN1030TestCodec.CODEC_NAME
                        + " can be built with, but names codec \""
                        + name
                        + "\""
                );
            }
            return new ClusterANN1030TestCodec(docBits.intValue());
        }
        try {
            return Codec.forName(name);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("No codec \"" + name + "\" is registered for SPI. Available: " + Codec.availableCodecs(), e);
        }
    }

    private static Integer docBitsFrom(final Map<String, Object> params) {
        final Object quantization = params.get("quantization");
        if (!(quantization instanceof Map<?, ?> values)) {
            return null;
        }
        final Object docBits = values.get("docBits");
        if (docBits == null) {
            return null;
        }
        if (!(docBits instanceof Number width)) {
            throw new IllegalArgumentException("quantization.docBits must be a number, found \"" + docBits + "\"");
        }
        return width.intValue();
    }

    private IndexWriterConfig mergeConfig() {
        return new IndexWriterConfig().setCodec(codec()).setUseCompoundFile(false);
    }

    private Directory directory() throws Exception {
        return new MMapDirectory(createTempDir("clusterann-corpus"));
    }

    public void testIndexAndSearch() throws Exception {
        final Integer segments = scenario.index().forceMerge();

        try (Directory dir = directory()) {
            log.info(
                "Indexing {} vectors of {} dimensions from {} under {} with {}, forceMerge={}",
                corpus.size(),
                corpus.dimension(),
                corpus.name(),
                corpus.similarity(),
                codec().getName(),
                segments
            );

            final long indexStart = System.nanoTime();
            try (FlightRecording recording = FlightRecording.start(phase("index")); IndexWriter writer = new IndexWriter(dir, config())) {
                for (int row = 0; row < corpus.size(); row++) {
                    final Document document = new Document();
                    document.add(new KnnFloatVectorField(VECTOR_FIELD, corpus.vector(row), corpus.similarity()));
                    document.add(new StoredField(ROW_FIELD, row));
                    writer.addDocument(document);
                }
                writer.commit();
            }
            log.info("Indexed in {} ms", millisSince(indexStart));

            if (segments != null) {
                final long mergeStart = System.nanoTime();
                try (
                    FlightRecording recording = FlightRecording.start(phase("merge"));
                    IndexWriter merger = new IndexWriter(dir, mergeConfig())
                ) {
                    merger.forceMerge(segments);
                    merger.commit();
                }
                log.info("Merged in {} ms", millisSince(mergeStart));
            }

            assertExpectedFilesExist(dir);

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                log.info("{} docs in {} segments", reader.numDocs(), reader.leaves().size());
                assertEquals("every vector in the corpus must have been indexed", corpus.size(), reader.maxDoc());
                if (segments != null) {
                    assertEquals("the corpus must have been merged into one segment", 1, reader.leaves().size());
                }

                final int threads = scenario.search().threads();
                final ExecutorService executor = threads > 1
                    ? Executors.newFixedThreadPool(threads, new NamedThreadFactory("knn-search"))
                    : null;
                try {
                    final Baselines.Measurement measured = measure(
                        executor == null ? new IndexSearcher(reader) : new IndexSearcher(reader, executor)
                    );
                    assertTrue("recall@" + k + " was " + measured.recall(), measured.recall() > 0.0);
                    checkAgainstBaseline(measured);
                } finally {
                    if (executor != null) {
                        executor.shutdown();
                        if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                            executor.shutdownNow();
                        }
                    }
                }
            }
        }
    }

    private void assertExpectedFilesExist(final Directory dir) throws Exception {
        final List<String> expected = scenario.index().expectFiles();
        if (expected.isEmpty()) {
            return;
        }
        final List<String> written = Arrays.asList(dir.listAll());
        for (final String extension : expected) {
            assertTrue(
                "Scenario \""
                    + scenario.name()
                    + "\" expects the codec to write ."
                    + extension
                    + " but it did not, so any"
                    + " recall measured here would be Lucene's exact-search fallback rather than this format: "
                    + written,
                written.stream().anyMatch(file -> file.endsWith("." + extension))
            );
        }
    }

    private Baselines.Measurement measure(final IndexSearcher searcher) throws Exception {
        final int queryCount = Math.min(corpus.queries(), scenario.search().queries());

        final List<Set<Integer>> expected = new ArrayList<>(queryCount);
        for (int query = 0; query < queryCount; query++) {
            final Set<Integer> neighbours = new HashSet<>();
            for (final int neighbour : corpus.neighbours(query, k)) {
                neighbours.add(neighbour);
            }
            expected.add(neighbours);
        }

        warmUp(searcher, queryCount);

        final TopDocs[] hits = new TopDocs[queryCount];
        final long[] latencies = new long[queryCount * scenario.search().repeat()];
        int measured = 0;
        final long start = System.nanoTime();
        try (FlightRecording recording = FlightRecording.start(phase("search"))) {
            for (int pass = 0; pass < scenario.search().repeat(); pass++) {
                for (int query = 0; query < queryCount; query++) {
                    final long queryStart = System.nanoTime();
                    final TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, corpus.query(query), k), k);
                    latencies[measured++] = System.nanoTime() - queryStart;
                    hits[query] = topDocs;
                }
            }
        }
        final long millis = millisSince(start);

        int found = 0;
        final ScoreError scoreError = new ScoreError();
        for (int query = 0; query < queryCount; query++) {
            assertScoresWellFormed(query, hits[query]);
            for (final ScoreDoc hit : hits[query].scoreDocs) {
                final int row = rowOf(searcher, hit.doc);
                if (expected.get(query).contains(row)) {
                    found++;
                }

                scoreError.add(query, row, hit.score, corpus.similarity().compare(corpus.query(query), corpus.vector(row)));
            }
        }
        assertScoresAreAccurate(scoreError);

        Arrays.sort(latencies);
        log.info(
            "{} queries x {} passes in {} ms: mean {} ms, p50 {} ms, p90 {} ms",
            queryCount,
            scenario.search().repeat(),
            millis,
            (double) millis / latencies.length,
            millis(percentile(latencies, 50)),
            millis(percentile(latencies, 90))
        );
        return new Baselines.Measurement(
            (double) found / ((long) queryCount * k),
            millis(percentile(latencies, 50)),
            millis(percentile(latencies, 90))
        );
    }

    private void assertScoresAreAccurate(final ScoreError error) {
        final double tolerance = scenario.search().scoreTolerance();
        final double bias = scenario.search().scoreBias();
        if (tolerance < 0 || error.count == 0) {
            return;
        }
        log.info(
            "score error over {} hits: max {} (query {} row {}: returned {}, exact {}), mean signed {}",
            error.count,
            error.maxAbsolute,
            error.worstQuery,
            error.worstRow,
            error.worstReturned,
            error.worstExact,
            error.meanSigned()
        );

        assertTrue(
            "score for query "
                + error.worstQuery
                + " row "
                + error.worstRow
                + " was "
                + error.worstReturned
                + " but the exact score of that document is "
                + error.worstExact
                + ", off by "
                + error.maxAbsolute
                + " which exceeds the scenario's scoreTolerance of "
                + tolerance,
            error.maxAbsolute <= tolerance
        );
        assertTrue(
            "scores lean "
                + (error.meanSigned() > 0 ? "high" : "low")
                + " by "
                + error.meanSigned()
                + " on average over "
                + error.count
                + " hits, beyond the scenario's scoreBias of "
                + bias
                + "; rounding alone averages to about zero, so this is a systematic error rather than noise",
            Math.abs(error.meanSigned()) <= bias
        );
    }

    private static final class ScoreError {
        private double maxAbsolute;
        private double sumSigned;
        private int count;
        private int worstQuery = -1;
        private int worstRow = -1;
        private float worstReturned;
        private float worstExact;

        private void add(final int query, final int row, final float returned, final float exact) {
            final double signed = (double) returned - exact;
            sumSigned += signed;
            count++;
            if (Math.abs(signed) >= maxAbsolute) {
                maxAbsolute = Math.abs(signed);
                worstQuery = query;
                worstRow = row;
                worstReturned = returned;
                worstExact = exact;
            }
        }

        private double meanSigned() {
            return count == 0 ? 0 : sumSigned / count;
        }
    }

    private void assertScoresWellFormed(final int query, final TopDocs topDocs) {
        float previous = Float.POSITIVE_INFINITY;
        for (int rank = 0; rank < topDocs.scoreDocs.length; rank++) {
            final ScoreDoc hit = topDocs.scoreDocs[rank];
            final String where = "query " + query + " rank " + rank + " (doc " + hit.doc + ", score " + hit.score + ")";
            assertTrue("score is not a finite number for " + where, Float.isFinite(hit.score));
            assertTrue("score is negative for " + where, hit.score >= 0f);
            assertTrue("scores are not descending: " + where + " scored above " + previous, hit.score <= previous);
            previous = hit.score;
        }
    }

    private static int rowOf(final IndexSearcher searcher, final int doc) throws Exception {
        return searcher.storedFields().document(doc).getField(ROW_FIELD).numericValue().intValue();
    }

    private void warmUp(final IndexSearcher searcher, final int queryCount) throws Exception {
        final int warmup = Math.min(scenario.search().warmup(), queryCount);
        if (warmup <= 0) {
            return;
        }
        final long start = System.nanoTime();
        for (int query = 0; query < warmup; query++) {
            searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, corpus.query(query), k), k);
        }
        log.info("Warmed up with {} queries in {} ms", warmup, millisSince(start));
    }

    private String phase(final String name) {
        final Integer segments = scenario.index().forceMerge();
        return corpus.name()
            + "-"
            + codec().getName()
            + "-"
            + scenario.name()
            + (segments == null ? "-flushed-" : "-merged" + segments + "-")
            + name;
    }

    private void checkAgainstBaseline(final Baselines.Measurement measured) throws Exception {
        final Baselines.Key key = new Baselines.Key(scenario.name(), corpus.name(), codec().getName(), k, scenario.search().queries());
        log.info("recall@{} = {}, p50 {} ms, p90 {} ms", k, measured.recall(), measured.p50Millis(), measured.p90Millis());

        final Baselines baselines = Baselines.forSuite(Suite.file());
        if (Boolean.getBoolean("knn.updateBaselines")) {
            baselines.update(key, measured);
            log.info("Recorded baseline in {}: {}", baselines.file(), measured.toCsv(key));
            return;
        }

        final Baselines.Measurement baseline = baselines.find(key).orElse(null);
        if (baseline == null) {
            log.info("No baseline for {} in {}. To record one, add this row or rerun with -Dknn.updateBaselines:", key, baselines.file());
            log.info("  {}", measured.toCsv(key));
            return;
        }

        log.info(
            "Baseline recall {} (delta {}), p50 {} ms (delta {} ms), p90 {} ms (delta {} ms)",
            baseline.recall(),
            measured.recall() - baseline.recall(),
            baseline.p50Millis(),
            measured.p50Millis() - baseline.p50Millis(),
            baseline.p90Millis(),
            measured.p90Millis() - baseline.p90Millis()
        );

        assertTrue(
            "recall@" + k + " fell from a baseline of " + baseline.recall() + " to " + measured.recall(),
            measured.recall() >= baseline.recall() - RECALL_TOLERANCE
        );
        if (Boolean.getBoolean("knn.assert.latency")) {
            assertTrue(
                "p50 rose from a baseline of " + baseline.p50Millis() + " ms to " + measured.p50Millis() + " ms",
                measured.p50Millis() <= baseline.p50Millis() * LATENCY_HEADROOM
            );
        }
    }

    private static long millisSince(final long nanos) {
        return (System.nanoTime() - nanos) / 1_000_000;
    }

    private static long percentile(final long[] sortedNanos, final int percentile) {
        final int rank = (int) Math.ceil(percentile / 100.0 * sortedNanos.length) - 1;
        return sortedNanos[Math.max(0, Math.min(sortedNanos.length - 1, rank))];
    }

    private static double millis(final long nanos) {
        return nanos / 1_000_000.0;
    }
}
