/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.benchmark;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.clusterann.codec.ClusterANNFieldState;
import org.opensearch.knn.index.clusterann.codec.QuantizedVectorReader;
import org.opensearch.knn.index.clusterann.codec.QuantizedVectorWriter;

import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.BLOCK_SIZE;

/**
 * JMH benchmark comparing two block-scoring strategies for ClusterANN's ADC scoring.
 *
 * <p>Writes {@code numBlocks} (default 50) consecutive blocks into a single file,
 * then scores ALL blocks sequentially per invocation — simulating one centroid's
 * posting list scan during IVF search.
 *
 * <p><b>Usage:</b>
 * <pre>
 * ./gradlew :jmh-benchmarks:jmh -Pjmh.args="ScoreBlock -f 3 -p dimension=768 -p docBits=1 -p metric=MAXIMUM_INNER_PRODUCT"
 * </pre>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 4, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(value = 3, jvmArgsAppend = {"-Xmx2g", "-Xms2g", "-XX:+AlwaysPreTouch", "--add-modules=jdk.incubator.vector", "--enable-preview"})
public class ScoreBlockBenchmark {

    @Param({"768", "1024"})
    int dimension;

    @Param({"1"})
    byte docBits;

    @Param({"EUCLIDEAN", "MAXIMUM_INNER_PRODUCT", "DOT_PRODUCT"})
    String metric;

    @Param({"20"})
    int numBlocks;

    private Directory directory;
    private IndexInput indexInput;
    private Path path;
    private String fileName;
    private float[] queryVector;
    private float[] centroid;
    private float centroidDp;
    private int[][] docIdBufs;
    private int[][] ordBufs;
    private boolean[][] validBufs;
    private VectorSimilarityFunction simFunc;
    private ClusterANNFieldState fieldState;
    private FakeScorer scorer;
    private QuantizedVectorReader reader;
    private int totalVectors;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        Random rng = new Random(42L);
        simFunc = VectorSimilarityFunction.valueOf(metric);
        totalVectors = numBlocks * BLOCK_SIZE;

        queryVector = randomVector(rng, dimension, simFunc);

        centroid = randomVector(rng, dimension, simFunc);
        centroidDp = dotProduct(queryVector, centroid);

        docIdBufs = new int[numBlocks][BLOCK_SIZE];
        ordBufs = new int[numBlocks][BLOCK_SIZE];
        validBufs = new boolean[numBlocks][BLOCK_SIZE];

        float[][] allVectors = new float[totalVectors][];
        int[] allDocIds = new int[totalVectors];

        for (int b = 0; b < numBlocks; b++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                int globalIdx = b * BLOCK_SIZE + j;
                allVectors[globalIdx] = randomVector(rng, dimension, simFunc);
                allDocIds[globalIdx] = globalIdx;
                docIdBufs[b][j] = globalIdx;
                ordBufs[b][j] = globalIdx;
                validBufs[b][j] = rng.nextFloat() < 0.8f;
            }
        }

        fieldState = new ClusterANNFieldState(dimension, docBits);
        scorer = new FakeScorer(allVectors, allDocIds, simFunc, queryVector);

        // Write all blocks sequentially into one file
        path = Files.createTempDirectory("ScoreBlockBenchmark");
        directory = new MMapDirectory(path);
        fileName = "bench_posting.clap";
        try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
            QuantizedVectorWriter writer = new QuantizedVectorWriter(simFunc, dimension, docBits);
            for (int b = 0; b < numBlocks; b++) {
                writer.writeBlocked(ordBufs[b], BLOCK_SIZE, ord -> allVectors[ord], centroid, output);
            }
            writer.close();
        }
        indexInput = directory.openInput(fileName, IOContext.DEFAULT);

        reader = new QuantizedVectorReader(scorer, indexInput, fieldState, simFunc, queryVector, 100);
        reader.ensureQueryQuantized(centroid);
    }

    @TearDown(Level.Trial)
    public void teardown() throws IOException {
        IOUtils.close(indexInput, directory);
        Files.deleteIfExists(path.resolve(fileName));
        Files.deleteIfExists(path);
    }

    @Benchmark
    public float scoreBlock() throws IOException {
        IndexInput input = indexInput.clone();
        input.seek(0);
        KnnCollector collector = new TopKnnCollector(100, Integer.MAX_VALUE);
        reader.setCollector(collector);
        for (int b = 0; b < numBlocks; b++) {
            reader.scoreBlock(input, 0, BLOCK_SIZE, docIdBufs[b], ordBufs[b], validBufs[b], centroidDp);
        }
        return collector.minCompetitiveSimilarity();
    }

    @Benchmark
    public float scoreBlock_bulkSIMD() throws IOException {
        IndexInput input = indexInput.clone();
        input.seek(0);
        KnnCollector collector = new TopKnnCollector(100, Integer.MAX_VALUE);
        reader.setCollector(collector);
        for (int b = 0; b < numBlocks; b++) {
            reader.scoreBlock(input, 0, BLOCK_SIZE, docIdBufs[b], ordBufs[b], validBufs[b], centroidDp, true);
        }
        return collector.minCompetitiveSimilarity();
    }

    // ========== Helpers ==========

    private static float[] randomVector(Random rng, int dim, VectorSimilarityFunction simFunc) {
        float[] vec = new float[dim];
        for (int d = 0; d < dim; d++) {
            vec[d] = rng.nextFloat() * 2 - 1;
        }
        if (simFunc == VectorSimilarityFunction.COSINE || simFunc == VectorSimilarityFunction.DOT_PRODUCT
            || simFunc == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            float norm = 0f;
            for (float v : vec) norm += v * v;
            norm = (float) Math.sqrt(norm);
            if (norm > 0f) {
                for (int d = 0; d < dim; d++) vec[d] /= norm;
            }
        }
        return vec;
    }

    private static float dotProduct(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    static final class FakeScorer implements RandomVectorScorer {
        private final float[][] vectors;
        private final int[] docIds;
        private final VectorSimilarityFunction simFunc;
        private final float[] query;

        FakeScorer(float[][] vectors, int[] docIds, VectorSimilarityFunction simFunc, float[] query) {
            this.vectors = vectors;
            this.docIds = docIds;
            this.simFunc = simFunc;
            this.query = query;
        }

        @Override
        public float score(int ord) throws IOException {
            return simFunc.compare(query, vectors[ord]);
        }

        @Override
        public int maxOrd() {
            return vectors.length;
        }

        @Override
        public int ordToDoc(int ord) {
            return docIds[ord];
        }
    }
}
