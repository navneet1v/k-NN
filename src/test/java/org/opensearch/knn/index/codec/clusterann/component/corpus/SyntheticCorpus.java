/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.Arrays;
import java.util.Random;

/**
 * Generates gaussian blobs and brute-forces their ground truth. Crisply separated and isotropic, so it exercises the
 * harness and the write path but says nothing about recall on real embeddings.
 */
public final class SyntheticCorpus implements Corpus {
    private final float[][] train;
    private final float[][] test;
    private final int[][] neighbours;
    private final VectorSimilarityFunction similarity;
    private final String name;

    private SyntheticCorpus(String name, float[][] train, float[][] test, int[][] neighbours, VectorSimilarityFunction similarity) {
        this.name = name;
        this.train = train;
        this.test = test;
        this.neighbours = neighbours;
        this.similarity = similarity;
    }

    public static SyntheticCorpus clustered(
        String name,
        int size,
        int queries,
        int dimension,
        int clusters,
        double spread,
        int depth,
        VectorSimilarityFunction similarity,
        long seed
    ) {
        final Random rng = new Random(seed);
        final float[][] centres = new float[clusters][];
        for (int cluster = 0; cluster < clusters; cluster++) {
            centres[cluster] = uniform(dimension, rng);
        }

        final float[][] train = new float[size][];
        for (int row = 0; row < size; row++) {
            train[row] = around(centres[rng.nextInt(clusters)], spread, rng);
        }
        final float[][] test = new float[queries][];
        for (int row = 0; row < queries; row++) {
            test[row] = around(centres[rng.nextInt(clusters)], spread, rng);
        }

        return new SyntheticCorpus(name, train, test, groundTruth(train, test, depth, similarity), similarity);
    }

    @Override
    public int size() {
        return train.length;
    }

    @Override
    public float[] vector(final int row) {
        return train[row];
    }

    @Override
    public int queries() {
        return test.length;
    }

    @Override
    public float[] query(final int index) {
        return test[index];
    }

    @Override
    public int[] neighbours(int index, int count) {
        if (count > neighbours[index].length) {
            throw new IllegalArgumentException(
                "This corpus holds " + neighbours[index].length + " neighbours per query, asked for " + count
            );
        }
        final int[] wanted = new int[count];
        System.arraycopy(neighbours[index], 0, wanted, 0, count);
        return wanted;
    }

    @Override
    public VectorSimilarityFunction similarity() {
        return similarity;
    }

    @Override
    public int dimension() {
        return train[0].length;
    }

    @Override
    public String name() {
        return name;
    }

    private static int[][] groundTruth(float[][] train, float[][] test, int depth, VectorSimilarityFunction similarity) {
        final int[][] truth = new int[test.length][];
        final int wanted = Math.min(depth, train.length);
        for (int query = 0; query < test.length; query++) {
            final int[] best = new int[wanted];
            final float[] bestScore = new float[wanted];
            int size = 0;
            for (int row = 0; row < train.length; row++) {
                final float score = similarity.compare(test[query], train[row]);
                if (size == wanted && score <= bestScore[wanted - 1]) {
                    continue;
                }
                int at = size == wanted ? wanted - 1 : size++;
                while (at > 0 && bestScore[at - 1] < score) {
                    bestScore[at] = bestScore[at - 1];
                    best[at] = best[at - 1];
                    at--;
                }
                bestScore[at] = score;
                best[at] = row;
            }
            truth[query] = Arrays.copyOf(best, size);
        }
        return truth;
    }

    private static float[] uniform(int dimension, Random rng) {
        final float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = rng.nextFloat() * 2f - 1f;
        }
        return vector;
    }

    private static float[] around(float[] centre, double spread, Random rng) {
        final float[] vector = new float[centre.length];
        for (int i = 0; i < centre.length; i++) {
            vector[i] = centre[i] + (float) (rng.nextGaussian() * spread);
        }
        return vector;
    }
}
