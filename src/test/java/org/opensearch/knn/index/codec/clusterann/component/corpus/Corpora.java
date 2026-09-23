/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import org.opensearch.knn.index.codec.clusterann.component.suite.Scenario;

import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Builds the {@link Corpus} a {@link Scenario} asks for, handing the same one to every scenario with the same
 * {@link Scenario.CorpusSpec}, since building it is the expensive part of a suite.
 */
public final class Corpora {
    private static final Map<Scenario.CorpusSpec, Corpus> BUILT = new ConcurrentHashMap<>();

    private static final String SYNTHETIC = "synthetic";
    private static final String HDF5 = "hdf5";

    private Corpora() {}

    public static Corpus of(final Scenario.CorpusSpec spec) {
        return BUILT.computeIfAbsent(spec, Corpora::build);
    }

    private static Corpus build(final Scenario.CorpusSpec spec) {
        final String type = spec.type().toLowerCase(Locale.ROOT);
        return switch (type) {
            case SYNTHETIC -> synthetic(spec);
            case HDF5 -> hdf5(spec);
            default -> throw new IllegalArgumentException("Unknown corpus type \"" + type + "\". Supported: " + SYNTHETIC + ", " + HDF5);
        };
    }

    private static Corpus synthetic(final Scenario.CorpusSpec spec) {
        final String name = String.format(
            Locale.ROOT,
            "synthetic-%s-%dx%d-c%d-sd%s-q%d-s%d",
            spec.similarity().name().toLowerCase(Locale.ROOT),
            spec.size(),
            spec.dimension(),
            spec.clusters(),
            Float.toString((float) spec.spread()),
            spec.queries(),
            spec.seed()
        );
        return SyntheticCorpus.clustered(
            name,
            spec.size(),
            spec.queries(),
            spec.dimension(),
            spec.clusters(),
            spec.spread(),
            spec.depth(),
            spec.similarity(),
            spec.seed()
        );
    }

    private static Corpus hdf5(final Scenario.CorpusSpec spec) {
        if (spec.file() == null) {
            throw new IllegalArgumentException(
                "corpus.type is hdf5 but the scenario names no corpus.file. Point it at a downloaded dataset in the"
                    + " ann-benchmarks layout: a train, test and neighbors dataset."
            );
        }

        return Hdf5Corpus.open(spec.file(), spec.queries(), spec.size(), spec.similarity());
    }
}
