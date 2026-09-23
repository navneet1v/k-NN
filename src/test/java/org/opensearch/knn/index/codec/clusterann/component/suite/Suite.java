/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.suite;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Reads the suite file named by {@code -Dknn.suite}, merging each scenario's overrides over the file's defaults and
 * rejecting unknown keys, so a typo fails rather than being carried silently.
 */
public final class Suite {
    private static final Logger log = LogManager.getLogger(Suite.class);

    private static final Path DEFAULT_FILE = Paths.get("src/test/resources/cohere-1k-suite.yml");

    private static final String DEFAULT_CODEC = "Lucene103";

    private static final Set<String> SECTIONS = Set.of("corpus", "index", "search", "codec");
    private static final Set<String> CORPUS_KEYS = Set.of(
        "type",
        "size",
        "queries",
        "dimension",
        "clusters",
        "spread",
        "depth",
        "seed",
        "similarity",
        "file"
    );
    private static final Set<String> INDEX_KEYS = Set.of("docsPerSegment", "expectFiles", "forceMerge");
    private static final Set<String> SEARCH_KEYS = Set.of("k", "queries", "repeat", "warmup", "threads", "scoreTolerance", "scoreBias");
    private static final Set<String> CODEC_KEYS = Set.of("name", "params");

    private Suite() {}

    public static Path file() {
        return Paths.get(System.getProperty("knn.suite", DEFAULT_FILE.toString()));
    }

    public static List<Scenario> load() {
        final Path file = file();
        if (Files.notExists(file)) {
            log.info("No suite file at {}, running the single default scenario", file.toAbsolutePath());
            return List.of(scenario("default", Map.of()));
        }

        final Map<String, Object> document = read(file);
        final Map<String, Object> defaults = section(document, "defaults");
        final Object scenarios = document.get("scenarios");
        if (!(scenarios instanceof List<?> rows) || rows.isEmpty()) {
            throw new IllegalArgumentException(file + " has no scenarios; remove the file to run the default scenario");
        }

        final List<Scenario> loaded = new ArrayList<>(rows.size());
        final Set<String> names = new LinkedHashSet<>();
        for (final Object row : rows) {
            final Map<String, Object> overrides = asMap(row, "a scenario");
            final Object name = overrides.get("name");
            if (name == null) {
                throw new IllegalArgumentException("Every scenario needs a name; one in " + file + " has none");
            }
            if (!names.add(name.toString())) {
                throw new IllegalArgumentException("Duplicate scenario name \"" + name + "\" in " + file);
            }
            loaded.add(scenario(name.toString(), merge(defaults, overrides, file)));
        }

        log.info("Loaded {} scenarios from {}: {}", loaded.size(), file.toAbsolutePath(), names);
        for (final Scenario candidate : loaded) {
            if (!candidate.codec().params().isEmpty()) {
                log.warn(
                    "Scenario \"{}\" sets codec params {} which nothing applies yet; the codec is still resolved by name",
                    candidate.name(),
                    candidate.codec().params()
                );
            }
        }
        return loaded;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> read(final Path file) {
        try (InputStream in = Files.newInputStream(file)) {
            final Object document = new Yaml().load(in);
            if (document == null) {
                throw new IllegalArgumentException(file + " is empty");
            }
            return (Map<String, Object>) asMap(document, file.toString());
        } catch (IOException e) {
            throw new IllegalStateException("Could not read the suite file " + file, e);
        }
    }

    private static Map<String, Object> merge(final Map<String, Object> defaults, final Map<String, Object> overrides, final Path file) {
        for (final String key : overrides.keySet()) {
            if (!"name".equals(key) && !SECTIONS.contains(key)) {
                throw new IllegalArgumentException("Unknown section \"" + key + "\" in " + file + "; expected one of " + SECTIONS);
            }
        }

        final Map<String, Object> merged = new LinkedHashMap<>();
        for (final String section : SECTIONS) {
            merged.put(section, deepMerge(section(defaults, section), section(overrides, section)));
        }
        return merged;
    }

    private static Map<String, Object> deepMerge(final Map<String, Object> base, final Map<String, Object> overlay) {
        final Map<String, Object> result = new LinkedHashMap<>(base);
        overlay.forEach((key, value) -> {
            final Object existing = result.get(key);
            if (existing instanceof Map<?, ?> && value instanceof Map<?, ?>) {
                result.put(key, deepMerge(asMap(existing, key), asMap(value, key)));
            } else {
                result.put(key, value);
            }
        });
        return result;
    }

    private static Scenario scenario(final String name, final Map<String, Object> merged) {
        final Map<String, Object> corpus = checkedSection(merged, "corpus", CORPUS_KEYS);
        final Map<String, Object> index = checkedSection(merged, "index", INDEX_KEYS);
        final Map<String, Object> search = checkedSection(merged, "search", SEARCH_KEYS);
        final Map<String, Object> codec = checkedSection(merged, "codec", CODEC_KEYS);

        final Scenario.CorpusSpec corpusSpec = new Scenario.CorpusSpec(
            string(corpus, "type", "synthetic"),
            integer(corpus, "size", 4_000),
            integer(corpus, "queries", 200),
            integer(corpus, "dimension", 32),
            integer(corpus, "clusters", 16),
            decimal(corpus, "spread", 0.05),
            integer(corpus, "depth", 10),
            number(corpus, "seed", 42L),
            similarity(string(corpus, "similarity", VectorSimilarityFunction.EUCLIDEAN.name())),
            string(corpus, "file", null)
        );
        final Scenario.SearchSpec searchSpec = new Scenario.SearchSpec(
            integer(search, "k", 10),
            integer(search, "queries", 200),
            integer(search, "repeat", 1),
            integer(search, "warmup", 50),
            integer(search, "threads", 1),
            decimal(search, "scoreTolerance", 1.0e-6),

            decimal(search, "scoreBias", decimal(search, "scoreTolerance", 1.0e-6) * 0.1)
        );

        if ("synthetic".equalsIgnoreCase(corpusSpec.type()) && searchSpec.k() > corpusSpec.depth()) {
            throw new IllegalArgumentException(
                "Scenario \""
                    + name
                    + "\" searches k="
                    + searchSpec.k()
                    + " against a generated corpus holding only "
                    + corpusSpec.depth()
                    + " neighbours per query; raise corpus.depth"
            );
        }

        return new Scenario(
            name,
            corpusSpec,
            new Scenario.IndexSpec(integer(index, "docsPerSegment", 1_000), strings(index, "expectFiles"), segments(index)),
            searchSpec,
            new Scenario.CodecSpec(string(codec, "name", DEFAULT_CODEC), params(codec))
        );
    }

    private static Map<String, Object> checkedSection(final Map<String, Object> merged, final String name, final Set<String> allowed) {
        final Map<String, Object> section = section(merged, name);
        for (final String key : section.keySet()) {
            if (!allowed.contains(key)) {
                throw new IllegalArgumentException("Unknown " + name + " key \"" + key + "\"; expected one of " + allowed);
            }
        }
        return section;
    }

    private static Map<String, Object> section(final Map<String, Object> document, final String name) {
        final Object value = document.get(name);
        return value == null ? Map.of() : asMap(value, name);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> params(final Map<String, Object> codec) {
        final Object value = codec.get("params");
        return value == null ? Map.of() : Map.copyOf((Map<String, Object>) asMap(value, "codec.params"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(final Object value, final String what) {
        if (!(value instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException(what + " must be a mapping, found " + value.getClass().getSimpleName());
        }
        return (Map<String, Object>) map;
    }

    private static Integer segments(final Map<String, Object> section) {
        final Object value = section.get("forceMerge");
        if (value == null) {
            return null;
        }
        if (!(value instanceof Number segments)) {
            throw new IllegalArgumentException("\"forceMerge\" must be a number of segments, found \"" + value + "\"");
        }
        if (segments.intValue() < 1) {
            throw new IllegalArgumentException("\"forceMerge\" must be at least 1 segment, got " + segments);
        }
        return segments.intValue();
    }

    private static List<String> strings(final Map<String, Object> section, final String key) {
        final Object value = section.get(key);
        if (value == null) {
            return List.of();
        }
        if (!(value instanceof List<?> items)) {
            throw new IllegalArgumentException("\"" + key + "\" must be a list, found \"" + value + "\"");
        }
        return items.stream().map(Object::toString).toList();
    }

    private static String string(final Map<String, Object> section, final String key, final String fallback) {
        final Object value = section.get(key);
        return value == null ? fallback : value.toString();
    }

    private static int integer(final Map<String, Object> section, final String key, final int fallback) {
        return (int) number(section, key, fallback);
    }

    private static double decimal(final Map<String, Object> section, final String key, final double fallback) {
        final Object value = section.get(key);
        if (value == null) {
            return fallback;
        }
        if (value instanceof Number parsed) {
            return parsed.doubleValue();
        }
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("\"" + key + "\" must be a number, found \"" + value + "\"", e);
        }
    }

    private static long number(final Map<String, Object> section, final String key, final long fallback) {
        final Object value = section.get(key);
        if (value == null) {
            return fallback;
        }
        if (value instanceof Number parsed) {
            return parsed.longValue();
        }
        throw new IllegalArgumentException("\"" + key + "\" must be a number, found \"" + value + "\"");
    }

    private static VectorSimilarityFunction similarity(final String name) {
        try {
            return VectorSimilarityFunction.valueOf(name.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                "Unknown similarity \"" + name + "\". Supported: " + List.of(VectorSimilarityFunction.values()),
                e
            );
        }
    }
}
