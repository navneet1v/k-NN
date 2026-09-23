/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.baseline;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Recall and latency per scenario, in one committed CSV per suite, so a change that costs recall shows in a diff. */
public final class Baselines {
    private static final Path DIRECTORY = Paths.get("src/test/resources/baselines");

    private static final String HEADER = "scenario,corpus,codec,k,queries,recall,p50Millis,p90Millis";

    private final Path file;

    private final Map<Key, Measurement> rows;

    private Baselines(Path file, Map<Key, Measurement> rows) {
        this.file = file;
        this.rows = rows;
    }

    public record Key(String scenario, String corpus, String codec, int k, int queries) {
        @Override
        public String toString() {
            return scenario + "/" + codec + "/recall@" + k + "/" + queries + " queries";
        }
    }

    public record Measurement(double recall, double p50Millis, double p90Millis) {
        public String toCsv(Key key) {
            return String.join(
                ",",
                key.scenario(),
                key.corpus(),
                key.codec(),
                Integer.toString(key.k()),
                Integer.toString(key.queries()),
                Double.toString(recall),
                Double.toString(p50Millis),
                Double.toString(p90Millis)
            );
        }
    }

    public static Baselines forSuite(final Path suiteFile) throws IOException {
        final Path file = fileFor(suiteFile);
        final Map<Key, Measurement> rows = new LinkedHashMap<>();
        if (Files.notExists(file)) {
            return new Baselines(file, rows);
        }
        for (final String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            final String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#") || trimmed.equals(HEADER)) {
                continue;
            }
            final String[] fields = trimmed.split(",");
            if (fields.length != 8) {
                throw new IOException("Malformed baseline row, expected 8 fields: " + line);
            }
            rows.put(
                new Key(fields[0], fields[1], fields[2], Integer.parseInt(fields[3]), Integer.parseInt(fields[4])),
                new Measurement(Double.parseDouble(fields[5]), Double.parseDouble(fields[6]), Double.parseDouble(fields[7]))
            );
        }
        return new Baselines(file, rows);
    }

    private static Path fileFor(final Path suiteFile) {
        final String name = suiteFile.getFileName().toString();
        final int extension = name.lastIndexOf('.');
        return DIRECTORY.resolve((extension < 0 ? name : name.substring(0, extension)) + ".csv");
    }

    public Path file() {
        return file;
    }

    public Optional<Measurement> find(Key key) {
        return Optional.ofNullable(rows.get(key));
    }

    public void update(Key key, Measurement measured) throws IOException {
        rows.put(key, measured);

        final List<String> lines = new ArrayList<>();
        lines.add("# Recall and latency per set of parameters. Regenerate a row by rerunning with -Dknn.updateBaselines.");
        lines.add("# Latency is machine-dependent and recorded for comparison; only recall is asserted by default.");
        lines.add(HEADER);
        rows.forEach((rowKey, rowValue) -> lines.add(rowValue.toCsv(rowKey)));

        Files.createDirectories(file.getParent());
        Files.write(file, lines, StandardCharsets.UTF_8);
    }
}
