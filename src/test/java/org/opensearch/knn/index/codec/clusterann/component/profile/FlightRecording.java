/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.profile;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import jdk.jfr.Configuration;
import jdk.jfr.Recording;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.Locale;

/** A JFR recording per phase, written to {@code build/jfr} when {@code -Dknn.jfr=true}, otherwise a no-op. */
public final class FlightRecording implements AutoCloseable {
    private static final Logger log = LogManager.getLogger(FlightRecording.class);

    private static final String ENABLED = "knn.jfr";
    private static final String SETTINGS = "knn.jfr.settings";
    private static final String DIRECTORY = "knn.jfr.dir";

    private final Recording recording;
    private final Path file;

    private FlightRecording(Recording recording, Path file) {
        this.recording = recording;
        this.file = file;
    }

    public static FlightRecording start(String label) {
        if (!Boolean.getBoolean(ENABLED)) {
            return new FlightRecording(null, null);
        }

        final String settings = System.getProperty(SETTINGS, "profile");
        final Configuration configuration;
        try {
            configuration = Configuration.getConfiguration(settings);
        } catch (IOException | ParseException e) {
            throw new IllegalArgumentException(
                "No JFR settings profile \""
                    + settings
                    + "\". Available: "
                    + Configuration.getConfigurations().stream().map(Configuration::getName).toList(),
                e
            );
        }

        try {
            final Path directory = Paths.get(System.getProperty(DIRECTORY, Paths.get("build", "jfr").toString()));
            Files.createDirectories(directory);
            final Path target = directory.resolve(sanitize(label) + ".jfr");

            final Recording recording = new Recording(configuration);
            recording.setName(label);
            recording.start();
            log.info("JFR recording \"{}\" started with the {} settings", label, settings);
            return new FlightRecording(recording, target);
        } catch (IOException | RuntimeException e) {
            log.warn("Could not start a JFR recording for \"{}\"; continuing without one", label, e);
            return new FlightRecording(null, null);
        }
    }

    @Override
    public void close() {
        if (recording == null) {
            return;
        }
        try (Recording open = recording) {
            open.dump(file);
            log.info("JFR recording \"{}\" written to {} ({} bytes)", open.getName(), file.toAbsolutePath(), Files.size(file));
        } catch (IOException | RuntimeException e) {
            log.warn("Could not write the JFR recording for \"{}\" to {}", recording.getName(), file, e);
        }
    }

    private static String sanitize(String label) {
        return label.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9._-]+", "-");
    }

    public static final class JfrThreadFilter implements ThreadFilter {
        public JfrThreadFilter() {}

        @Override
        public boolean reject(Thread thread) {
            return thread.getName().startsWith("JFR ");
        }
    }
}
