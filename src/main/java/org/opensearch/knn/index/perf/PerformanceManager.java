/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.perf;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class PerformanceManager {
    private static PerformanceManager INSTANCE;
    private static Path LOG_FILE_PATH;

    private final String PERFORMANCE_STATS_FILE_NAME = "knn-perf-stats.log";

    public static synchronized PerformanceManager getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new PerformanceManager();
        }
        return INSTANCE;
    }

    public static void setLogFilePath(Path logFilePath) {
        LOG_FILE_PATH = logFilePath;
    }

    public Map<String, Object> getPerformanceStats() {
        final Map<String, Object> stats = new LinkedHashMap<>();
        KNNCounter.KNN_PERF_STATS.increment();
        writeStatsToFile();
        stats.put(KNNCounter.KNN_PERF_STATS.getName(), KNNCounter.KNN_PERF_STATS.getCount().toString());
        Arrays.stream(PerformanceStats.values()).map(PerformanceStats::getStats).forEach(stats::putAll);
        return stats;
    }

    private void writeStatsToFile() {
        final String statsFileName = LOG_FILE_PATH.toAbsolutePath() + File.separator + PERFORMANCE_STATS_FILE_NAME;
        File statsFile = null;
        FileWriter fr = null;
        BufferedWriter br = null;
        try {
            statsFile = new File(statsFileName);
            boolean fileCreated = statsFile.exists();
            if (!fileCreated) {
                fileCreated = statsFile.createNewFile();
            }
            if (fileCreated) {
                fr = new FileWriter(statsFile, true);
                br = new BufferedWriter(fr);
                writeHeader(br);
                writeStats(br);
                br.close();
                fr.close();
            } else {
                log.error(
                    "Unable to write stats in the k-NN perf stats file as the file is not created. File Name: " + "{}",
                    statsFileName
                );
            }
        } catch (Exception e) {
            log.error("Error happened while writing k-NN stats in stats file {}", statsFileName, e);
        }
    }

    private void writeHeader(BufferedWriter br) throws IOException {
        br.write("---------------------- " + KNNCounter.KNN_PERF_STATS.getCount() + " ------------------------");
        br.newLine();
        final PerformanceStats[] statsArray = PerformanceStats.values();
        for (int i = 0; i < statsArray.length; i++) {
            br.write(statsArray[i].getStatsHeaderName());
            if (i + 1 < statsArray.length) {
                br.write(",");
            }
        }
        br.newLine();
    }

    private void writeStats(BufferedWriter br) throws IOException {
        final List<List<Double>> latenciesList = Arrays.stream(PerformanceStats.values())
            .map(PerformanceStats::getLatencies)
            .collect(Collectors.toList());
        for (int i = 0;; i++) {
            boolean canBreak = true;
            for (List<Double> doubles : latenciesList) {
                if (i < doubles.size()) {
                    br.write(doubles.get(i).toString() + ",");
                    canBreak = false;
                }
            }
            br.newLine();
            if (canBreak) {
                break;
            }
        }
    }

}
