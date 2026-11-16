/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.parallel;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.TaskExecutor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Runs a partitioned, thread-safe parallel loop over {@link FloatVectorValues}.
 *
 * <p>Work is split into contiguous ordinal ranges, one per worker. Each worker
 * receives its own independent {@link FloatVectorValues#copy()} so that concurrent
 * reads never share a backing reader — the encrypted {@code CryptoBufferedIndexInput}
 * is not thread-safe, and a shared reader corrupts state under parallel seeks.
 *
 * <p>Parallelism is driven by a caller-supplied Lucene {@link TaskExecutor} (typically
 * threaded down from the merge, e.g. {@code KnnVectorsWriter}'s merge {@code TaskExecutor}).
 * {@link TaskExecutor} owns task submission, result collection, exception propagation, and
 * sibling cancellation on failure, so this class only splits the work and enforces the
 * per-worker vector copy. When the executor is {@code null} the loop runs inline on the
 * calling thread, so callers do not need a separate sequential code path.
 */
public final class ParallelVectorProcessor {

    /** Default worker count when running in parallel. */
    public static final int DEFAULT_WORKERS = 4;

    private ParallelVectorProcessor() {}

    /**
     * Execute {@code task} over {@code [0, total)} split across workers.
     *
     * @param vectors      source partitioned across workers; each worker gets its own copy
     * @param total        number of ordinals to process
     * @param taskExecutor Lucene executor to run on, or {@code null} for inline sequential execution
     * @param task         work performed for a contiguous ordinal range on a private vector view
     */
    public static void execute(FloatVectorValues vectors, int total, TaskExecutor taskExecutor, VectorRangeTask task) throws IOException {
        if (taskExecutor == null) {
            task.run(0, total, vectors);
            return;
        }

        int workers = Math.min(DEFAULT_WORKERS, Math.max(1, total));
        int sliceSize = (total + workers - 1) / workers;

        List<Callable<Void>> tasks = new ArrayList<>(workers);
        for (int w = 0; w < workers; w++) {
            int start = w * sliceSize;
            int end = Math.min(start + sliceSize, total);
            if (start >= end) {
                break;
            }
            // Each worker reads through its own view: the backing CryptoBufferedIndexInput is not
            // thread-safe, so workers must never share a reader.
            FloatVectorValues workerView = vectors.copy();
            tasks.add(() -> {
                task.run(start, end, workerView);
                return null;
            });
        }

        // TaskExecutor#invokeAll runs the tasks (some on the calling thread), collects results,
        // propagates a worker's IOException / RuntimeException / Error unchanged, and cancels
        // remaining tasks if one fails — so no manual future management is needed here.
        taskExecutor.invokeAll(tasks);
    }

    /**
     * A unit of work over a contiguous ordinal range {@code [start, end)} using a
     * private {@link FloatVectorValues} view that is safe to read on the worker thread.
     */
    @FunctionalInterface
    public interface VectorRangeTask {
        void run(int start, int end, FloatVectorValues vectors) throws IOException;
    }
}
