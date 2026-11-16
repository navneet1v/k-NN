/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.parallel;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.TaskExecutor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class ParallelVectorProcessorTests {

    private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);
    private static final TaskExecutor TASK_EXECUTOR = new TaskExecutor(EXECUTOR);

    @AfterAll
    static void shutdown() {
        EXECUTOR.shutdownNow();
    }

    @Test
    void inlineExecution_coversFullRangeOnce() throws IOException {
        FloatVectorValues source = createTestVectors(100, 8);
        AtomicInteger visited = new AtomicInteger(0);

        ParallelVectorProcessor.execute(source, 100, null, (start, end, view) -> {
            assertEquals(0, start);
            assertEquals(100, end);
            for (int i = start; i < end; i++) {
                view.vectorValue(i);
                visited.incrementAndGet();
            }
        });

        assertEquals(100, visited.get());
    }

    @Test
    void parallelExecution_visitsEveryOrdinalExactlyOnce() throws IOException {
        int n = 1000;
        FloatVectorValues source = createTestVectors(n, 4);
        int[] hits = new int[n];

        ParallelVectorProcessor.execute(source, n, TASK_EXECUTOR, (start, end, view) -> {
            for (int i = start; i < end; i++) {
                float[] vec = view.vectorValue(i);
                assertEquals(4, vec.length);
                synchronized (hits) {
                    hits[i]++;
                }
            }
        });

        for (int i = 0; i < n; i++) {
            assertEquals(1, hits[i], "ordinal " + i + " must be processed exactly once");
        }
    }

    @Test
    void parallelExecution_propagatesIOException() {
        FloatVectorValues source = createTestVectors(50, 4);
        try {
            ParallelVectorProcessor.execute(source, 50, TASK_EXECUTOR, (start, end, view) -> { throw new IOException("boom"); });
            fail("expected IOException to propagate");
        } catch (IOException e) {
            assertTrue(e.getMessage().contains("boom"));
        }
    }

    @Test
    void inlineExecution_propagatesIOException() {
        FloatVectorValues source = createTestVectors(10, 4);
        try {
            ParallelVectorProcessor.execute(source, 10, null, (start, end, view) -> { throw new IOException("inline-boom"); });
            fail("expected IOException to propagate");
        } catch (IOException e) {
            assertTrue(e.getMessage().contains("inline-boom"));
        }
    }

    @Test
    void parallelExecution_propagatesRuntimeException() {
        FloatVectorValues source = createTestVectors(50, 4);
        try {
            ParallelVectorProcessor.execute(
                source,
                50,
                TASK_EXECUTOR,
                (start, end, view) -> { throw new IllegalStateException("rt-boom"); }
            );
            fail("expected RuntimeException to propagate");
        } catch (IOException e) {
            fail("RuntimeException should not be wrapped as IOException: " + e);
        } catch (RuntimeException e) {
            assertTrue(e.getMessage().contains("rt-boom"));
        }
    }

    @Test
    void parallelExecution_fewerOrdinalsThanWorkers_leavesEmptySlices() throws IOException {
        // total (2) < DEFAULT_WORKERS (4): later slices are empty and hit the start>=end break.
        int n = 2;
        FloatVectorValues source = createTestVectors(n, 4);
        int[] hits = new int[n];

        ParallelVectorProcessor.execute(source, n, TASK_EXECUTOR, (start, end, view) -> {
            for (int i = start; i < end; i++) {
                synchronized (hits) {
                    hits[i]++;
                }
            }
        });

        for (int i = 0; i < n; i++) {
            assertEquals(1, hits[i], "ordinal " + i + " processed exactly once");
        }
    }

    private FloatVectorValues createTestVectors(int n, int dim) {
        List<float[]> vecs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            v[0] = (float) i;
            for (int d = 1; d < dim; d++)
                v[d] = (float) Math.sin(i + d);
            vecs.add(v);
        }
        return FloatVectorValues.fromFloats(vecs, dim);
    }
}
