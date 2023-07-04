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

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.perf.PerformanceManager;

import java.util.function.Function;
import java.util.function.Supplier;

public class KNNPerfStatsSupplier<T> implements Supplier<T> {
    private final Function<PerformanceManager, T> getter;

    public KNNPerfStatsSupplier(Function<PerformanceManager, T> getter) {
        this.getter = getter;
    }

    /**
     * Gets a result.
     *
     * @return a result
     */
    @Override
    public T get() {
        return getter.apply(PerformanceManager.getInstance());
    }
}
