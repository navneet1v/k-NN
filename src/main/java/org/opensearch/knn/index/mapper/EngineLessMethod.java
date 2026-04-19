/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.ClusterANN1040KnnVectorsFormat;

import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;

/**
 * Enum representing engine-less ANN algorithms that bypass the KNNEngine layer.
 * Each value carries its mapper factory and codec format supplier.
 */
@Getter
public enum EngineLessMethod {
    CLUSTER(METHOD_CLUSTER, ClusterANN1040KnnVectorsFormat::new);

    private final String name;
    private final Supplier<KnnVectorsFormat> formatSupplier;

    EngineLessMethod(String name, Supplier<KnnVectorsFormat> formatSupplier) {
        this.name = name;
        this.formatSupplier = formatSupplier;
    }

    /**
     * Returns a new instance of the {@link KnnVectorsFormat} for this algorithm.
     *
     * @return the vectors format
     */
    public KnnVectorsFormat getFormat() {
        return formatSupplier.get();
    }

    /**
     * Get the enum value from a method name string.
     *
     * @param name the method name
     * @return the enum value, or null if not an engine-less method
     */
    public static EngineLessMethod fromName(String name) {
        if (name == null) {
            return null;
        }
        for (EngineLessMethod method : values()) {
            if (method.name.equals(name)) {
                return method;
            }
        }
        return null;
    }

    /**
     * Check if the given method name is an engine-less algorithm.
     *
     * @param name the method name
     * @return true if engine-less
     */
    public static boolean isEngineLess(String name) {
        return fromName(name) != null;
    }
}
