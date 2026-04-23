/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.engine.MethodResolver;

import java.util.HashMap;
import java.util.Map;

/**
 * Interface for engine-less ANN algorithms that bypass the KNNEngine layer.
 * Each implementation provides its own mapper factory, codec format factory, and method resolver.
 *
 * <p>To add a new engine-less algorithm:
 * <ol>
 *   <li>Implement this interface (see {@link ClusterANNMethod} for an example)</li>
 *   <li>Self-register via {@link #register(EngineLessMethod)} in a static initializer</li>
 *   <li>Ensure the class is loaded at startup (e.g., referenced from {@code KNNVectorFieldMapper})</li>
 * </ol>
 */
public interface EngineLessMethod {

    /**
     * Returns the method name as specified in the mapping's {@code method.name} field.
     * This is the routing key used to identify the algorithm (e.g., "cluster").
     *
     * @return the method name, never null
     */
    String getName();

    /**
     * Returns the factory for creating the field mapper for this algorithm.
     * The mapper handles field type creation, validation, and vector ingestion.
     *
     * @return the mapper factory, never null
     */
    EngineLessMapperFactory getMapperFactory();

    /**
     * Creates a new codec format instance for this algorithm with the specified quantization.
     * The format produces the {@link org.apache.lucene.codecs.KnnVectorsWriter} and
     * {@link org.apache.lucene.codecs.KnnVectorsReader} for indexing and search.
     *
     * @param docBits quantization bits per dimension (1, 2, or 4)
     * @return a new format instance, never null
     */
    KnnVectorsFormat createFormat(int docBits);

    /**
     * Returns the method resolver that reconciles {@code compression_level} and {@code encoder}
     * parameters, validates conflicts, and produces a {@link org.opensearch.knn.index.engine.ResolvedMethodContext}.
     *
     * @return the method resolver, never null
     */
    MethodResolver getMethodResolver();

    // ======================== Static Registry ========================

    /** Internal registry mapping method names to implementations. */
    Map<String, EngineLessMethod> REGISTRY = new HashMap<>();

    /**
     * Registers an engine-less method implementation. Must be called during class initialization
     * only (e.g., from a static initializer). The JVM guarantees that static initializers are
     * thread-safe, so no synchronization is needed. Reads via {@link #fromName} are safe after
     * class loading completes due to the happens-before relationship established by class
     * initialization.
     *
     * @param method the method to register, must not be null
     * @throws IllegalStateException if a method with the same name is already registered
     */
    static void register(EngineLessMethod method) {
        if (REGISTRY.containsKey(method.getName())) {
            throw new IllegalStateException(
                "Engine-less method '" + method.getName() + "' is already registered by "
                    + REGISTRY.get(method.getName()).getClass().getName()
            );
        }
        REGISTRY.put(method.getName(), method);
    }

    /**
     * Looks up an engine-less method by its name.
     *
     * @param name the method name (e.g., "cluster"), may be null
     * @return the registered method, or null if not found or name is null
     */
    static EngineLessMethod fromName(String name) {
        return name == null ? null : REGISTRY.get(name);
    }

    /**
     * Checks whether the given method name corresponds to a registered engine-less algorithm.
     *
     * @param name the method name, may be null
     * @return true if the name is a registered engine-less method
     */
    static boolean isEngineLess(String name) {
        return fromName(name) != null;
    }
}
