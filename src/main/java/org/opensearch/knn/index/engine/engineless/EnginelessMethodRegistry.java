/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless;

import org.opensearch.knn.index.engine.engineless.cluster.ClusterANNMethod;

import java.util.Map;
import java.util.Optional;

/**
 * Static registry of engineless KNN methods. The set of methods is fixed at class load time.
 * Adding a new engineless method is a source-level change to {@link #METHODS}.
 *
 * <p>Consulted at:
 * <ul>
 *   <li>mapping parse time — to skip engine resolution when the method is engineless,</li>
 *   <li>field-mapper build time — to dispatch to the method's own {@link EnginelessMapperFactory},</li>
 *   <li>codec time — by {@code EnginelessCodecFormatResolver} to pick the method's {@code KnnVectorsFormat}.</li>
 * </ul>
 *
 * Thread-safe by construction: {@link Map#of} returns a genuinely immutable map, and the
 * {@code static final} field is safely published by the JMM's class-init guarantees.
 */
public final class EnginelessMethodRegistry {

    private static final Map<String, EnginelessMethod> METHODS = Map.of(ClusterANNMethod.INSTANCE.getName(), ClusterANNMethod.INSTANCE);

    private EnginelessMethodRegistry() {}

    /**
     * Lookup by method routing name. Returns empty if the name is null, empty, or not registered.
     */
    public static Optional<EnginelessMethod> get(String name) {
        if (name == null || name.isEmpty()) {
            return Optional.empty();
        }
        return Optional.ofNullable(METHODS.get(name));
    }

    /**
     * Returns {@code true} if {@code name} is the routing name of a registered engineless method.
     */
    public static boolean isEnginelessMethod(String name) {
        return get(name).isPresent();
    }
}
