/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;

/**
 * An engineless KNN method owns its own format and reader/writer without going through a
 * {@link org.opensearch.knn.index.engine.KNNEngine}. Every implementation supplies four things:
 *
 * <ol>
 *   <li>a unique routing {@link #getName() name} (e.g. "cluster") stored in segment field attributes,</li>
 *   <li>a {@link MethodResolver method resolver} that validates and normalizes user parameters,</li>
 *   <li>a {@link EnginelessMapperFactory mapper factory} that produces the field mapper for the method,</li>
 *   <li>a {@link KnnVectorsFormat} obtained via {@link #resolveKnnVectorsFormat(int)} for the codec layer.</li>
 * </ol>
 *
 * The {@link org.opensearch.knn.index.engine.KNNEngine} value on a resolved {@link KNNMethodContext}
 * for an engineless method is always {@link org.opensearch.knn.index.engine.KNNEngine#UNDEFINED}.
 */
public interface EnginelessMethod {

    /**
     * The routing name for this method. Written to the segment field attribute
     * {@link org.opensearch.knn.common.KNNConstants#KNN_METHOD} and used as the registry key.
     */
    String getName();

    MethodResolver getMethodResolver();

    EnginelessMapperFactory getMapperFactory();

    /**
     * Produce the Lucene codec format used to write and read this method's on-disk structures.
     * Called from the PerField codec resolver; mapping/parse-time paths never invoke this method.
     */
    KnnVectorsFormat resolveKnnVectorsFormat(int bits);
}
