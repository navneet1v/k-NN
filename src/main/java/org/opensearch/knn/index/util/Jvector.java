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

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.SpaceType;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

public class Jvector extends JVMLibrary {

    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
            METHOD_HNSW,
            KNNMethod.Builder.builder(
                    MethodComponent.Builder.builder(METHOD_HNSW)
                            .addParameter(
                                    METHOD_PARAMETER_M,
                                    new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0)
                            )
                            .addParameter(
                                    METHOD_PARAMETER_EF_CONSTRUCTION,
                                    new Parameter.IntegerParameter(
                                            METHOD_PARAMETER_EF_CONSTRUCTION,
                                            KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                                            v -> v > 0
                                    )
                            )
                            .build()
            ).addSpaces(SpaceType.L2).build()
    );

    final static Jvector INSTANCE = new Jvector(METHODS, Version.LATEST.toString());

    /**
     * Constructor
     *
     * @param methods Map of k-NN methods that the library supports
     * @param version String representing version of library
     */
    Jvector(Map<String, KNNMethod> methods, String version) {
        super(methods, version);
    }

    /**
     * Gets the extension that files written with this library should have
     *
     * @return extension
     */
    @Override
    public String getExtension() {
        return ".inline";
    }

    /**
     * Gets the compound extension that files written with this library should have
     *
     * @return compound extension
     */
    @Override
    public String getCompoundExtension() {
        throw new UnsupportedOperationException("Getting compound extension for JVector is not supported");
    }

    /**
     * Generate the Lucene score from the rawScore returned by the library. With k-NN, often times the library
     * will return a score where the lower the score, the better the result. This is the opposite of how Lucene scores
     * documents.
     *
     * @param rawScore  returned by the library
     * @param spaceType spaceType used to compute the score
     * @return Lucene score for the rawScore
     */
    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return 0;
    }

    /**
     * Translate the distance radius input from end user to the engine's threshold.
     *
     * @param distance  distance radius input from end user
     * @param spaceType spaceType used to compute the radius
     * @return transformed distance for the library
     */
    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return null;
    }

    /**
     * Translate the score threshold input from end user to the engine's threshold.
     *
     * @param score     score threshold input from end user
     * @param spaceType spaceType used to compute the threshold
     * @return transformed score for the library
     */
    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return null;
    }
}
