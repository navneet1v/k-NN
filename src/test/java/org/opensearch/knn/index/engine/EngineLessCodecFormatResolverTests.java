/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1040Codec.ClusterANN1040KnnVectorsFormat;

import java.util.Collections;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;

public class EngineLessCodecFormatResolverTests extends KNNTestCase {

    private final EngineLessCodecFormatResolver resolver = new EngineLessCodecFormatResolver();

    public void testResolve_cluster() {
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.UNDEFINED,
            SpaceType.L2,
            new MethodComponentContext(METHOD_CLUSTER, Collections.emptyMap())
        );
        KnnVectorsFormat format = resolver.resolve("test-field", methodContext, null, 16, 100);
        assertTrue(format instanceof ClusterANN1040KnnVectorsFormat);
    }

    public void testResolve_unknownMethod() {
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.UNDEFINED,
            SpaceType.L2,
            new MethodComponentContext("unknown_algo", Collections.emptyMap())
        );
        expectThrows(IllegalArgumentException.class, () -> resolver.resolve("test-field", methodContext, null, 16, 100));
    }

    public void testResolve_noArg_throws() {
        expectThrows(UnsupportedOperationException.class, () -> resolver.resolve());
    }
}
