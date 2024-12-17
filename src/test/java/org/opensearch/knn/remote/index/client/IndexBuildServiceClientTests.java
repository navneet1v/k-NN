/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.remote.index.client;

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.remote.index.model.CreateIndexResponse;

public class IndexBuildServiceClientTests extends KNNTestCase {

    @SneakyThrows
    public void testResponseParsing() {
        String response = "{\"indexCreationRequestId\":\"1234211\",\"status\":\"in_progress\"}";
        CreateIndexResponse response1 = IndexBuildServiceClient.parseCreateIndexResponse(response);
        assertEquals("1234211", response1.getIndexCreationRequestId());
        assertEquals("in_progress", response1.getStatus());
    }

}
