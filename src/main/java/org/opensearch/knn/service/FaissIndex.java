/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.service;

import org.opensearch.knn.jni.FaissService;

import java.util.Map;

public class FaissIndex {

    private long memoryAddress = 0;

    // We should be calling this only once. Ideally we should call this once when the shard is getting created.
    // We will figure out a way to do this, later
    public void initIndex(int dim, Map<String, Object> parameters) {
        memoryAddress = FaissService.initIndex(0, dim , parameters);
    }

    public void indexData(int docId, float[] vector) {

    }

    public void searchIndex(float[] queryVector, int k) {

    }



}
