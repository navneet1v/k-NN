/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

/**
 * Place holder for the score of the document
 */
public class KNNQueryResult {
    private final int id;
    private final float score;

    private final long latency;

    public KNNQueryResult(final int id, final float score) {
        this.id = id;
        this.score = score;
        this.latency = 0;
    }

    // Don't want to use Lombok here as, this class is used for generating the JNI Code
    public KNNQueryResult(final int id, final float score, long latency) {
        this.id = id;
        this.score = score;
        this.latency = latency;
    }

    public int getId() {
        return this.id;
    }

    public float getScore() {
        return this.score;
    }

    public long getLatency() {
        return latency;
    }
}
