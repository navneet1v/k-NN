/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

/**
 * One cluster's centroid together with its squared norm.
 *
 * <p>The two travel as one value because every score needs both and ‖c‖² is not recoverable for free — deriving
 * it per query would cost a pass over the centroid, and per posting would cost one per cluster visited. It is
 * computed once, wherever the centroid itself is read from {@code .clac}.
 *
 * @param vector the centroid, not copied.
 * @param normSq ‖c‖².
 */
public record Centroid(float[] vector, float normSq) {
}
