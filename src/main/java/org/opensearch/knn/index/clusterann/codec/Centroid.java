/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

/**
 * One cluster's centroid geometry in scoring space: the vector and its squared norm, read together in a
 * single seek because every use needs both.
 *
 * <p>Exists so a cluster can be handed <em>its own</em> geometry — as an {@link org.apache.lucene.util.IOSupplier}
 * it can invoke once, lazily — rather than a view over the field's centroids. A cluster has no business
 * being able to address another cluster's centroid, or knowing that centroids live in a file at all.
 *
 * @param vector the centroid, in the space the stored codes were written in
 * @param normSq {@code ‖c‖²}
 */
record Centroid(float[] vector, float normSq) {}
