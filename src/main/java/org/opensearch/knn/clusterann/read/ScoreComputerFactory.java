/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read;

import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;

import javax.annotation.Nullable;

/**
 * Supplies the {@link ScoreComputer} a field is scored with. Passed to the format's constructor and consulted once per
 * segment field when the reader opens, after the field's metadata has been read, so the choice can depend on the
 * quantizer, the code width, the similarity and the dimension recorded for that field.
 *
 * <p>Returning {@code null} selects the format's own Java implementation for that field. The format never calls the
 * factory again for the same field of the same segment.
 *
 * <p>Segments open concurrently, so {@link #create} may be invoked from several threads at once and must be
 * thread-safe.
 */
public interface ScoreComputerFactory {

    /** Always defers to the format's Java implementation. What the no-arg format constructor uses. */
    ScoreComputerFactory NONE = (quantizer, meta) -> null;

    /**
     * Chooses the computer for one field of one segment.
     *
     * @param quantizer the quantization backend the field's codes were written with
     * @param meta the field's entry from {@code .clam}: code width, similarity, dimension, block size and layout
     * @return the computer to score this field with, or {@code null} for the format's Java implementation
     */
    @Nullable
    ScoreComputer create(ClusterANNEncoding quantizer, ClusterANNFieldMeta meta);
}
