/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103Codec;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat;

/**
 * Test-only codec that plugs {@link KNN1030ClusterANNVectorsFormat} into a standard Lucene 10.3 codec, registered via
 * SPI so a reader can resolve it. Not for production use.
 */
public class ClusterANN1030TestCodec extends FilterCodec {

    /** SPI name under which this codec is registered; must match the META-INF/services entry. */
    public static final String CODEC_NAME = "ClusterANN1030TestCodec";

    private final QuantizationParams quantizationParams;

    /** Required by SPI ({@code ServiceLoader}); uses the format's default quantization. */
    public ClusterANN1030TestCodec() {
        this(QuantizationParams.DEFAULT);
    }

    /** At an explicit code width, which is how a scenario's {@code codec.params} reaches the format. */
    public ClusterANN1030TestCodec(final int docBits) {
        this(QuantizationParams.of(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, docBits));
    }

    /** At an explicit quantization. */
    public ClusterANN1030TestCodec(final QuantizationParams quantizationParams) {
        super(CODEC_NAME, new Lucene103Codec());
        this.quantizationParams = quantizationParams;
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new KNN1030ClusterANNVectorsFormat(KNN1030ClusterANNVectorsFormat.FORMAT_NAME, quantizationParams);
    }
}
