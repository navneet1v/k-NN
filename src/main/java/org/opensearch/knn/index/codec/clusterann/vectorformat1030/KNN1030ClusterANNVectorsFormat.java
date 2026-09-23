/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.clusterann.format.QuantizationParams;

import java.io.IOException;

/**
 * A cluster-based approximate nearest neighbor (ANN) {@link KnnVectorsFormat}.
 *
 * <p>This is the entry point that Lucene uses to obtain the per-segment
 * {@link KnnVectorsWriter} and {@link KnnVectorsReader} for fields backed by this
 * format. In this initial implementation the format stores the raw, full-precision
 * vectors through a delegate {@link FlatVectorsFormat} (Lucene's
 * {@link Lucene99FlatVectorsFormat}) and delegates search to it; the cluster-based
 * ANN structures are intended to be layered on top of this raw storage in a later
 * iteration.
 *
 * @see KNN1030ClusterANNVectorsWriter
 * @see KNN1030ClusterANNVectorsReader
 */
@Log4j2
public class KNN1030ClusterANNVectorsFormat extends KnnVectorsFormat {

    /** The unique, human-readable name that identifies this format. */
    public static final String FORMAT_NAME = "KNN1030ClusterANNVectorsFormat";

    /** Codec name written into the header of the {@code .clam} metadata file. */
    static final String META_CODEC_NAME = "KNN1030ClusterANNVectorsFormatMeta";

    /** Extension of the metadata file: one entry per vector field, read once when the segment opens. */
    static final String META_EXTENSION = "clam";

    /** Codec name written into the header of the {@code .clac} centroids file. */
    static final String CENTROIDS_CODEC_NAME = "KNN1030ClusterANNVectorsFormatCentroids";

    /** Extension of the centroids file: each field's centroids, raw and in the space the codes were written in. */
    static final String CENTROIDS_EXTENSION = "clac";

    /** Codec name written into the header of the {@code .clap} postings file. */
    static final String POSTINGS_CODEC_NAME = "KNN1030ClusterANNVectorsFormatPostings";

    /** Extension of the postings file: one posting per cluster, holding its ordinals and quantized vectors. */
    static final String POSTINGS_EXTENSION = "clap";

    /** Codec name written into the header of the {@code .clar} rotation file. */
    static final String ROTATION_CODEC_NAME = "KNN1030ClusterANNVectorsFormatRotation";

    /** Extension of the rotation file: per-field block-diagonal rotation matrix (EUCLIDEAN fields only). */
    static final String ROTATION_EXTENSION = "clar";

    /** Field number that ends the entry list in the metadata file; no real field can have it. */
    static final int NO_MORE_FIELDS = -1;

    /** Vectors per block in the code-block layout; written once to {@code .clam} and shared by every field. */
    static final int DEFAULT_BLOCK_SIZE = 32;

    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;

    /** Delegate format used to store and read the raw, full-precision vectors.
     * TODO: Add prefetchable scorer for .vec */
    private static final FlatVectorsFormat RAW_FLAT_VECTORS_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );

    /** How this format's fields are quantized on write; chosen at index time and recorded per field in {@code .clam}. */
    private final QuantizationParams quantizationParams;

    /**
     * SPI constructor: {@code KnnVectorsFormat.forName(FORMAT_NAME)} needs a public no-arg constructor to
     * instantiate the format when a segment written through {@code PerFieldKnnVectorsFormat} is read back.
     * A reader takes every per-field encoding from {@code .clam}, so the default quantization here only
     * matters if this instance is also used to write.
     */
    public KNN1030ClusterANNVectorsFormat() {
        this(FORMAT_NAME);
    }

    /**
     * Creates a new format instance with the default encoding
     * ({@link QuantizationParams#DEFAULT} — optimized scalar quantization at 2 bits).
     *
     * @param name the name that uniquely identifies this format; must match the name
     *             used when the format is resolved on read
     */
    public KNN1030ClusterANNVectorsFormat(final String name) {
        this(name, QuantizationParams.DEFAULT);
    }

    /**
     * Creates a new format instance with an explicit encoding.
     *
     * @param name         the name that uniquely identifies this format; must match the name used on read
     * @param quantizationParams the quantization backend and bit width to encode fields with
     */
    public KNN1030ClusterANNVectorsFormat(final String name, final QuantizationParams quantizationParams) {
        super(name);
        this.quantizationParams = quantizationParams;
    }

    /**
     * Returns a writer that persists the vectors for a single segment. The returned
     * writer wraps the delegate flat-vectors writer for the given segment.
     *
     * @param segmentWriteState state describing the segment being written
     * @return a {@link KNN1030ClusterANNVectorsWriter} for the segment
     * @throws IOException if the delegate writer cannot be created
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState segmentWriteState) throws IOException {
        return new KNN1030ClusterANNVectorsWriter(segmentWriteState, RAW_FLAT_VECTORS_FORMAT, quantizationParams);
    }

    /**
     * Returns a reader that provides access to the vectors of a single segment. The
     * returned reader wraps the delegate flat-vectors reader for the given segment.
     *
     * @param segmentReadState state describing the segment being read
     * @return a {@link KNN1030ClusterANNVectorsReader} for the segment
     * @throws IOException if the delegate reader cannot be created
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState segmentReadState) throws IOException {
        return new KNN1030ClusterANNVectorsReader(segmentReadState, RAW_FLAT_VECTORS_FORMAT.fieldsReader(segmentReadState));
    }

    /**
     * Returns the maximum number of dimensions a vector field using this format may have.
     *
     * @param s the field name (unused; the limit is the same for every field)
     * @return the maximum supported vector dimensionality
     */
    @Override
    public int getMaxDimensions(String s) {
        return 16000;
    }
}
