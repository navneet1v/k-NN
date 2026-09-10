package org.opensearch.knn.clusterann.writer;

import org.opensearch.knn.clusterann.reader.block.scalar.ScalarEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.clusterann.reader.ClusterFactory;
import org.opensearch.knn.clusterann.writer.block.scalar.ScalarQuantizedClusterWriter;

/**
 * Picks the {@link ClusterWriter} for a field's storage — the twin of {@code ClusterFactory}, and the only place the
 * write side names a concrete family.
 *
 * <p>The pairing with the read side is by {@code quantizerId}, recorded in {@code .clam}. Adding a family is a case
 * here and a case there; nothing between them changes.
 */
public final class ClusterWriterFactory {

    private ClusterWriterFactory() {}

    /**
     * @param params the write parameters, of which {@code quantizerId}, {@code docBits} and {@code blockSize} decide
     *     the layout and are all recorded in {@code .clam}
     * @param similarity the field's metric, which decides what the per-vector corrective term means
     */
    public static ClusterWriter create(ClusterANNWriteParams params, int dimension, VectorSimilarityFunction similarity) {
        if (params.quantizerId() != ClusterFactory.QUANTIZER_SQ) {
            throw new IllegalArgumentException("Unsupported quantizerId: " + params.quantizerId());
        }
        ScalarEncoding encoding = encoding(params.docBits());
        return new ScalarQuantizedClusterWriter(
            params.blockSize(),
            dimension,
            encoding,
            new OptimizedScalarQuantizer(similarity),
            similarity
        );
    }

    /** Maps the code width to its encoding, so the bit count is the only source of truth on both sides. */
    private static ScalarEncoding encoding(int docBits) {
        try {
            return ScalarEncoding.fromNumBits(docBits);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Unsupported docBits: " + docBits, e);
        }
    }
}
