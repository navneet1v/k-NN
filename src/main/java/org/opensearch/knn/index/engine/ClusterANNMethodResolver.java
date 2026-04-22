/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Method resolver for the cluster ANN algorithm. Follows the same resolution pattern as
 * {@link org.opensearch.knn.index.engine.faiss.FaissMethodResolver} — reconciles compression level
 * and encoder parameters, validates conflicts, and produces a {@link ResolvedMethodContext}.
 *
 * <p>Cluster ANN uses scalar quantization at 1, 2, or 4 bits per dimension. The quantization
 * can be specified via top-level {@code compression_level} or via an explicit encoder in
 * {@code method.parameters.encoder}.
 */
public class ClusterANNMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION = Set.of(
        CompressionLevel.x8,
        CompressionLevel.x16,
        CompressionLevel.x32
    );

    private static final Map<String, Encoder> ENCODER_MAP = Map.of(ClusterANNSQEncoder.NAME, new ClusterANNSQEncoder());

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        // Create a resolved copy
        KNNMethodContext resolvedMethodContext = new KNNMethodContext(
            KNNEngine.UNDEFINED,
            knnMethodContext.getSpaceType(),
            new MethodComponentContext(knnMethodContext.getMethodComponentContext()),
            false
        );

        // Resolve encoder: if compression is set but encoder is not, create the encoder
        resolveEncoder(resolvedMethodContext, knnMethodConfigContext);

        // Calculate compression from the (possibly just-created) encoder
        CompressionLevel resolvedCompression = resolveCompressionLevelFromMethodContext(
            resolvedMethodContext,
            knnMethodConfigContext,
            ENCODER_MAP
        );

        // Validate no conflict between user-specified compression and encoder-derived compression
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompression);

        return ResolvedMethodContext.builder().knnMethodContext(resolvedMethodContext).compressionLevel(resolvedCompression).build();
    }

    private void resolveEncoder(KNNMethodContext resolvedMethodContext, KNNMethodConfigContext configContext) {
        // If encoder is already specified by the user, nothing to resolve
        if (isEncoderSpecified(resolvedMethodContext)) {
            return;
        }

        // If compression is not configured, use default (1-bit / x32)
        CompressionLevel compression = configContext.getCompressionLevel();
        int bits;
        if (!CompressionLevel.isConfigured(compression)) {
            bits = ClusterANNSQEncoder.DEFAULT_BITS;
        } else if (SUPPORTED_COMPRESSION.contains(compression)) {
            bits = compressionToBits(compression);
        } else {
            return;
        }

        // Create encoder and set it on the method context
        MethodComponentContext encoderContext = new MethodComponentContext(
            ClusterANNSQEncoder.NAME,
            new HashMap<>(Map.of(ClusterANNSQEncoder.BITS_PARAM, bits))
        );
        resolvedMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderContext);
    }

    private static int compressionToBits(CompressionLevel level) {
        return switch (level) {
            case x8 -> 4;
            case x16 -> 2;
            case x32 -> 1;
            default -> ClusterANNSQEncoder.DEFAULT_BITS;
        };
    }
}
