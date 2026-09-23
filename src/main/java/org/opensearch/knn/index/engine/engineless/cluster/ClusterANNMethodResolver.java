/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.engineless.cluster;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * ClusterANN's resolver. Supports {@code compression_level} (8x, 16x, 32x; 8x default) and an
 * optional {@code encoder} block that must agree with the compression level. Uses the shared
 * {@link AbstractMethodResolver} helpers to reconcile the two.
 */
public class ClusterANNMethodResolver extends AbstractMethodResolver {

    private static final Set<SpaceType> SUPPORTED_SPACE_TYPES = Set.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION = Set.of(
        CompressionLevel.x8,
        CompressionLevel.x16,
        CompressionLevel.x32
    );

    private static final CompressionLevel DEFAULT_COMPRESSION = CompressionLevel.x8;

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        ValidationException validationException = ClusterANNMethod.METHOD_COMPONENT.validate(
            knnMethodContext.getMethodComponentContext(),
            knnMethodConfigContext
        );
        if (validationException != null) {
            throw validationException;
        }
        validateSpaceType(spaceType);
        validateMode(knnMethodConfigContext);
        validateCompressionSupported(knnMethodConfigContext.getCompressionLevel());

        KNNMethodContext resolvedMethodContext = new KNNMethodContext(
            KNNEngine.UNDEFINED,
            spaceType,
            new MethodComponentContext(knnMethodContext.getMethodComponentContext())
        );

        resolveEncoder(resolvedMethodContext, knnMethodConfigContext);

        CompressionLevel resolvedCompression = resolveCompressionLevelFromMethodContext(
            resolvedMethodContext,
            knnMethodConfigContext,
            ClusterANNMethod.SUPPORTED_ENCODERS
        );

        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompression);

        return ResolvedMethodContext.builder().knnMethodContext(resolvedMethodContext).compressionLevel(resolvedCompression).build();
    }

    private void resolveEncoder(KNNMethodContext resolvedMethodContext, KNNMethodConfigContext configContext) {
        if (isEncoderSpecified(resolvedMethodContext)) {
            return;
        }

        CompressionLevel compression = configContext.getCompressionLevel();
        int bits = ClusterANNSQEncoder.compressionLevelToBits(
            CompressionLevel.isConfigured(compression) ? compression : DEFAULT_COMPRESSION
        );

        MethodComponentContext encoderContext = new MethodComponentContext(ClusterANNSQEncoder.NAME, new HashMap<>(Map.of(SQ_BITS, bits)));
        resolvedMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderContext);
    }

    private static void validateMode(KNNMethodConfigContext knnMethodConfigContext) {
        if (Mode.isConfigured(knnMethodConfigContext.getMode())) {
            ValidationException e = new ValidationException();
            e.addValidationError(
                String.format(Locale.ROOT, "\"%s\" is not supported for the \"%s\" method", MODE_PARAMETER, METHOD_CLUSTER)
            );
            throw e;
        }
    }

    private static void validateSpaceType(SpaceType spaceType) {
        if (!SUPPORTED_SPACE_TYPES.contains(spaceType)) {
            ValidationException e = new ValidationException();
            e.addValidationError(
                String.format(
                    Locale.ROOT,
                    "%s method does not support space type \"%s\"; supported types are %s",
                    METHOD_CLUSTER,
                    spaceType.getValue(),
                    SUPPORTED_SPACE_TYPES.stream().map(SpaceType::getValue).collect(Collectors.joining(", "))
                )
            );
            throw e;
        }
    }

    private static void validateCompressionSupported(CompressionLevel requested) {
        if (CompressionLevel.isConfigured(requested) && !SUPPORTED_COMPRESSION.contains(requested)) {
            ValidationException e = new ValidationException();
            e.addValidationError(
                String.format(
                    Locale.ROOT,
                    "%s method does not support \"%s\" compression; supported levels are %s",
                    METHOD_CLUSTER,
                    requested.getName(),
                    SUPPORTED_COMPRESSION.stream().map(CompressionLevel::getName).collect(Collectors.joining(", "))
                )
            );
            throw e;
        }
    }
}
