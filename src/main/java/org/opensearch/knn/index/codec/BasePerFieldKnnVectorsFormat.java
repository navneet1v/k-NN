/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.Version;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.KNN990FlatVectorsFormat;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Map;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Base class for PerFieldKnnVectorsFormat, builds KnnVectorsFormat based on specific Lucene version
 */
@AllArgsConstructor
@Log4j2
public abstract class BasePerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> optionalMapperService;
    private final int defaultMaxConnections;
    private final int defaultBeamWidth;
    private final Supplier<KnnVectorsFormat> defaultFormatSupplier;
    private final Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier;
    private Function<KNNScalarQuantizedVectorsFormatParams, KnnVectorsFormat> scalarQuantizedVectorsFormatSupplier;
    private static final String MAX_CONNECTIONS = "max_connections";
    private static final String BEAM_WIDTH = "beam_width";

    public BasePerFieldKnnVectorsFormat(
        Optional<MapperService> mapperService,
        int defaultMaxConnections,
        int defaultBeamWidth,
        Supplier<KnnVectorsFormat> defaultFormatSupplier,
        Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier
    ) {
        this.optionalMapperService = mapperService;
        this.defaultMaxConnections = defaultMaxConnections;
        this.defaultBeamWidth = defaultBeamWidth;
        this.defaultFormatSupplier = defaultFormatSupplier;
        this.vectorsFormatSupplier = vectorsFormatSupplier;
    }

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(final String field) {
        if (isKnnVectorFieldType(field) == false) {
            log.debug(
                "Initialize KNN vector format for field [{}] with default params [{}] = \"{}\" and [{}] = \"{}\"",
                field,
                MAX_CONNECTIONS,
                defaultMaxConnections,
                BEAM_WIDTH,
                defaultBeamWidth
            );
            return defaultFormatSupplier.get();
        }

        if (optionalMapperService.isEmpty()) {
            throw new IllegalStateException(
                String.format(Locale.ROOT, "Cannot read field type for field [%s] because mapper service is not available", field)
            );
        }
        final MapperService mapperService = optionalMapperService.get();
        final KNNVectorFieldType mappedFieldType = (KNNVectorFieldType) mapperService.fieldType(field);
        // Version check is added with isSearchable because earlier for every Vector Field we were setting index:false
        // in mappings. So we don't want to break those cases.
        if (mappedFieldType.isSearchable() == false
            && mapperService.getIndexSettings().getIndexVersionCreated().onOrAfter(Version.V_2_17_0)) {
            return new KNN990FlatVectorsFormat();
        }

        KNNMappingConfig knnMappingConfig = mappedFieldType.getKnnMappingConfig();
        KNNMethodContext knnMethodContext = knnMappingConfig.getKnnMethodContext()
            .orElseThrow(() -> new IllegalArgumentException("KNN method context cannot be empty"));
        final KNNEngine engine = knnMethodContext.getKnnEngine();
        final Map<String, Object> params = knnMethodContext.getMethodComponentContext().getParameters();

        if (engine == KNNEngine.LUCENE) {
            if (params != null && params.containsKey(METHOD_ENCODER_PARAMETER)) {
                KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams(
                    params,
                    defaultMaxConnections,
                    defaultBeamWidth
                );
                if (knnScalarQuantizedVectorsFormatParams.validate(params)) {
                    log.debug(
                        "Initialize KNN vector format for field [{}] with params [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\"",
                        field,
                        MAX_CONNECTIONS,
                        knnScalarQuantizedVectorsFormatParams.getMaxConnections(),
                        BEAM_WIDTH,
                        knnScalarQuantizedVectorsFormatParams.getBeamWidth(),
                        LUCENE_SQ_CONFIDENCE_INTERVAL,
                        knnScalarQuantizedVectorsFormatParams.getConfidenceInterval(),
                        LUCENE_SQ_BITS,
                        knnScalarQuantizedVectorsFormatParams.getBits()
                    );
                    return scalarQuantizedVectorsFormatSupplier.apply(knnScalarQuantizedVectorsFormatParams);
                }
            }

            KNNVectorsFormatParams knnVectorsFormatParams = new KNNVectorsFormatParams(params, defaultMaxConnections, defaultBeamWidth);
            log.debug(
                "Initialize KNN vector format for field [{}] with params [{}] = \"{}\" and [{}] = \"{}\"",
                field,
                MAX_CONNECTIONS,
                knnVectorsFormatParams.getMaxConnections(),
                BEAM_WIDTH,
                knnVectorsFormatParams.getBeamWidth()
            );
            return vectorsFormatSupplier.apply(knnVectorsFormatParams);
        }

        // All native engines to use NativeEngines990KnnVectorsFormat
        return new NativeEngines990KnnVectorsFormat(new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
    }

    private boolean isKnnVectorFieldType(final String field) {
        return optionalMapperService.isPresent() && optionalMapperService.get().fieldType(field) instanceof KNNVectorFieldType;
    }
}
