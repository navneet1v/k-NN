/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1030Codec.KNN1030Codec;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.indices.ModelDao;

import java.util.Optional;
import java.util.Random;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.Version.CURRENT;

/**
 * Drives a {@code "method": {"name": "cluster"}} mapping through the mapper, the per-field codec
 * dispatch, and a real {@link KNN1030Codec} index: the wiring that lets an OpenSearch index use
 * the ClusterANN format end to end (mirrors SearchServicesESKNN adf03dc).
 */
public class ClusterANNVectorFieldMapperTests extends KNNTestCase {

    private static final String FIELD = "cluster_vec";
    private static final int DIM = 16;

    public void testMappingResolvesToClusterANNMapperAndFormat() {
        final ClusterANNVectorFieldMapper mapper = buildMapper(SpaceType.L2, null, 16);

        final KNNMappingConfig config = mapper.fieldType().getKnnMappingConfig();
        assertEquals(CompressionLevel.x16, config.getCompressionLevel());
        assertEquals(KNNEngine.UNDEFINED, config.getKnnMethodContext().orElseThrow().getKnnEngine());
        assertEquals(METHOD_CLUSTER, config.getKnnMethodContext().orElseThrow().getMethodComponentContext().getName());
        final Object encoder = config.getKnnMethodContext()
            .orElseThrow()
            .getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        assertEquals(2, ((MethodComponentContext) encoder).getParameters().get(SQ_BITS));

        // Segment field attributes downstream readers rely on
        assertEquals(METHOD_CLUSTER, mapper.luceneFieldType().getAttributes().get(KNN_METHOD));
        assertEquals(String.valueOf(DIM), mapper.luceneFieldType().getAttributes().get(DIMENSION));
        assertEquals(SpaceType.L2.getValue(), mapper.luceneFieldType().getAttributes().get(SPACE_TYPE));
        assertEquals("2", mapper.luceneFieldType().getAttributes().get(SQ_BITS));
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, mapper.luceneFieldType().vectorSimilarityFunction());

        final KnnVectorsFormat format = perFieldFormat(mapper).getKnnVectorsFormatForField(FIELD);
        assertTrue(format instanceof KNN1030ClusterANNVectorsFormat);
    }

    public void testDefaultCompressionIsX8() {
        final ClusterANNVectorFieldMapper mapper = buildMapper(SpaceType.INNER_PRODUCT, null, null);
        assertEquals(CompressionLevel.x8, mapper.fieldType().getKnnMappingConfig().getCompressionLevel());
        assertEquals("4", mapper.luceneFieldType().getAttributes().get(SQ_BITS));
    }

    public void testEngineRejected() {
        final MapperParsingException e = expectThrows(MapperParsingException.class, () -> buildMapper(SpaceType.L2, "lucene", null));
        assertTrue(e.getMessage(), e.getMessage().contains("engine must not be specified"));
    }

    @SneakyThrows
    public void testIndexesAndSearchesThroughKNN1030Codec() {
        final ClusterANNVectorFieldMapper mapper = buildMapper(SpaceType.L2, null, 32);
        final Codec codec = KNN1030Codec.builder()
            .delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE)
            .knnVectorsFormat(perFieldFormat(mapper))
            .build();

        final int numDocs = 300;
        final Random random = new Random(42);
        final float[][] vectors = new float[numDocs][DIM];
        try (Directory dir = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false))) {
                for (int i = 0; i < numDocs; i++) {
                    for (int d = 0; d < DIM; d++) {
                        vectors[i][d] = random.nextFloat();
                    }
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, vectors[i], mapper.luceneFieldType()));
                    writer.addDocument(doc);
                    if (i % 100 == 99) {
                        writer.flush();
                    }
                }
                writer.forceMerge(1);
            }

            // The ClusterANN files are on disk: the field went through KNN1030ClusterANNVectorsWriter, not the flat fallback.
            final String[] files = dir.listAll();
            for (String ext : new String[] { "clam", "clac", "clap", "clar" }) {
                assertTrue("expected a ." + ext + " file, got " + String.join(",", files), containsExtension(files, ext));
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(numDocs, reader.numDocs());
                assertEquals(1, reader.leaves().size());
                final LeafReaderContext leaf = reader.leaves().get(0);
                final FloatVectorValues values = leaf.reader().getFloatVectorValues(FIELD);
                assertEquals(numDocs, values.size());

                final TopDocs hits = new IndexSearcher(reader).search(new KnnFloatVectorQuery(FIELD, vectors[7], 5), 5);
                assertEquals(5, hits.scoreDocs.length);
            }
        }
    }

    private static boolean containsExtension(String[] files, String ext) {
        for (String f : files) {
            if (f.endsWith("." + ext)) {
                return true;
            }
        }
        return false;
    }

    /** Parses the mapping through the real TypeParser and builds the field mapper, as index creation would. */
    @SneakyThrows
    private ClusterANNVectorFieldMapper buildMapper(SpaceType spaceType, String engine, Integer compression) {
        final XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field("type", "knn_vector")
            .field("dimension", DIM);
        if (compression != null) {
            xContentBuilder.field(COMPRESSION_LEVEL_PARAMETER, compression + "x");
        }
        xContentBuilder.startObject(KNN_METHOD).field(NAME, METHOD_CLUSTER).field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        if (engine != null) {
            xContentBuilder.field(KNN_ENGINE, engine);
        }
        xContentBuilder.endObject().endObject();

        final Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        final KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser(() -> mock(ModelDao.class));
        final KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            FIELD,
            xContentBuilderToMap(xContentBuilder),
            new KNNVectorFieldMapperTests().buildParserContext("test", settings)
        );
        final KNNVectorFieldMapper mapper = builder.build(new Mapper.BuilderContext(settings, new ContentPath()));
        assertTrue(
            "expected ClusterANNVectorFieldMapper, got " + mapper.getClass().getSimpleName(),
            mapper instanceof ClusterANNVectorFieldMapper
        );
        return (ClusterANNVectorFieldMapper) mapper;
    }

    /** The per-field format KNNCodecService installs, over a MapperService that knows this one field. */
    private static PerFieldKnnVectorsFormat perFieldFormat(KNNVectorFieldMapper mapper) {
        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType(FIELD)).thenReturn(mapper.fieldType());
        return new KNN9120PerFieldKnnVectorsFormat(Optional.of(mapperService));
    }
}
