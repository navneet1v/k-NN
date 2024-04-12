/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import com.google.common.collect.ImmutableMap;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.StopWatch;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.codecs.CodecUtil.FOOTER_MAGIC;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;

/**
 * This class writes the KNN docvalues to the segments
 */
@Log4j2
class KNN80DocValuesConsumer extends DocValuesConsumer implements Closeable {

    private final Logger logger = LogManager.getLogger(KNN80DocValuesConsumer.class);

    private final DocValuesConsumer delegatee;
    private final SegmentWriteState state;

    private static final Long CRC32_CHECKSUM_SANITY = 0xFFFFFFFF00000000L;

    KNN80DocValuesConsumer(DocValuesConsumer delegatee, SegmentWriteState state) {
        this.delegatee = delegatee;
        this.state = state;
    }

    @Override
    public void addBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addBinaryField(field, valuesProducer);
        if (isKNNBinaryFieldRequired(field)) {
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            addKNNBinaryField(field, valuesProducer, false, true);
            stopWatch.stop();
            long time_in_millis = stopWatch.totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.set(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue() + time_in_millis);
            logger.warn("Refresh operation complete in " + time_in_millis + " ms");
        }
    }

    private boolean isKNNBinaryFieldRequired(FieldInfo field) {
        final KNNEngine knnEngine = getKNNEngine(field);
        log.debug(String.format("Read engine [%s] for field [%s]", knnEngine.getName(), field.getName()));
        return field.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)
            && KNNEngine.getEnginesThatCreateCustomSegmentFiles().stream().anyMatch(engine -> engine == knnEngine);
    }

    private KNNEngine getKNNEngine(@NonNull FieldInfo field) {
        final String modelId = field.attributes().get(MODEL_ID);
        if (modelId != null) {
            var model = ModelCache.getInstance().get(modelId);
            return model.getModelMetadata().getKnnEngine();
        }
        final String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
        return KNNEngine.getEngine(engineName);
    }

    public void addKNNBinaryField(FieldInfo field, DocValuesProducer valuesProducer, boolean isMerge, boolean isRefresh)
        throws IOException {
        // Get values to be indexed
        BinaryDocValues values = valuesProducer.getBinary(field);
        if (values == null) {
            log.info("BinaryDocValues is null. Returning..");
            return;
        }
        // Get the KNN engine
        final KNNEngine knnEngine = getKNNEngine(field);
        final String engineFileName = buildEngineFileName(
            state.segmentInfo.name,
            knnEngine.getVersion(),
            field.name,
            knnEngine.getExtension()
        );
        final String indexPath = Paths.get(
            ((FSDirectory) (FilterDirectory.unwrap(state.directory))).getDirectory().toString(),
            engineFileName
        ).toString();
        Map<String, Object> parametersMap = getKNNIndexFromScratchParameters(field, knnEngine);

        long indexAddress = createIndex(values, knnEngine, parametersMap);

        if (indexAddress == 0) {
            log.info("Index is not created. Returning..");
        }

        state.directory.createOutput(engineFileName, state.context).close();
        // Now we can write the index
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.writeIndex(indexAddress, parametersMap, indexPath, knnEngine);
            return null;
        });
        writeFooter(indexPath, engineFileName);
    }

    private long createIndex(BinaryDocValues values, final KNNEngine knnEngine, final Map<String, Object> parametersMap)
        throws IOException {
        List<float[]> vectorList = new ArrayList<>();
        List<Integer> docIdList = new ArrayList<>();
        int dimension = 0;
        SerializationMode serializationMode = SerializationMode.COLLECTION_OF_FLOATS;

        long totalLiveDocs = KNNCodecUtil.getTotalLiveDocsCount(values);
        long vectorsStreamingMemoryLimit = KNNSettings.getVectorStreamingMemoryLimit().getBytes();
        long vectorsPerTransfer = Integer.MIN_VALUE;
        Long indexAddress = 0L;

        for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            BytesRef bytesref = values.binaryValue();
            try (ByteArrayInputStream byteStream = new ByteArrayInputStream(bytesref.bytes, bytesref.offset, bytesref.length)) {
                final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                final float[] vector = vectorSerializer.byteToFloatArray(byteStream);
                dimension = vector.length;

                if (vectorsPerTransfer == Integer.MIN_VALUE) {
                    vectorsPerTransfer = (dimension * Float.BYTES * totalLiveDocs) / vectorsStreamingMemoryLimit;
                    if (vectorsPerTransfer == 0) {
                        vectorsPerTransfer = totalLiveDocs;
                    }
                }
                if (vectorList.size() == vectorsPerTransfer) {
                    final long vectorAddress = JNICommons.storeVectorData(
                        0,
                        vectorList.toArray(new float[][] {}),
                        (long) vectorList.size() * dimension
                    );
                    List<Integer> docIdList2 = docIdList;
                    int finalDimension = dimension;
                    long indexAddress2 = indexAddress;
                    indexAddress = AccessController.doPrivileged(
                        (PrivilegedAction<Long>) () -> JNIService.buildIndex(
                            docIdList2.stream().mapToInt(Integer::intValue).toArray(),
                            vectorAddress,
                            indexAddress2,
                            finalDimension,
                            parametersMap,
                            knnEngine
                        )
                    );

                    // We should probably come up with a better way to reuse the vectorList memory which we have
                    // created. Problem here is doing like this can lead to a lot of list memory which is of no use and
                    // will be garbage collected later on, but it creates pressure on JVM. We should revisit this.
                    vectorList = new ArrayList<>();
                    docIdList = new ArrayList<>();
                    // JNICommons.freeVectorData(vectorAddress);
                }
                vectorList.add(vector);
            }
            docIdList.add(doc);
        }
        if (vectorList.isEmpty() == false) {
            long vectorAddress = JNICommons.storeVectorData(0, vectorList.toArray(new float[][] {}), (long) vectorList.size() * dimension);
            List<Integer> docIdList2 = docIdList;
            int finalDimension = dimension;
            long indexAddress2 = indexAddress;
            indexAddress = AccessController.doPrivileged(
                (PrivilegedAction<Long>) () -> JNIService.buildIndex(
                    docIdList2.stream().mapToInt(Integer::intValue).toArray(),
                    vectorAddress,
                    indexAddress2,
                    finalDimension,
                    parametersMap,
                    knnEngine
                )
            );
            // JNICommons.freeVectorData(vectorAddress);
        }
        return indexAddress;
    }

    private void recordMergeStats(int length, long arraySize) {
        KNNGraphValue.MERGE_CURRENT_OPERATIONS.decrement();
        KNNGraphValue.MERGE_CURRENT_DOCS.decrementBy(length);
        KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.decrementBy(arraySize);
        KNNGraphValue.MERGE_TOTAL_OPERATIONS.increment();
        KNNGraphValue.MERGE_TOTAL_DOCS.incrementBy(length);
        KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.incrementBy(arraySize);
    }

    private void recordRefreshStats() {
        KNNGraphValue.REFRESH_TOTAL_OPERATIONS.increment();
    }

    private void createKNNIndexFromTemplate(byte[] model, KNNCodecUtil.Pair pair, KNNEngine knnEngine, String indexPath) {
        Map<String, Object> parameters = ImmutableMap.of(
            KNNConstants.INDEX_THREAD_QTY,
            KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY)
        );
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.createIndexFromTemplate(
                pair.docs,
                pair.getVectorAddress(),
                pair.getDimension(),
                indexPath,
                model,
                parameters,
                knnEngine
            );
            return null;
        });
    }

    private Map<String, Object> getKNNIndexFromScratchParameters(FieldInfo fieldInfo, KNNEngine knnEngine) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

        // parametersString will be null when legacy mapper is used
        if (parametersString == null) {
            parameters.put(KNNConstants.SPACE_TYPE, fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue()));

            String efConstruction = fieldAttributes.get(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION);
            Map<String, Object> algoParams = new HashMap<>();
            if (efConstruction != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, Integer.parseInt(efConstruction));
            }

            String m = fieldAttributes.get(KNNConstants.HNSW_ALGO_M);
            if (m != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_M, Integer.parseInt(m));
            }
            parameters.put(PARAMETERS, algoParams);
        } else {
            parameters.putAll(
                XContentType.JSON.xContent()
                    .createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, parametersString)
                    .map()
            );
        }

        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));

        return parameters;
    }

    /**
     * Merges in the fields from the readers in mergeState
     *
     * @param mergeState Holds common state used during segment merging
     */
    @Override
    public void merge(MergeState mergeState) {
        try {
            delegatee.merge(mergeState);
            assert mergeState != null;
            assert mergeState.mergeFieldInfos != null;
            for (FieldInfo fieldInfo : mergeState.mergeFieldInfos) {
                DocValuesType type = fieldInfo.getDocValuesType();
                if (type == DocValuesType.BINARY && fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)) {
                    StopWatch stopWatch = new StopWatch();
                    stopWatch.start();
                    addKNNBinaryField(fieldInfo, new KNN80DocValuesReader(mergeState), true, false);
                    stopWatch.stop();
                    long time_in_millis = stopWatch.totalTime().millis();
                    KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.set(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue() + time_in_millis);
                    logger.warn("Merge operation complete in " + time_in_millis + " ms");
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addSortedSetField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedSetField(field, valuesProducer);
    }

    @Override
    public void addSortedNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedNumericField(field, valuesProducer);
    }

    @Override
    public void addSortedField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addSortedField(field, valuesProducer);
    }

    @Override
    public void addNumericField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
        delegatee.addNumericField(field, valuesProducer);
    }

    @Override
    public void close() throws IOException {
        delegatee.close();
    }

    @FunctionalInterface
    private interface NativeIndexCreator {
        void createIndex() throws IOException;
    }

    private void writeFooter(String indexPath, String engineFileName) throws IOException {
        // Opens the engine file that was created and appends a footer to it. The footer consists of
        // 1. A Footer magic number (int - 4 bytes)
        // 2. A checksum algorithm id (int - 4 bytes)
        // 3. A checksum (long - bytes)
        // The checksum is computed on all the bytes written to the file up to that point.
        // Logic where footer is written in Lucene can be found here:
        // https://github.com/apache/lucene/blob/branch_9_0/lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java#L390-L412
        OutputStream os = Files.newOutputStream(Paths.get(indexPath), StandardOpenOption.APPEND);
        ByteBuffer byteBuffer = ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.putInt(FOOTER_MAGIC);
        byteBuffer.putInt(0);
        os.write(byteBuffer.array());
        os.flush();

        ChecksumIndexInput checksumIndexInput = state.directory.openChecksumInput(engineFileName, state.context);
        checksumIndexInput.seek(checksumIndexInput.length());
        long value = checksumIndexInput.getChecksum();
        checksumIndexInput.close();

        if (isChecksumValid(value)) {
            throw new IllegalStateException("Illegal CRC-32 checksum: " + value + " (resource=" + os + ")");
        }

        // Write the CRC checksum to the end of the OutputStream and close the stream
        byteBuffer.putLong(0, value);
        os.write(byteBuffer.array());
        os.close();
    }

    private boolean isChecksumValid(long value) {
        // Check pulled from
        // https://github.com/apache/lucene/blob/branch_9_0/lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java#L644-L647
        return (value & CRC32_CHECKSUM_SANITY) != 0;
    }
}
