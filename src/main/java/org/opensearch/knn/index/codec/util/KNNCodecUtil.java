/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80BinaryDocValues;
import org.opensearch.knn.jni.JNICommons;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Log4j2
public class KNNCodecUtil {
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;
    // References to objects are 4 bytes in size
    public static final int JAVA_REFERENCE_SIZE = 4;
    // Each array in Java has a header that is 12 bytes
    public static final int JAVA_ARRAY_HEADER_SIZE = 12;
    // Java rounds each array size up to multiples of 8 bytes
    public static final int JAVA_ROUNDING_NUMBER = 8;

    @AllArgsConstructor
    public static final class Pair {
        public int[] docs;
        @Getter
        @Setter
        private long vectorAddress;
        @Getter
        @Setter
        private int dimension;
        public SerializationMode serializationMode;

    }

    public static KNNCodecUtil.Pair getFloats(BinaryDocValues values) throws IOException {
        List<float[]> vectorList = new ArrayList<>();
        List<Integer> docIdList = new ArrayList<>();
        long vectorAddress = 0;
        int dimension = 0;
        SerializationMode serializationMode = SerializationMode.COLLECTION_OF_FLOATS;

        long totalLiveDocs = getTotalLiveDocsCount(values);
        long vectorsStreamingMemoryLimit = KNNSettings.getVectorStreamingMemoryLimit().getBytes();
        long vectorsPerTransfer = Integer.MIN_VALUE;

        for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            BytesRef bytesref = values.binaryValue();
            try (ByteArrayInputStream byteStream = new ByteArrayInputStream(bytesref.bytes, bytesref.offset, bytesref.length)) {
                final float[] vector;
                StopWatch stopWatch = new StopWatch();
                stopWatch.start();
                if (isVectorRepresentedAsString(bytesref)) {
                    String vectorString = new String(bytesref.bytes, 1, bytesref.bytes.length - 1);
                    String[] array = vectorString.split(",");
                    vector = new float[array.length];
                    for (int i = 0; i < array.length; i++) {
                        vector[i] = Float.parseFloat(array[i]);
                    }
                    stopWatch.stop();
                    log.info("Time taken to deserialize vector with string is : {} ms", stopWatch.totalTime().millis());
                } else {
                    serializationMode = KNNVectorSerializerFactory.serializerModeFromStream(byteStream);
                    final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                    vector = vectorSerializer.byteToFloatArray(byteStream);
                    stopWatch.stop();
                    log.info("Time taken to deserialize vector with float array is : {} ms", stopWatch.totalTime().millis());
                }
                dimension = vector.length;

                if (vectorsPerTransfer == Integer.MIN_VALUE) {
                    vectorsPerTransfer = (dimension * Float.BYTES * totalLiveDocs) / vectorsStreamingMemoryLimit;
                    // This condition comes if vectorsStreamingMemoryLimit is higher than total number floats to transfer
                    // Doing this will reduce 1 extra trip to JNI layer.
                    if (vectorsPerTransfer == 0) {
                        vectorsPerTransfer = totalLiveDocs;
                    }
                }
                vectorList.add(vector);
            }
            docIdList.add(doc);
        }
        if (vectorList.isEmpty() == false) {
            vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
        }
        return new KNNCodecUtil.Pair(docIdList.stream().mapToInt(Integer::intValue).toArray(), vectorAddress, dimension, serializationMode);
    }

    public static long calculateArraySize(int numVectors, int vectorLength, SerializationMode serializationMode) {
        if (serializationMode == SerializationMode.ARRAY) {
            int vectorSize = vectorLength * FLOAT_BYTE_SIZE + JAVA_ARRAY_HEADER_SIZE;
            if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorSize += vectorSize % JAVA_ROUNDING_NUMBER;
            }
            int vectorsSize = numVectors * (vectorSize + JAVA_REFERENCE_SIZE) + JAVA_ARRAY_HEADER_SIZE;
            if (vectorsSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorsSize += vectorsSize % JAVA_ROUNDING_NUMBER;
            }
            return vectorsSize;
        } else {
            int vectorSize = vectorLength * FLOAT_BYTE_SIZE;
            if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorSize += vectorSize % JAVA_ROUNDING_NUMBER;
            }
            int vectorsSize = numVectors * (vectorSize + JAVA_REFERENCE_SIZE);
            if (vectorsSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorsSize += vectorsSize % JAVA_ROUNDING_NUMBER;
            }
            return vectorsSize;
        }
    }

    public static String buildEngineFileName(String segmentName, String latestBuildVersion, String fieldName, String extension) {
        return String.format("%s%s%s", buildEngineFilePrefix(segmentName), latestBuildVersion, buildEngineFileSuffix(fieldName, extension));
    }

    public static String buildEngineFilePrefix(String segmentName) {
        return String.format("%s_", segmentName);
    }

    public static String buildEngineFileSuffix(String fieldName, String extension) {
        return String.format("_%s%s", fieldName, extension);
    }

    private static long getTotalLiveDocsCount(final BinaryDocValues binaryDocValues) {
        long totalLiveDocs;
        if (binaryDocValues instanceof KNN80BinaryDocValues) {
            totalLiveDocs = ((KNN80BinaryDocValues) binaryDocValues).getTotalLiveDocs();
        } else {
            totalLiveDocs = binaryDocValues.cost();
        }
        return totalLiveDocs;
    }

    private static boolean isVectorRepresentedAsString(BytesRef bytesref) {
        // Check if first bye is the special character that we have added or not.
        return "$".equals(new String(bytesref.bytes, 0, 1));
    }
}
