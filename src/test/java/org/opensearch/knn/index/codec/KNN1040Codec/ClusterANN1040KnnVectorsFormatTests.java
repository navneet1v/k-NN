/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.warmup.WarmableReader;

import java.util.Iterator;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;

public class ClusterANN1040KnnVectorsFormatTests extends KNNTestCase {

    public void testFormatName() {
        assertEquals("ClusterANN1040KnnVectorsFormat", new ClusterANN1040KnnVectorsFormat().getName());
    }

    public void testGetMaxDimensions() {
        assertTrue(new ClusterANN1040KnnVectorsFormat().getMaxDimensions("test-field") > 0);
    }

    @SneakyThrows
    public void testFieldsWriter_returnsClusterANNWriter() {
        SegmentWriteState writeState = createMockedWriteState();
        ClusterANN1040KnnVectorsFormat format = new ClusterANN1040KnnVectorsFormat();

        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);

            KnnVectorsWriter writer = format.fieldsWriter(writeState);
            assertTrue(writer instanceof ClusterANN1040KnnVectorsWriter);
            writer.close();
        }
    }

    @SneakyThrows
    public void testFieldsReader_returnsClusterANNReader() {
        SegmentReadState readState = createMockedReadState();
        ClusterANN1040KnnVectorsFormat format = new ClusterANN1040KnnVectorsFormat();

        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class))).thenAnswer((Answer<Void>) invocation -> null);

            KnnVectorsReader reader = format.fieldsReader(readState);
            assertTrue(reader instanceof ClusterANN1040KnnVectorsReader);
            reader.close();
        }
    }

    @SneakyThrows
    public void testFieldsReader_implementsWarmableReader() {
        SegmentReadState readState = createMockedReadState();
        ClusterANN1040KnnVectorsFormat format = new ClusterANN1040KnnVectorsFormat();

        try (MockedStatic<CodecUtil> mockedCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedCodecUtil.when(() -> CodecUtil.retrieveChecksum(any(IndexInput.class))).thenAnswer((Answer<Void>) invocation -> null);

            KnnVectorsReader reader = format.fieldsReader(readState);
            assertTrue(reader instanceof WarmableReader);
            reader.close();
        }
    }

    @SneakyThrows
    private SegmentWriteState createMockedWriteState() {
        Directory directory = mock(Directory.class);
        Mockito.when(directory.createOutput(anyString(), any())).thenReturn(mock(IndexOutput.class));

        SegmentInfo segmentInfo = new SegmentInfo(
            mock(Directory.class),
            mock(Version.class),
            mock(Version.class),
            "test-segment",
            0,
            false,
            false,
            mock(org.apache.lucene.codecs.Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            mock(Sort.class)
        );

        return new SegmentWriteState(mock(InfoStream.class), directory, segmentInfo, mock(FieldInfos.class), null, mock(IOContext.class));
    }

    @SneakyThrows
    private SegmentReadState createMockedReadState() {
        Directory directory = mock(Directory.class);
        IndexInput input = mock(IndexInput.class);
        Mockito.when(directory.openInput(any(), any())).thenReturn(input);

        FieldInfos fieldInfos = mock(FieldInfos.class);
        Mockito.when(fieldInfos.iterator()).thenReturn(new Iterator<FieldInfo>() {
            @Override
            public boolean hasNext() {
                return false;
            }

            @Override
            public FieldInfo next() {
                return null;
            }
        });

        SegmentInfo segmentInfo = new SegmentInfo(
            mock(Directory.class),
            mock(Version.class),
            mock(Version.class),
            "test-segment",
            0,
            false,
            false,
            mock(org.apache.lucene.codecs.Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            mock(Sort.class)
        );

        return new SegmentReadState(directory, segmentInfo, fieldInfos, mock(IOContext.class), "test-segment-suffix");
    }
}
