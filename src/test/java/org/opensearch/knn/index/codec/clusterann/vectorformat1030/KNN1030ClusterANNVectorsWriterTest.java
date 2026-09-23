/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Isolated unit tests for {@link KNN1030ClusterANNVectorsWriter} using Mockito for the delegate it forwards to.
 *
 * <p>Most methods just forward to the injected {@link FlatVectorsWriter}, and these tests verify that
 * delegation. {@link KNN1030ClusterANNVectorsWriter#finish()} does more — it writes the segment's metadata file —
 * so the writer is given a real segment over a {@link ByteBuffersDirectory} rather than a null state, and what
 * it wrote is read back the way the reader reads it.
 */
class KNN1030ClusterANNVectorsWriterTest {

    private static final String SEGMENT = "_0";
    private static final int MAX_DOC = 10;

    private final FlatVectorsWriter raw = mock(FlatVectorsWriter.class);
    private final FlatVectorsFormat rawFormat = mock(FlatVectorsFormat.class);

    private Directory directory;
    private SegmentWriteState state;
    private KNN1030ClusterANNVectorsWriter writer;

    @BeforeEach
    void setUp() throws IOException {
        directory = new ByteBuffersDirectory();
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            null,
            SEGMENT,
            MAX_DOC,
            false,
            false,
            null,
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null
        );
        state = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            directory,
            segmentInfo,
            new FieldInfos(new FieldInfo[0]),
            null,
            IOContext.DEFAULT
        );
        when(rawFormat.fieldsWriter(state)).thenReturn(raw);
        writer = new KNN1030ClusterANNVectorsWriter(state, rawFormat, QuantizationParams.DEFAULT);
    }

    @AfterEach
    void tearDown() throws IOException {
        // The writer holds the cluster-file outputs open across its life; close them before the directory.
        IOUtils.closeWhileHandlingException(writer);
        directory.close();
    }

    @Test
    void addField_delegates() throws Exception {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        final FlatFieldVectorsWriter<?> expected = mock(FlatFieldVectorsWriter.class);
        doReturn(expected).when(raw).addField(fieldInfo);
        assertSame(expected, writer.addField(fieldInfo));
    }

    @Test
    void addField_rejectsNonFloat32() {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        assertThrows(UnsupportedOperationException.class, () -> writer.addField(fieldInfo));
    }

    @Test
    void flush_delegates() throws Exception {
        writer.flush(10, null);
        verify(raw).flush(10, null);
    }

    @Test
    void finish_delegates() throws Exception {
        writer.finish();
        verify(raw).finish();
    }

    /**
     * The metadata file must be there and be well formed even with nothing to describe yet, since the reader
     * opens it unconditionally. Reading it the way the reader does — header, block size, terminator, footer —
     * is what proves it: a wrong order or a missing piece fails on the checksum.
     */
    @Test
    void finish_writesAnOpenableMetaFile_whenThereAreNoFieldEntries() throws Exception {
        writer.finish();
        writer.close(); // commit the outputs so the .clam can be read back

        final String metaFileName = IndexFileNames.segmentFileName(
            SEGMENT,
            state.segmentSuffix,
            KNN1030ClusterANNVectorsFormat.META_EXTENSION
        );
        assertTrue(directory.listAll().length > 0, "the writer must have created the metadata file");

        try (ChecksumIndexInput meta = directory.openChecksumInput(metaFileName)) {
            CodecUtil.checkIndexHeader(
                meta,
                KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                KNN1030ClusterANNVectorsFormat.VERSION_START,
                KNN1030ClusterANNVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            assertEquals(32, meta.readVInt());
            assertEquals(KNN1030ClusterANNVectorsFormat.NO_MORE_FIELDS, meta.readInt(), "no field entries yet");
            CodecUtil.checkFooter(meta);
        }
    }

    @Test
    void close_delegates() throws Exception {
        writer.close();
        verify(raw).close();
    }

    @Test
    void ramBytesUsed_delegates() {
        when(raw.ramBytesUsed()).thenReturn(123L);
        assertEquals(123L, writer.ramBytesUsed());
    }
}
