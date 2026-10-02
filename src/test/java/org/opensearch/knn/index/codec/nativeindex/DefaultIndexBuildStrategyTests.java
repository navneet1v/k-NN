/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.store.IndexOutput;
import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapFloatVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class DefaultIndexBuildStrategyTests extends OpenSearchTestCase {

    ArgumentCaptor<float[]> vectorTransferCapture = ArgumentCaptor.forClass(float[].class);

    @Before
    public void init() {
        vectorTransferCapture = ArgumentCaptor.forClass(float[].class);
    }

    @SneakyThrows
    public void testBuildAndWrite() {
        // Given
        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });

        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        try (
            MockedStatic<JNIService> mockedJNIService = mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = mockStatic(OffHeapVectorTransferFactory.class)
        ) {
            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.NMSLIB)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            DefaultIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then
            mockedJNIService.verify(
                () -> JNIService.createIndex(
                    eq(new int[] { 0, 1, 2 }),
                    eq(200L),
                    eq(knnVectorValues.dimension()),
                    eq(indexOutputWithBuffer),
                    eq(Map.of("index", "param")),
                    eq(KNNEngine.NMSLIB)
                )
            );
            mockedJNIService.verifyNoMoreInteractions();
            verify(offHeapVectorTransfer).flush(true);
            verify(offHeapVectorTransfer, times(3)).transfer(vectorTransferCapture.capture(), eq(true));
            verify(offHeapVectorTransfer).reset();

            float[] prev = null;
            for (float[] vector : vectorTransferCapture.getAllValues()) {
                if (prev != null) {
                    assertNotSame(prev, vector);
                }
                prev = vector;
            }
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_withQuantization() {
        // Given
        ArgumentCaptor<Long> vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);
        ArgumentCaptor<Object> vectorTransferCapture = ArgumentCaptor.forClass(Object.class);

        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        try (
            MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class);
            MockedStatic<JNIService> mockedJNIService = mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = mockStatic(OffHeapVectorTransferFactory.class);
            MockedStatic<QuantizationService> mockedQuantizationIntegration = mockStatic(QuantizationService.class)
        ) {

            // Limits transfer to 2 vectors
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            QuantizationService quantizationService = mock(QuantizationService.class);
            mockedQuantizationIntegration.when(QuantizationService::getInstance).thenReturn(quantizationService);

            QuantizationState quantizationState = mock(QuantizationState.class);
            ArgumentCaptor<float[]> vectorCaptor = ArgumentCaptor.forClass(float[].class);
            // New: Create QuantizationOutput and mock the quantization process
            QuantizationOutput<byte[]> quantizationOutput = mock(QuantizationOutput.class);
            when(quantizationOutput.getQuantizedVectorCopy()).thenReturn(new byte[] { 1, 2 });
            when(quantizationService.createQuantizationOutput(eq(quantizationState.getQuantizationParams()))).thenReturn(
                quantizationOutput
            );

            // Quantize the vector with the quantization output
            when(quantizationService.quantize(eq(quantizationState), vectorCaptor.capture(), eq(quantizationOutput))).thenAnswer(
                invocation -> {
                    quantizationOutput.getQuantizedVectorCopy();
                    return quantizationOutput.getQuantizedVectorCopy();
                }
            );
            when(quantizationState.getDimensions()).thenReturn(2);
            when(quantizationState.getBytesPerVector()).thenReturn(8);

            when(offHeapVectorTransfer.transfer(vectorTransferCapture.capture(), eq(false))).thenReturn(false)
                .thenReturn(true)
                .thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(true);
            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .quantizationState(quantizationState)
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then
            mockedJNIService.verify(
                () -> JNIService.initIndex(
                    knnVectorValues.totalLiveDocs(),
                    knnVectorValues.dimension(),
                    Map.of("index", "param"),
                    KNNEngine.FAISS
                )
            );

            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 0, 1 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            // For the flush
            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 2 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            mockedJNIService.verify(
                () -> JNIService.writeIndex(
                    eq(indexOutputWithBuffer),
                    eq(100L),
                    eq(KNNEngine.FAISS),
                    eq(Map.of("index", "param")),
                    eq(false)
                )
            );
            assertEquals(200L, vectorAddressCaptor.getValue().longValue());
            assertEquals(vectorAddressCaptor.getValue().longValue(), vectorAddressCaptor.getAllValues().get(0).longValue());
            verify(offHeapVectorTransfer, times(0)).reset();

            for (Object vector : vectorTransferCapture.getAllValues()) {
                // Assert that the vector is in byte[] format due to quantization
                assertTrue(vector instanceof byte[]);
            }
        }
    }

    @SneakyThrows
    public void testBuildAndWriteWithModel() {
        // Given
        final Map<Integer, float[]> docs = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 }, 2, new float[] { 3, 4 });
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docs.keySet().stream().sorted().forEach(docsWithFieldSet::add);

        byte[] modelBlob = new byte[] { 1 };

        KNNFloatVectorValues knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            docs
        );
        try (
            MockedStatic<JNIService> mockedJNIService = mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = mockStatic(OffHeapVectorTransferFactory.class)
        ) {

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.NMSLIB)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("model_id", "id", "model_blob", modelBlob))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            DefaultIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then
            mockedJNIService.verify(
                () -> JNIService.createIndexFromTemplate(
                    eq(new int[] { 0, 1, 2 }),
                    eq(200L),
                    eq(2),
                    eq(indexOutputWithBuffer),
                    eq(modelBlob),
                    eq(Map.of("model_id", "id", "model_blob", modelBlob)),
                    eq(KNNEngine.NMSLIB)
                )
            );
            mockedJNIService.verifyNoMoreInteractions();
            verify(offHeapVectorTransfer).flush(true);
            verify(offHeapVectorTransfer, times(3)).transfer(vectorTransferCapture.capture(), eq(true));

            float[] prev = null;
            for (float[] vector : vectorTransferCapture.getAllValues()) {
                if (prev != null) {
                    assertNotSame(prev, vector);
                }
                prev = vector;
            }
        }
    }

    /**
     * Repro for the NMSLIB double free seen as a SIGSEGV in {@code JNICommons.freeVectorData} during merges.
     *
     * <p>NMSLIB's native {@code CreateIndex} takes ownership of the off-heap vectors and runs {@code delete inputVectors}
     * right after copying them, before the graph is built and serialized. If serialization then fails (e.g. the merge
     * {@link IndexOutput} throws because the merge was aborted), the exception reaches Java before
     * {@code vectorTransfer.reset()} runs, so try-with-resources {@code close()} calls {@code deallocate()} on the
     * already-freed address.</p>
     *
     * <p>This test runs the real native NMSLIB build with an {@link IndexOutput} that fails on write and stubs out
     * {@code deallocate()} so the JVM does not crash. On the buggy code it fails because {@code deallocate()} is invoked
     * after native has already freed the vectors.</p>
     */
    @SneakyThrows
    public void testBuildAndWrite_nmslib_whenWriteFailsAfterNativeTookOwnership_thenVectorsAreNotFreedTwice() {
        final KNNVectorValues<?> knnVectorValues = randomFloatVectorValues(100, 8);

        try (
            MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class, CALLS_REAL_METHODS);
            MockedStatic<OffHeapVectorTransferFactory> mockedFactory = mockStatic(OffHeapVectorTransferFactory.class)
        ) {
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(1024 * 1024));
            final OffHeapFloatVectorTransfer vectorTransfer = spy(new OffHeapFloatVectorTransfer(8 * Float.BYTES, 100));
            // Prevent the real double free so the JVM survives; we only want to observe that it would have happened.
            doNothing().when(vectorTransfer).deallocate();
            mockedFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(any(), anyInt(), anyInt())).thenReturn(vectorTransfer);

            expectThrows(
                RuntimeException.class,
                () -> DefaultIndexBuildStrategy.getInstance().buildAndWriteIndex(nmslibParams(knnVectorValues))
            );

            // Native already deleted the vectors inside CreateIndex, so Java must not free them again.
            verify(vectorTransfer, never()).deallocate();
        }
    }

    /**
     * Same scenario as above but lets {@code deallocate()} run for real. On the buggy code this double frees the native
     * {@code std::vector<float>} and crashes the test JVM (SIGSEGV with jemalloc, SIGABRT "pointer being freed was not
     * allocated" with macOS libmalloc), reproducing the production crash. Run it on its own.
     */
    @SneakyThrows
    public void testBuildAndWrite_nmslib_whenWriteFailsAfterNativeTookOwnership_thenNoNativeCrash() {
        final KNNVectorValues<?> knnVectorValues = randomFloatVectorValues(100, 8);

        try (MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class, CALLS_REAL_METHODS)) {
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(1024 * 1024));
            expectThrows(
                RuntimeException.class,
                () -> DefaultIndexBuildStrategy.getInstance().buildAndWriteIndex(nmslibParams(knnVectorValues))
            );
        }
    }

    private static KNNVectorValues<?> randomFloatVectorValues(final int count, final int dimension) {
        final List<float[]> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            final float[] vector = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = randomFloat();
            }
            vectors.add(vector);
        }
        return KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, new TestVectorValues.PreDefinedFloatVectorValues(vectors));
    }

    private static BuildIndexParams nmslibParams(final KNNVectorValues<?> knnVectorValues) throws IOException {
        // Simulates the merge IndexOutput failing (e.g. MergeAbortedException) while NMSLIB serializes the graph.
        final IndexOutput failingIndexOutput = mock(IndexOutput.class);
        doThrow(new IOException("simulated merge abort while writing NMSLIB index")).when(failingIndexOutput)
            .writeBytes(any(byte[].class), anyInt(), anyInt());
        return BuildIndexParams.builder()
            .indexOutputWithBuffer(new IndexOutputWithBuffer(failingIndexOutput))
            .knnEngine(KNNEngine.NMSLIB)
            .vectorDataType(VectorDataType.FLOAT)
            .indexParameters(Map.of(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue()))
            .knnVectorValuesSupplier(() -> knnVectorValues)
            .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
            .build();
    }
}
