/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.FloatArrayList;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.packed.DirectWriter;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040LocalityAwareSQVectorsFormat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.QUANTIZED_VECTOR_COMPONENT;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;

/**
 * Writes 1-bit scalar-quantized vectors to a locality-ordered store split across two files: a
 * metadata file ({@value #METADATA_EXTENSION}) and a records file ({@value #EXTENSION}), in an
 * <b>arbitrary ordinal ordering</b>.
 *
 * <p>This is the foundational write path for locality-aware layout (Milestone 0). The caller
 * supplies an {@code order} array where {@code order[physicalPos] = originalOrdinal}; records are
 * emitted in that physical order. For the initial milestone the ordering is a random shuffle
 * (a test vehicle); later it will be the greedy page-assignment permutation.
 *
 * <p>An {@code originalOrdinal -> physicalPosition} map ({@code ordToPhysicalOrdMap}) is stored in
 * the metadata file so the reader can translate an original (insertion-order) ordinal to the physical
 * position holding its record. It is the inverse of the physical-layout permutation
 * {@code physicalOrdinals} (where {@code physicalOrdinals[p]} is the original ordinal of the record
 * at physical position {@code p}), computed and written here.
 *
 * <p>docId is <b>not</b> stored here. It stays the join key for anything cross-file (e.g.
 * full-precision rescoring reads {@code .vec}), but the reader obtains it directly from the raw flat
 * vectors reader by original ordinal ({@code ordToDoc(originalOrdinal)}) — no docId table is
 * duplicated in this file.
 *
 * <h2>File layout</h2>
 * Metadata file ({@value #METADATA_EXTENSION}):
 * <pre>
 * [CodecUtil index header]
 *    fieldNumber           (int)
 *    dimension             (vInt)
 *    count                 (vInt)
 *    codeLength            (vInt)  packed 1-bit code bytes per record
 *    recordSize            (vInt)  codeLength + CORRECTION_BYTES (= codeLength + 16)
 *    encodingWireNumber    (vInt)  ScalarEncoding.getWireNumber() (similarity is derived from FieldInfo)
 *    centroid              (dimension x f32, LE int bits)
 *    centroidDp            (int, f32 bits)  dot(centroid, centroid)
 *    requiredBits          (vInt)  bits/value of the DirectWriter block that follows
 *    ordToPhysicalOrdMap   (DirectWriter-packed: count values x requiredBits bits, then padding)
 *                                  originalOrdinal -&gt; physicalPosition
 * [CodecUtil footer + CRC32]
 * </pre>
 * Records file ({@value #EXTENSION}):
 * <pre>
 * [CodecUtil index header]
 * [records (permuted): for physicalPos p in 0..count-1 -&gt; record of physicalOrdinals[p]]
 *    record = [ packed 1-bit code (codeLength B) ]
 *             [ lowerInterval (4B f32 LE) ]
 *             [ upperInterval (4B f32 LE) ]
 *             [ additionalCorrection (4B f32 LE) ]
 *             [ quantizedComponentSum (4B i32 LE) ]
 * [CodecUtil footer + CRC32]
 * </pre>
 *
 * <p><b>The {@code ordToPhysicalOrdMap} block.</b> It is packed with {@link DirectWriter} (fixed
 * {@code requiredBits} bits/value — near-optimal for a permutation, which is not monotonic so
 * {@code DirectMonotonic} does not apply). {@code DirectWriter.finish()} pads to a block boundary and
 * {@link org.apache.lucene.util.packed.DirectReader} may over-read into that padding, so the reader
 * needs the exact block length — which it recomputes from the two values it already has:
 * {@code DirectWriter.bytesRequired(count, requiredBits)} (padding included). No byte length is stored
 * on disk, and the block does not have to be the last section (a following section would start at
 * {@code packedStart + bytesRequired(count, requiredBits)}).
 *
 * <p>Records live in {@value #EXTENSION} immediately after its index header
 * ({@code CodecUtil.indexHeaderLength}); the record region is
 * {@code [indexHeaderLength, fileLength - footerLength)}, so metadata sizes never shift it.
 *
 * <p>All ints/floats use Lucene's little-endian encoding, matching the byte order the native SIMD
 * scoring path expects.
 */
public final class LocalityOrderedQuantizedVectorsWriter extends FlatVectorsWriter {

    public static final String CODEC_NAME = "LocalityOrderedQuantizedVectors";
    public static final String EXTENSION = "veqlo";
    public static final String METADATA_EXTENSION = "vemlo";
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;

    /** 3 correction floats + 1 correction int, all 4 bytes. */
    static final int CORRECTION_BYTES = Integer.BYTES * 4;

    private final IndexOutput vectordata;
    private final IndexOutput metadata;
    private boolean wroteField;
    private boolean finished;
    private final QuantizedByteVectorValues.ScalarEncoding scalarEncoding;
    private final FlatVectorsWriter rawFlatVectorsWriter;
    private final List<SQFieldWriter> fields = new ArrayList<>();
    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsFormat flatVectorsFormat;
    private FlatVectorsReader flatVectorsReader;
    private boolean flatWriterClosed;


    public LocalityOrderedQuantizedVectorsWriter(final SegmentWriteState segmentWriteState, FlatVectorsFormat flatVectorsFormat, QuantizedByteVectorValues.ScalarEncoding scalarEncoding, FlatVectorsWriter rawFlatVectorsWriter, Lucene104ScalarQuantizedVectorScorer vectorsScorer) throws IOException {
        super(vectorsScorer);
        this.scalarEncoding = scalarEncoding;
        this.rawFlatVectorsWriter = rawFlatVectorsWriter;
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsFormat = flatVectorsFormat;

        final String vectorFileName = IndexFileNames.segmentFileName(segmentWriteState.segmentInfo.name, segmentWriteState.segmentSuffix, EXTENSION);
        final String metaFileName = IndexFileNames.segmentFileName(segmentWriteState.segmentInfo.name, segmentWriteState.segmentSuffix, METADATA_EXTENSION);
        boolean success = false;
        IndexOutput vectordata = null;
        IndexOutput metadata = null;
        try {
            vectordata = segmentWriteState.directory.createOutput(vectorFileName, segmentWriteState.context);
            metadata = segmentWriteState.directory.createOutput(metaFileName, segmentWriteState.context);
            CodecUtil.writeIndexHeader(vectordata, CODEC_NAME, VERSION_CURRENT, segmentWriteState.segmentInfo.getId(), segmentWriteState.segmentSuffix);
            CodecUtil.writeIndexHeader(metadata, CODEC_NAME, VERSION_CURRENT, segmentWriteState.segmentInfo.getId(), segmentWriteState.segmentSuffix);
            success = true;
        } finally {
            if (!success && vectordata != null) {
                IOUtils.closeWhileHandlingException(vectordata, metadata);
            }
        }
        this.vectordata = vectordata;
        this.metadata = metadata;
    }

    /** Supplies the float vector for a given original (insertion-order) ordinal. */
    @FunctionalInterface
    private interface FloatVectorProvider {
        float[] apply(int originalOrdinal) throws IOException;
    }

    /**
     * Writes one field's quantized vectors: metadata + mappings first, then the records in the given
     * physical ordering. Shared by the flush and merge paths; the only difference between them is
     * where the source float vector for an original ordinal comes from, which is supplied by
     * {@code vectorProvider}.
     *
     * @param fieldInfo        the field being written
     * @param centroid         the centroid vectors are centered against at quantization time
     * @param physicalOrdinals {@code physicalOrdinals[physicalPos] = originalOrdinal}; the permutation
     *                         defining physical layout order. Length == number of vectors.
     * @param vectorProvider   maps an original ordinal to its (already normalized, if COSINE) float
     *                         vector; the value is quantized here, not re-quantized
     */
    private void writeFieldInternal(
        final FieldInfo fieldInfo,
        final float[] centroid,
        final IntArrayList physicalOrdinals,
        final FloatVectorProvider vectorProvider
    ) throws IOException {
        if (wroteField) {
            throw new IllegalStateException("LocalityOrderedQuantizedVectorsWriter supports a single field per file");
        }
        wroteField = true;
        // Metadata (scalars + centroid + ordinal map) -> .vemlo; the permuted quantized records -> .veqlo.
        writeMetadata(fieldInfo, centroid, physicalOrdinals);
        writeVectorData(fieldInfo, centroid, physicalOrdinals, vectorProvider);
    }

    /**
     * Writes one field's metadata to the {@code .vemlo} file: the scalar header, the centroid, and
     * the {@code originalOrdinal -> physicalPosition} map.
     */
    private void writeMetadata(final FieldInfo fieldInfo, final float[] centroid, final IntArrayList physicalOrdinals)
        throws IOException {
        final int count = physicalOrdinals.size();
        final int dimension = fieldInfo.getVectorDimension();
        final float centroidDp = count > 0 ? VectorUtil.dotProduct(centroid, centroid) : 0;
        // Scalar-encoding wire number; the reader recovers the encoding via ScalarEncoding.fromWireNumber.
        // (The similarity function is NOT stored here — the reader derives it from FieldInfo.)
        final int encodingWireNumber = scalarEncoding.getWireNumber();
        // Number of bytes of the packed quantized vector.
        final int codeLength = scalarEncoding.getDocPackedLength(dimension);
        // Number of bytes per record, including the correction factors.
        final int recordSize = codeLength + CORRECTION_BYTES;

        // field.number stays a fixed writeInt (mirrors Lucene, which uses -1 as an end-of-fields
        // marker). The rest are small non-negative ints -> VInt to save space (like Lucene's meta).
        metadata.writeInt(fieldInfo.number);
        metadata.writeVInt(dimension);
        metadata.writeVInt(count);
        metadata.writeVInt(codeLength);
        metadata.writeVInt(recordSize);
        metadata.writeVInt(encodingWireNumber);

        final ByteBuffer buffer = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(centroid);
        metadata.writeBytes(buffer.array(), buffer.array().length);
        // centroidDp is a full 32-bit float pattern -> keep fixed writeInt (VInt would usually be larger).
        metadata.writeInt(Float.floatToIntBits(centroidDp));

        // originalOrdinal -> physicalPosition, the inverse of the physical-layout permutation.
        final int[] ordToPhysicalOrdMap = new int[count];
        for (int i = 0; i < count; i++) {
            ordToPhysicalOrdMap[physicalOrdinals.get(i)] = i;
        }

        // Pack with DirectWriter at a fixed bits/value (values are in [0, count), so the max is
        // count-1 -> bitsRequired(count-1)). Near-optimal for a permutation (uniform values, not
        // monotonic -> no DirectMonotonic). Only requiredBits (the width) is written; the reader
        // recomputes the exact block length (padding included) as bytesRequired(count, requiredBits),
        // so we don't spend bytes storing it.
        final int requiredBits = DirectWriter.bitsRequired(count - 1);
        metadata.writeVInt(requiredBits);
        final DirectWriter directWriter = DirectWriter.getInstance(metadata, count, requiredBits);
        for (int p = 0; p < count; p++) {
            directWriter.add(ordToPhysicalOrdMap[p]);
        }
        directWriter.finish();
    }

    /**
     * Quantizes each vector in physical (permuted) order and writes its record to the {@code .veqlo}
     * file: the packed 1-bit code followed by the four correction terms.
     */
    private void writeVectorData(
        final FieldInfo fieldInfo,
        final float[] centroid,
        final IntArrayList physicalOrdinals,
        final FloatVectorProvider vectorProvider
    ) throws IOException {
        final int count = physicalOrdinals.size();
        final int dimension = fieldInfo.getVectorDimension();
        final int codeLength = scalarEncoding.getDocPackedLength(dimension);
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());

        final byte[] quantizedVectorScratch = new byte[scalarEncoding.getDiscreteDimensions(dimension)];
        final byte[] vectorToBeWritten = new byte[codeLength];

        for (int p = 0; p < count; p++) {
            final float[] vectorToQuantize = vectorProvider.apply(physicalOrdinals.get(p));
            // Quantize the vector first
            final OptimizedScalarQuantizer.QuantizationResult quantizationResult = quantizer.scalarQuantize(
                vectorToQuantize,
                quantizedVectorScratch,
                scalarEncoding.getBits(),
                centroid
            );
            // pack the vector into bytes
            OptimizedScalarQuantizer.packAsBinary(quantizedVectorScratch, vectorToBeWritten);
            // write the vector first, and then the correction terms
            vectordata.writeBytes(vectorToBeWritten, vectorToBeWritten.length);
            vectordata.writeInt(Float.floatToIntBits(quantizationResult.lowerInterval()));
            vectordata.writeInt(Float.floatToIntBits(quantizationResult.upperInterval()));
            vectordata.writeInt(Float.floatToIntBits(quantizationResult.additionalCorrection()));
            vectordata.writeInt(quantizationResult.quantizedComponentSum());
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        final FlatFieldVectorsWriter<?> flatFieldVectorsWriter = this.rawFlatVectorsWriter.addField(fieldInfo);
        if(fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
            @SuppressWarnings("unchecked")
            final SQFieldWriter sqFieldWriter = new SQFieldWriter(fieldInfo, (FlatFieldVectorsWriter<float[]>)flatFieldVectorsWriter);
            fields.add(sqFieldWriter);
            return sqFieldWriter;
        }
        // This should never happen, we should just throw exception from here.
        return flatFieldVectorsWriter;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        rawFlatVectorsWriter.flush(maxDoc, sortMap);
        if(fields.isEmpty()) {
            return;
        }
        // TODO : Come and fix this.
        if(fields.size() > 1) {
            throw new UnsupportedOperationException("For more than 1 field we don't support LocalityOrderedQuantizedVectorsWriter");
        }
        final SQFieldWriter sqFieldWriter = fields.get(0);
        // after raw vectors are written, normalize vectors for clustering and quantization
        if (VectorSimilarityFunction.COSINE == sqFieldWriter.fieldInfo.getVectorSimilarityFunction()) {
            sqFieldWriter.normalizeVectors();
        }
        int dimensions = sqFieldWriter.fieldInfo.getVectorDimension();
        final float[] centroid = new float[sqFieldWriter.fieldInfo.getVectorDimension()];
        int vectorCount = sqFieldWriter.getVectors().size();

        if(vectorCount > 0) {
            for(int i = 0 ; i < dimensions; i++) {
                centroid[i] = sqFieldWriter.dimensionSums[i] / vectorCount;
            }
            if (VectorSimilarityFunction.COSINE == sqFieldWriter.fieldInfo.getVectorSimilarityFunction()) {
                VectorUtil.l2normalize(centroid);
            }
        }

        if (segmentWriteState.infoStream.isEnabled(KNN1040LocalityAwareSQVectorsFormat.QUANTIZED_VECTOR_COMPONENT)) {
            segmentWriteState.infoStream.message(
                    KNN1040LocalityAwareSQVectorsFormat.QUANTIZED_VECTOR_COMPONENT, "Vectors' count:" + vectorCount);
        }

        // now we have to write the field
        final DocIdSetIterator disi = sqFieldWriter.getDocsWithFieldSet().iterator();
        final IntArrayList physicalOrdinalsList = new IntArrayList();
        int i = 0;
        for(int docId = disi.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = disi.nextDoc()) {
            physicalOrdinalsList.add(i);
            i++;
        }
        // This is just a temporary thing, in original implementation this will come from BFS
        randomizeOrdinals(physicalOrdinalsList);
        // Flush path: vectors are the in-memory (already normalized, if COSINE) float vectors.
        writeFieldInternal(sqFieldWriter.fieldInfo, centroid, physicalOrdinalsList, ord -> sqFieldWriter.getVectors().get(ord));
        sqFieldWriter.finish();
    }

    private static void randomizeOrdinals(final IntArrayList ordinals) {
        final Random random = new Random(1234);
        for (int i = ordinals.size() - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);
            final int tmp = ordinals.get(i);
            ordinals.set(i, ordinals.get(j));
            ordinals.set(j, tmp);
        }
    }

    private void ensureFlatReaderOpen() throws IOException {
        if (flatVectorsReader == null) {
            rawFlatVectorsWriter.finish();
            rawFlatVectorsWriter.close();
            flatWriterClosed = true;
            SegmentReadState readState =
                    new SegmentReadState(
                            segmentWriteState.directory,
                            segmentWriteState.segmentInfo,
                            segmentWriteState.fieldInfos,
                            segmentWriteState.context,
                            segmentWriteState.segmentSuffix);
            flatVectorsReader = flatVectorsFormat.fieldsReader(readState);
        }
    }

    @Override
    public void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        rawFlatVectorsWriter.mergeOneFlatVectorField(fieldInfo, mergeState);
        if (!fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
            return;
        }

        final float[] centroid;
        final float[] mergedCentroid = new float[fieldInfo.getVectorDimension()];
        int vectorCount = mergeAndRecalculateCentroids(mergeState, fieldInfo, mergedCentroid);
        centroid = mergedCentroid;
        if (segmentWriteState.infoStream.isEnabled(QUANTIZED_VECTOR_COMPONENT)) {
            segmentWriteState.infoStream.message(
                    QUANTIZED_VECTOR_COMPONENT, "Vectors' count:" + vectorCount);
        }
        // Lazily finish flat writer and open a reader for the written segment
        ensureFlatReaderOpen();
        FloatVectorValues floatVectorValues = flatVectorsReader.getFloatVectorValues(fieldInfo.name);

        if (fieldInfo.getVectorSimilarityFunction() == COSINE) {
            floatVectorValues = new NormalizedFloatVectorValues(floatVectorValues);
        }

        final DocIdSetIterator disi = floatVectorValues.iterator();
        final IntArrayList physicalOrdinalsList = new IntArrayList();
        int i = 0;
        for(int docId = disi.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = disi.nextDoc()) {
            physicalOrdinalsList.add(i);
            i++;
        }
        randomizeOrdinals(physicalOrdinalsList);

        // Merge path: vectors come (by original ordinal) from the reopened merged flat reader,
        // which supports random access. floatVectorValues is reassigned above for COSINE, so
        // capture it in a final for the provider.
        final FloatVectorValues mergedVectors = floatVectorValues;
        writeFieldInternal(fieldInfo, centroid, physicalOrdinalsList, mergedVectors::vectorValue);
    }



    private int mergeAndRecalculateCentroids(MergeState mergeState, FieldInfo fieldInfo, float[] mergedCentroid) throws IOException {
        int totalVectorCount = 0;

        for(int i = 0 ; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
            if (knnVectorsReader == null
                    || knnVectorsReader.getFloatVectorValues(fieldInfo.name) == null) {
                continue;
            }
            float[] centroid = getCentroid(knnVectorsReader, fieldInfo.name);
            if(centroid == null) {
                continue;
            }
            int vectorCount = knnVectorsReader.getFloatVectorValues(fieldInfo.name).size();
            if (vectorCount == 0) {
                continue;
            }
            totalVectorCount += vectorCount;
            for (int j = 0; j < centroid.length; j++) {
                mergedCentroid[j] += centroid[j] * vectorCount;
            }
        }

        if (totalVectorCount == 0) {
            return 0;
        } else {
            for(int j = 0 ; j < mergedCentroid.length; j++) {
                mergedCentroid[j] = mergedCentroid[j] / totalVectorCount;
            }
            if (fieldInfo.getVectorSimilarityFunction() == COSINE) {
                VectorUtil.l2normalize(mergedCentroid);
            }
            return totalVectorCount;
        }
    }

    private float[] getCentroid(KnnVectorsReader vectorsReader, String fieldName) {
        if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader candidateReader) {
            vectorsReader = candidateReader.getFieldReader(fieldName);
        }

        // This might come handy for BWC for merging old segments to new segments, otherwise we can just remove it
        if (vectorsReader instanceof Lucene104ScalarQuantizedVectorsReader reader) {
            return reader.getCentroid(fieldName);
        }

        if (vectorsReader instanceof LocalityOrderedQuantizedVectorsReader reader) {
            return reader.getCentroid(fieldName);
        }
        return null;
    }


    /** Writes the footer. Must be called once after {@link #writeFieldInternal}. */
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        if (!wroteField) {
            throw new IllegalStateException("finish() called before writeField()");
        }
        if(flatWriterClosed == false) {
            rawFlatVectorsWriter.finish();
        }
        CodecUtil.writeFooter(metadata);
        CodecUtil.writeFooter(vectordata);
    }

    @Override
    public void close() throws IOException {
        // Merge path: the raw writer is already closed (flatWriterClosed) and a reopened flat reader
        // may be held. Flush path: we still own the raw writer and no reader was opened.
        if (flatWriterClosed) {
            IOUtils.close(vectordata, metadata, flatVectorsReader);
        } else {
            IOUtils.close(vectordata, metadata, rawFlatVectorsWriter);
        }
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    private static final class SQFieldWriter extends FlatFieldVectorsWriter<float[]> {

        private final FieldInfo fieldInfo;
        private boolean finished;
        private final FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter;
        // This will be later used to calculate the centroid
        private final float[] dimensionSums;
        private final FloatArrayList magnitudes = new FloatArrayList();

        SQFieldWriter(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter) {
            this.fieldInfo = fieldInfo;
            this.flatFieldVectorsWriter = flatFieldVectorsWriter;
            this.dimensionSums = new float[fieldInfo.getVectorDimension()];
        }

        @Override
        public List<float[]> getVectors() {
            return flatFieldVectorsWriter.getVectors();
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return flatFieldVectorsWriter.getDocsWithFieldSet();
        }

        @Override
        public void finish() throws IOException {
            if (finished) {
                return;
            }
            assert flatFieldVectorsWriter.isFinished();
            finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished && flatFieldVectorsWriter.isFinished();
        }

        @Override
        public void addValue(int docID, float[] vectorValue) throws IOException {
            flatFieldVectorsWriter.addValue(docID, vectorValue);
            // we might never hit this case since we always use IP when we have cosine
            if(fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE) {
                float dp = VectorUtil.dotProduct(vectorValue, vectorValue);
                float divisor = (float) Math.sqrt(dp);
                magnitudes.add(divisor);
                for (int i = 0; i < vectorValue.length; i++) {
                    dimensionSums[i] += (vectorValue[i] / divisor);
                }
            } else {
                for (int i = 0; i < vectorValue.length; i++) {
                    dimensionSums[i] += vectorValue[i];
                }
            }
        }

        public void normalizeVectors() {
            for (int i = 0; i < flatFieldVectorsWriter.getVectors().size(); i++) {
                float[] vector = flatFieldVectorsWriter.getVectors().get(i);
                float magnitude = magnitudes.get(i);
                for (int j = 0; j < vector.length; j++) {
                    vector[j] /= magnitude;
                }
            }
        }

        @Override
        public float[] copyValue(float[] vectorValue) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long ramBytesUsed() {
            return 0;
        }
    }

    static final class NormalizedFloatVectorValues extends FloatVectorValues {
        private final FloatVectorValues values;
        private final float[] normalizedVector;

        NormalizedFloatVectorValues(FloatVectorValues values) {
            this.values = values;
            this.normalizedVector = new float[values.dimension()];
        }

        @Override
        public int dimension() {
            return values.dimension();
        }

        @Override
        public int size() {
            return values.size();
        }

        @Override
        public int ordToDoc(int ord) {
            return values.ordToDoc(ord);
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            System.arraycopy(values.vectorValue(ord), 0, normalizedVector, 0, normalizedVector.length);
            VectorUtil.l2normalize(normalizedVector);
            return normalizedVector;
        }

        @Override
        public DocIndexIterator iterator() {
            return values.iterator();
        }

        @Override
        public NormalizedFloatVectorValues copy() throws IOException {
            return new NormalizedFloatVectorValues(values.copy());
        }
    }
}
