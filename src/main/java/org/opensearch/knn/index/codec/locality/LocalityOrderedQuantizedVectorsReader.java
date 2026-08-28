/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectReader;
import org.apache.lucene.util.packed.DirectWriter;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;
import org.apache.lucene.util.quantization.ScalarQuantizer;
import org.opensearch.knn.index.codec.KNN1040Codec.ScalarQuantizedFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

import static org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter.CODEC_NAME;
import static org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter.EXTENSION;
import static org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter.METADATA_EXTENSION;
import static org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter.VERSION_CURRENT;
import static org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter.VERSION_START;

/**
 * Reads the two-file locality-ordered store written by {@link LocalityOrderedQuantizedVectorsWriter}:
 * a metadata file ({@code .vemlo}) and a records file ({@code .veqlo}). See that writer for the exact
 * on-disk layout.
 *
 * <p>The store holds only the permuted quantized records plus an
 * {@code originalOrdinal -> physicalPosition} map. docId resolution and document iteration are
 * <b>not</b> served here — they are delegated to the raw flat vectors reader ({@code .vec}), which
 * owns the {@code ordinal -> docId} mapping and the doc-ascending iterator. This reader only
 * translates an original ordinal to its physical record offset.
 *
 * <p>Parsing: the {@code .vemlo} scalars are read front-to-back; the {@code DirectWriter} block
 * ({@code ordToPhysicalOrdMap}) is sliced using its exact length, recomputed as
 * {@code DirectWriter.bytesRequired(count, requiredBits)} (see {@link #readFieldEntry}). The
 * {@code .veqlo} record region is {@code [indexHeaderLength, fileLength - footerLength)} — it starts
 * right after the data file's header, independent of metadata size. Per-field parsed state is held
 * in a {@link FieldEntry}.
 */
public final class LocalityOrderedQuantizedVectorsReader extends FlatVectorsReader implements QuantizedVectorsReader {

    private static final IOContext.FileOpenHint[] RANDOM_ACCESS_HINT = Stream.of(
        FileTypeHint.DATA,
        FileDataHint.KNN_VECTORS,
        DataAccessHint.RANDOM
    ).filter(Objects::nonNull).toArray(IOContext.FileOpenHint[]::new);

    // Plain Lucene104 (Java ADC) scorer used to score the query against the in-heap hub records. It
    // reads through the values API, so it works on heap-backed values with no off-heap slice — unlike
    // the SIMD scorer, which requires one.
    private static final Lucene104ScalarQuantizedVectorScorer HUB_SCORER = new Lucene104ScalarQuantizedVectorScorer(
        FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()
    );

    private final SegmentReadState segmentReadState;
    private final IndexInput metaInput;
    private final IndexInput dataInput;
    private final FlatVectorsReader rawFlatVectorsReader;
    private final Lucene104ScalarQuantizedVectorScorer quantizedVectorScorer;
    private final Map<String, FieldEntry> fields = new HashMap<>();

    public LocalityOrderedQuantizedVectorsReader(
        final SegmentReadState state,
        final FlatVectorsReader rawFlatVectorsReader,
        final Lucene104ScalarQuantizedVectorScorer quantizedVectorScorer
    ) throws IOException {
        final String dataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, EXTENSION);
        final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, METADATA_EXTENSION);
        this.segmentReadState = state;
        this.rawFlatVectorsReader = rawFlatVectorsReader;
        this.quantizedVectorScorer = quantizedVectorScorer;

        boolean success = false;
        IndexInput meta = null;
        IndexInput data = null;
        try {
            // Metadata (.vemlo): per-field scalars + centroid + ordToPhysicalOrdMap.
            meta = state.directory.openInput(metaFileName, state.context.withHints(RANDOM_ACCESS_HINT));
            CodecUtil.checkIndexHeader(meta, CODEC_NAME, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);

            // Data (.veqlo): the permuted quantized records.
            data = state.directory.openInput(dataFileName, state.context.withHints(RANDOM_ACCESS_HINT));
            CodecUtil.checkIndexHeader(data, CODEC_NAME, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);

            // we will have only 1 field for now
            final FieldEntry entry = readFieldEntry(meta, data, state.fieldInfos, state.segmentSuffix);

            // Validate the footer checksums (each seek to end; done after we've captured the record slice).
            CodecUtil.retrieveChecksum(meta);
            CodecUtil.retrieveChecksum(data);

            // Resolve the field name from its number and register the parsed entry.
            final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(entry.fieldNumber);
            if (fieldInfo == null) {
                throw new IllegalStateException("Unknown field number [" + entry.fieldNumber + "] in " + metaFileName);
            }
            fields.put(fieldInfo.name, entry);

            this.metaInput = meta;
            this.dataInput = data;
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(meta, data);
            }
        }
    }

    /**
     * Parses one field's metadata from the {@code .vemlo} meta input and slices its record region
     * out of the {@code .veqlo} data input. Both files currently hold a single field.
     */
    private static FieldEntry readFieldEntry(
        final IndexInput meta,
        final IndexInput data,
        final FieldInfos fieldInfos,
        final String segmentSuffix
    ) throws IOException {
        final int fieldNumber = meta.readInt();
        final int dimension = meta.readVInt();
        final int count = meta.readVInt();
        final int codeLength = meta.readVInt();
        final int recordSize = meta.readVInt();
        final QuantizedByteVectorValues.ScalarEncoding scalarEncoding = QuantizedByteVectorValues.ScalarEncoding.fromWireNumber(
            meta.readVInt()
        ).get();

        final float[] centroid = new float[dimension];
        meta.readFloats(centroid, 0, dimension);

        final float centroidDp = Float.intBitsToFloat(meta.readInt());

        // originalOrdinal -> physicalPosition, DirectWriter-packed. Only requiredBits is stored; the
        // exact block length (padding included) is recomputed as DirectWriter.bytesRequired(count,
        // requiredBits), so no byte length is spent on disk. docId is NOT stored here; it is resolved
        // via the raw flat vectors reader (originalOrdinal -> docId).
        final int requiredBits = meta.readVInt();
        final long packedStart = meta.getFilePointer();
        final long packedLen = DirectWriter.bytesRequired(count, requiredBits);
        final RandomAccessInput packed = meta.randomAccessSlice(packedStart, packedLen);
        final LongValues directReader = DirectReader.getInstance(packed, requiredBits);
        final int[] ordToPhysicalOrdMap = new int[count];
        for (int i = 0; i < count; i++) {
            ordToPhysicalOrdMap[i] = (int) directReader.get(i);
        }

        final String fieldName = fieldInfos.fieldInfo(fieldNumber).getName();

        // Hub section lives at the end, right after the DirectWriter block (which was read via a slice
        // and did NOT advance meta). Seek past it, then read the entry-point candidates: numHubs, the
        // hub ordinals (highest-degree first), then the numHubs quantized records — parsed straight into
        // an in-heap QuantizedByteVectorValues so the entry-point scorer can be built anywhere (no slice).
        meta.seek(packedStart + packedLen);
        final int numHubs = meta.readVInt();
        final int[] hubOrdinals = new int[numHubs];
        for (int i = 0; i < numHubs; i++) {
            hubOrdinals[i] = meta.readVInt();
        }
        final OptimizedScalarQuantizer hubQuantizer = new OptimizedScalarQuantizer(
            fieldInfos.fieldInfo(fieldNumber).getVectorSimilarityFunction()
        );
        final HeapQuantizedByteVectorValues hubValues = HeapQuantizedByteVectorValues.read(
            meta,
            numHubs,
            dimension,
            codeLength,
            centroid,
            centroidDp,
            hubQuantizer,
            scalarEncoding
        );

        // Records live in the data file, right after its index header.
        final long dataOffset = CodecUtil.indexHeaderLength(CODEC_NAME, segmentSuffix);
        final IndexInput dataSlice = data.slice("veqo-records" + fieldName, dataOffset, (long) count * recordSize);

        return new FieldEntry(
            fieldNumber,
            dimension,
            count,
            codeLength,
            recordSize,
            centroidDp,
            scalarEncoding,
            centroid,
            ordToPhysicalOrdMap,
            dataSlice,
            hubOrdinals,
            hubValues
        );
    }

    @Override
    public void close() throws IOException {
        // Close our own meta + data inputs and the raw flat vectors reader we own (holds the .vec handle).
        IOUtils.close(metaInput, dataInput, rawFlatVectorsReader);
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        // This reader serves quantized vector values, so the appropriate scorer is the
        // quantized (ADC) scorer — not the raw full-precision reader's scorer.
        return quantizedVectorScorer;
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        final FieldEntry entry = fields.get(field);
        if (entry == null) {
            return null;
        }
        final VectorSimilarityFunction similarity = segmentReadState.fieldInfos.fieldInfo(field).getVectorSimilarityFunction();
        // The delegate scores by PHYSICAL position. Give it an identity (physical-addressed) view so
        // BOTH scorer implementations read the right record: the SIMD path via raw arithmetic
        // (base + ord*recordSize) and the Java fallback via vectorValue(ord). Passing the
        // original-ordinal view here would double-map on the Java fallback.
        final QuantizedByteVectorValues physicalValues = createValues(field, entry, null);
        final RandomVectorScorer delegate = quantizedVectorScorer.getRandomVectorScorer(similarity, physicalValues, target);

        // The wrapper stays in ORIGINAL-ordinal space for ordToDoc/maxOrd/getAcceptOrds (the graph's
        // space) and translates each graph ordinal -> physical before delegating the score.
        final QuantizedByteVectorValues originalValues = createValues(field, entry, entry.ordToPhysicalOrdMap);
        return new PhysicalOrdinalTranslatingScorer(originalValues, delegate, entry.ordToPhysicalOrdMap);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        // FLOAT32-only field; byte-vector queries are not supported over scalar-quantized data.
        return rawFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public void checkIntegrity() throws IOException {
        // Validate the raw full-precision file plus our own quantized meta (.vemlo) and data (.veqlo) files.
        rawFlatVectorsReader.checkIntegrity();
        CodecUtil.checksumEntireFile(metaInput);
        CodecUtil.checksumEntireFile(dataInput);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FloatVectorValues floatVectorValues = rawFlatVectorsReader.getFloatVectorValues(field);
        if (floatVectorValues == null) {
            return null;
        }

        if (floatVectorValues.size() == 0) {
            return new ScalarQuantizedFloatVectorValues(floatVectorValues, null);
        }

        return new ScalarQuantizedFloatVectorValues(floatVectorValues, getQuantizedVectorValues(field));
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return rawFlatVectorsReader.getByteVectorValues(field);
    }

    /**
     * @return the hub entry-point ordinals for the field (original ordinals, highest-degree first), or
     * an empty array if the field is unknown or has no hubs
     */
    public int[] getHubOrdinals(final String field) {
        final FieldEntry entry = fields.get(field);
        return entry == null ? new int[0] : entry.hubOrdinals;
    }

    /**
     * Picks the single hub closest to {@code target} to seed the HNSW search.
     *
     * <p>Scores the query against each stored hub's quantized record (from the metadata hub section)
     * using the same SQ ADC scorer used for the graph, and returns the <b>original ordinal</b> of the
     * best-scoring hub. The returned ordinal is a valid graph entry point (no {@code acceptOrds}
     * filtering is applied here — the filter is honored later during traversal/collection).
     *
     * @return the best hub's original ordinal, or {@code -1} if the field is unknown or has no hubs
     */
    public int selectBestHubOrdinal(final String field, final float[] target) throws IOException {
        final FieldEntry entry = fields.get(field);
        if (entry == null || entry.hubOrdinals.length == 0) {
            return -1;
        }
        final VectorSimilarityFunction similarity = segmentReadState.fieldInfos.fieldInfo(field).getVectorSimilarityFunction();

        // The hubs are already an in-heap QuantizedByteVectorValues; score the query against all of them
        // in one bulk pass. HUB_SCORER is the plain Lucene104 (Java ADC) scorer — the heap values have no
        // off-heap slice, so the SIMD path doesn't apply (and isn't worth it for <=32 hubs).
        final RandomVectorScorer scorer = HUB_SCORER.getRandomVectorScorer(similarity, entry.hubValues, target);
        final int numHubs = entry.hubOrdinals.length;
        final int[] hubIndices = new int[numHubs];
        for (int i = 0; i < numHubs; i++) {
            hubIndices[i] = i;
        }
        final float[] scores = new float[numHubs];
        scorer.bulkScore(hubIndices, scores, numHubs);

        int bestHubIndex = 0;
        for (int i = 1; i < numHubs; i++) {
            if (scores[i] > scores[bestHubIndex]) {
                bestHubIndex = i;
            }
        }
        return entry.hubOrdinals[bestHubIndex];
    }

    @Override
    public QuantizedByteVectorValues getQuantizedVectorValues(String fieldName) throws IOException {
        final FieldEntry entry = fields.get(fieldName);
        if (entry == null) {
            throw new IllegalArgumentException("Field is not a locality-ordered SQ vector field: " + fieldName);
        }
        // Original-ordinal-addressed view (maps original -> physical); this is the view exposed for
        // iteration / ordToDoc, in the raw reader's ordinal space.
        return createValues(fieldName, entry, entry.ordToPhysicalOrdMap);
    }

    /**
     * Builds a {@link LocalityOrderedQuantizedByteVectorValues} over this field's records.
     *
     * @param ordinalMap {@code entry.ordToPhysicalOrdMap} for the original-ordinal-addressed view
     *                   (iteration/ordToDoc), or {@code null} for the identity (physical-addressed)
     *                   view the scorer reads records from.
     */
    private LocalityOrderedQuantizedByteVectorValues createValues(final String field, final FieldEntry entry, final int[] ordinalMap)
        throws IOException {
        final FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
        final FloatVectorValues floatVectorValues = rawFlatVectorsReader.getFloatVectorValues(field);
        final VectorSimilarityFunction similarity = fieldInfo.getVectorSimilarityFunction();
        final OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(similarity);
        return new LocalityOrderedQuantizedByteVectorValues(
            entry.dimension,
            entry.count,
            entry.codeLength,
            entry.recordSize,
            entry.centroid,
            entry.centroidDp,
            quantizer,
            entry.scalarEncoding,
            similarity,
            quantizedVectorScorer,
            ordinalMap,
            entry.dataSlice.clone(),
            floatVectorValues
        );
    }

    @Override
    public ScalarQuantizer getQuantizationState(String fieldName) {
        return null;
    }

    @Override
    public CloseableRandomVectorScorerSupplier getRandomVectorScorerSupplierForMerge(
        FieldInfo fieldInfo,
        SegmentWriteState segmentWriteState
    ) throws IOException {
        // Intentionally unsupported: our writer re-quantizes from the raw flat vectors on merge and
        // does not build an HNSW graph over this reader during merge, so no scorer supplier is needed.
        return null;
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    public float[] getCentroid(String field) {
        FieldEntry fieldEntry = fields.get(field);
        if (fieldEntry != null) {
            return fieldEntry.centroid;
        }
        return null;
    }

    /**
     * Bridges an <b>original-ordinal</b> HNSW graph to the <b>physically-permuted</b> {@code .veqo}
     * store (design Option C).
     *
     * <p>The Lucene HNSW graph traverses in original-ordinal space, but the underlying SIMD scorer
     * addresses records by physical position ({@code base + ord * recordSize}) and never routes
     * through {@link LocalityOrderedQuantizedByteVectorValues#vectorValue(int)} / the
     * {@code ordToPhysicalOrdMap}. This wrapper therefore translates each graph ordinal to its
     * physical position before delegating the score, while {@code ordToDoc}/{@code maxOrd}/
     * {@code getAcceptOrds} stay in original-ordinal space (inherited from the values).
     *
     * <p>Prefetch stays correct: the delegate's bulk path prefetches via
     * {@code PrefetchableVectorValuesHelper.doPrefetch}, which addresses {@code getSlice()} by raw
     * ordinal arithmetic (the same space as the native SIMD read) rather than through the mapping
     * {@code values.prefetch}. Since we pass it the already-translated physical ordinals, it warms
     * exactly the records the native score will read.
     */
    public static final class PhysicalOrdinalTranslatingScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final RandomVectorScorer delegate;
        private final int[] ordToPhysicalOrdMap;
        private int[] scratch = new int[0];

        PhysicalOrdinalTranslatingScorer(final KnnVectorValues values, final RandomVectorScorer delegate, final int[] ordToPhysicalOrdMap) {
            super(values);
            this.delegate = delegate;
            this.ordToPhysicalOrdMap = ordToPhysicalOrdMap;
        }

        @Override
        public float score(final int node) throws IOException {
            return delegate.score(ordToPhysicalOrdMap[node]);
        }

        @Override
        public float bulkScore(final int[] nodes, final float[] scores, final int numNodes) throws IOException {
            if (scratch.length < numNodes) {
                scratch = new int[numNodes];
            }
            for (int i = 0; i < numNodes; i++) {
                scratch[i] = ordToPhysicalOrdMap[nodes[i]];
            }
            return delegate.bulkScore(scratch, scores, numNodes);
        }
    }

    /**
     * Per-field state parsed from a {@code .veqo} file. Records are laid out by <b>physical</b>
     * (post-ordering) position: the record for physical position {@code p} lives at
     * {@code dataSlice[p * recordSize]}. {@code ordToPhysicalOrdMap} translates an original
     * (insertion-order) ordinal to the physical position holding its record. docId is not stored
     * here; it is obtained from the raw flat vectors reader by original ordinal.
     */
    private static final class FieldEntry {
        /** Lucene field number this entry was written for. */
        final int fieldNumber;
        /** Vector dimension. */
        final int dimension;
        /** Number of vectors (physical positions 0..count-1). */
        final int count;
        /** Packed 1-bit code length per record, {@code ceil(dimension / 8)} bytes. */
        final int codeLength;
        /** Total bytes per record, {@code codeLength + 16} (code + 4 correction fields). */
        final int recordSize;
        /** Dot product of the centroid with itself, used in ADC scoring. */
        final float centroidDp;
        /** {@code VectorSimilarityFunction.ordinal()} the vectors were quantized against. */
        final QuantizedByteVectorValues.ScalarEncoding scalarEncoding;
        /** The centroid vectors were centered against at quantization time. */
        final float[] centroid;
        /** Bounded view over the record region, addressed by physical ordinal. */
        final IndexInput dataSlice;

        /** Original (insertion-order) ordinal -&gt; physical position of its record in {@link #dataSlice}. */
        final int[] ordToPhysicalOrdMap;

        /** Hub entry-point candidates: original ordinals, highest-degree first. */
        final int[] hubOrdinals;
        /** In-heap quantized records for the hubs (parallel to {@link #hubOrdinals}), ready to score. */
        final QuantizedByteVectorValues hubValues;

        FieldEntry(
            final int fieldNumber,
            final int dimension,
            final int count,
            final int codeLength,
            final int recordSize,
            final float centroidDp,
            final QuantizedByteVectorValues.ScalarEncoding scalarEncoding,
            final float[] centroid,
            final int[] ordToPhysicalOrdMap,
            final IndexInput dataSlice,
            final int[] hubOrdinals,
            final QuantizedByteVectorValues hubValues
        ) {
            this.fieldNumber = fieldNumber;
            this.dimension = dimension;
            this.count = count;
            this.codeLength = codeLength;
            this.recordSize = recordSize;
            this.centroidDp = centroidDp;
            this.scalarEncoding = scalarEncoding;
            this.centroid = centroid;
            this.dataSlice = dataSlice;
            this.ordToPhysicalOrdMap = ordToPhysicalOrdMap;
            this.hubOrdinals = hubOrdinals;
            this.hubValues = hubValues;
        }
    }
}
