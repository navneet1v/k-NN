/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOFunction;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.model.Layer0LocalityOrdering;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Writer for locality-reordered Faiss SQ 1-bit HNSW fields (single field per format instance).
 *
 * <p>It drives <b>two</b> flat writers:
 * <ol>
 *   <li><b>SQ flat writer</b> ({@code sqFlatVectorsWriter}) — the standard scalar-quantized flat
 *       store. It receives the vectors (via {@link #addField}) and, on flush/merge, produces the
 *       ordinal-order quantized store from which the native Faiss HNSW graph is built (.faiss).</li>
 *   <li><b>Locality writer</b> ({@code localityVectorsWriter}) — writes the physically
 *       <b>reordered</b> quantized store (.veqlo/.vemlo), where records are laid out in the
 *       graph-derived page-locality permutation.</li>
 * </ol>
 *
 * <p><b>Build order</b> (per {@code Faiss_HNSW_Locality_Integration_Proposal.md}):
 * quantize (SQ flat) → build HNSW graph (native) → derive the locality permutation from the graph →
 * write the reordered store (locality) by copying the already-quantized codes in permuted order.
 * The graph must be built single-layer for locality (see
 * {@code MemOptimizedScalarQuantizedIndexBuildStrategy} + {@code setFaissSQHnswToSingleLayer}).
 *
 * <p><b>How the permutation is delivered.</b> It is derived from the graph adjacency, which only
 * exists inside the native build (the build strategy holds the {@code indexMemoryAddress}). The
 * strategy computes it there and writes it into a {@link Layer0LocalityOrdering} sink that this writer
 * creates in {@link #flush}/{@link #mergeOneField} and threads down through {@code doFlush}/
 * {@code doMergeOneField}. After the build returns, {@link #writeReorderedLocalityStore} reads the sink
 * and hands the permutation to the locality writer.
 *
 * <p><b>Files.</b> The SQ flat store owns the raw {@code .vec} (kept for full-precision rescoring); the
 * locality store writes only {@code .veqlo}/{@code .vemlo} and emits no {@code .vec}, so there is no
 * file collision between the two.
 *
 * <p><b>Current limitation.</b> {@link LocalityOrderedQuantizedVectorsWriter#writeReorderedLocalityStore}
 * only handles a <b>dense</b> permutation (the BFS ordering,
 * {@code FaissService.buildOrderingOfVectorsUsingBFS}); the greedy strict-page ordering
 * ({@code buildOrderingOfVectorsUsingIndexStructure}) produces sparse (padded) physical positions it
 * does not yet support.
 *
 * @see Faiss1040ScalarQuantizedKnnVectorsWriter
 */
@Log4j2
class Faiss1040SQHNSWReorderedWriter extends AbstractNativeEnginesKnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(Faiss1040SQHNSWReorderedWriter.class);

    // The SQ flat store's quantized sub-files. These mirror the package-private
    // Lucene104ScalarQuantizedVectorsFormat.{VECTOR_DATA_EXTENSION, META_EXTENSION} (Lucene 10.5.0):
    // the quantized codes (.veq) and their meta (.vemq). The raw .vec/.vemf from the wrapped Lucene99
    // flat format are NOT listed here — they are kept for rescoring and merge re-quantization.
    private static final String SQ_QUANTIZED_DATA_EXTENSION = "veq";
    private static final String SQ_QUANTIZED_META_EXTENSION = "vemq";

    private final SegmentWriteState segmentWriteState;
    /** SQ flat store: receives the vectors and backs the native HNSW graph build. */
    private final FlatVectorsWriter sqFlatVectorsWriter;
    /** Reopens a reader over the just-written SQ flat store to extract quantized values. */
    private final IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier;
    /** Locality store: writes the reordered (.veqlo/.vemlo) quantized records. */
    private final LocalityOrderedQuantizedVectorsWriter localityVectorsWriter;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    // Single field — SQ gets a dedicated format per field via BasePerFieldKnnVectorsFormat.
    private FlatFieldVectorsWriter<?> fieldWriter;
    private FieldInfo fieldInfo;
    private boolean finished;

    Faiss1040SQHNSWReorderedWriter(
        @NonNull final SegmentWriteState segmentWriteState,
        @NonNull final FlatVectorsWriter sqFlatVectorsWriter,
        @NonNull final IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier,
        @NonNull final LocalityOrderedQuantizedVectorsWriter localityVectorsWriter,
        @NonNull final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.segmentWriteState = segmentWriteState;
        this.sqFlatVectorsWriter = sqFlatVectorsWriter;
        this.quantizedFlatVectorsReaderSupplier = quantizedFlatVectorsReaderSupplier;
        this.localityVectorsWriter = localityVectorsWriter;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Only one field is expected per format instance. The field's vectors are collected by the SQ
     * flat writer (the graph is built from the quantized store); the locality store is written later,
     * post-graph, by copying quantized codes in the reordered order.
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo newFieldInfo) throws IOException {
        if (this.fieldWriter != null) {
            throw new IllegalStateException(
                Faiss1040SQHNSWReorderedWriter.class.getSimpleName()
                    + " supports only a single field, but addField was called for ["
                    + newFieldInfo.name
                    + "] after ["
                    + this.fieldInfo.name
                    + "]"
            );
        }
        this.fieldInfo = newFieldInfo;
        this.fieldWriter = sqFlatVectorsWriter.addField(newFieldInfo);
        return fieldWriter;
    }

    /**
     * Flushes the SQ flat store, builds the native HNSW graph from it, then writes the reordered
     * locality store.
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        // Flush, finish, and close the SQ flat writer so its files are fully written and readable.
        sqFlatVectorsWriter.flush(maxDoc, sortMap);
        sqFlatVectorsWriter.finish();
        IOUtils.close(sqFlatVectorsWriter);

        if (fieldWriter == null) {
            return;
        }

        // Reopen the SQ flat store; extract quantized values and build the native HNSW graph.
        final FlatVectorsReader sqFlatVectorReader = openSqFlatVectorsReader();
        try {
            final QuantizedByteVectorValues quantizedValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                sqFlatVectorReader.getFloatVectorValues(fieldInfo.getName())
            );
            // Sink the native build fills with the graph-derived page-locality permutation.
            final Layer0LocalityOrdering layer0LocalityOrdering = new Layer0LocalityOrdering();
            doFlush(
                fieldInfo,
                fieldWriter,
                fieldWriter.getVectors(),
                null,
                null,
                segmentWriteState,
                nativeIndexBuildStrategyFactory,
                quantizedValues,
                layer0LocalityOrdering
            );
            // Post-graph: write the physically-reordered locality store.
            writeReorderedLocalityStore(fieldInfo, quantizedValues, layer0LocalityOrdering);
        } finally {
            IOUtils.close(sqFlatVectorReader);
        }
        // The reader over the SQ quantized files is now closed; drop the redundant .veq/.vemq — the
        // reordered .veqlo supersedes them. (Runs only on the non-empty path; the early return above
        // for a null fieldWriter leaves the SQ writer's empty files untouched.)
        deleteRedundantSqQuantizedFiles();
    }

    /**
     * Merges the SQ flat store, builds the native HNSW graph for the merged segment, then writes the
     * reordered locality store.
     */
    @Override
    public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        this.fieldInfo = fieldInfo;

        final IORunnable mergeRunnable = sqFlatVectorsWriter.mergeOneField(fieldInfo, mergeState);
        if (mergeRunnable != null) {
            mergeRunnable.run();
        }
        sqFlatVectorsWriter.finish();
        IOUtils.close(sqFlatVectorsWriter);

        final FlatVectorsReader flatVectorsReader = openSqFlatVectorsReader();
        try {
            final FloatVectorValues floatVectorValues = flatVectorsReader.getFloatVectorValues(fieldInfo.getName());
            if (floatVectorValues == null || floatVectorValues.size() == 0) {
                log.debug("No scalar-quantized vectors found for field [{}], skipping native build", fieldInfo.getName());
                return null;
            }
            final QuantizedByteVectorValues quantizedValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                floatVectorValues
            );
            // Sink the native build fills with the graph-derived page-locality permutation.
            final Layer0LocalityOrdering layer0LocalityOrdering = new Layer0LocalityOrdering();
            doMergeOneField(
                fieldInfo,
                mergeState,
                null,
                null,
                segmentWriteState,
                nativeIndexBuildStrategyFactory,
                quantizedValues,
                layer0LocalityOrdering
            );
            // Post-graph: write the physically-reordered locality store.
            writeReorderedLocalityStore(fieldInfo, quantizedValues, layer0LocalityOrdering);
        } finally {
            IOUtils.close(flatVectorsReader);
        }
        // Reader over the SQ quantized files is closed; drop the redundant .veq/.vemq for the merged
        // segment. The empty-field case early-returns above and never reaches here.
        deleteRedundantSqQuantizedFiles();
        return null;
    }

    /**
     * Writes the reordered locality store (.veqlo/.vemlo) for the field, copying the already-quantized
     * codes in the graph-derived page-locality order.
     *
     * <p>The permutation is produced by the native build and delivered via {@code layer0LocalityOrdering}
     * (see {@link Layer0LocalityOrdering} for why a sink is used instead of a return value). It is the
     * <b>forward</b> map {@code physicalOrdinals[originalOrdinal] = physicalPosition}; the locality
     * writer inverts it to lay records out in physical order.
     *
     * <p><b>Limitation:</b> {@link LocalityOrderedQuantizedVectorsWriter#writeReorderedLocalityStore}
     * currently only supports a <b>dense</b> permutation (the BFS ordering,
     * {@code buildOrderingOfVectorsUsingBFS}). The greedy strict-page ordering
     * ({@code buildOrderingOfVectorsUsingIndexStructure}) produces sparse physical positions (padding
     * gaps) that it does not yet handle — see that method's Javadoc.
     *
     * @param fieldInfo              the field being written
     * @param quantizedValues        quantized codes (by original ordinal) sourced from the SQ flat store
     * @param layer0LocalityOrdering sink carrying the graph-derived forward permutation
     */
    private void writeReorderedLocalityStore(
        final FieldInfo fieldInfo,
        final QuantizedByteVectorValues quantizedValues,
        final Layer0LocalityOrdering layer0LocalityOrdering
    ) throws IOException {
        localityVectorsWriter.writeReorderedLocalityStore(fieldInfo, quantizedValues, layer0LocalityOrdering);
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException(Faiss1040SQHNSWReorderedWriter.class.getSimpleName() + " is already finished");
        }
        finished = true;
        // sqFlatVectorsWriter.finish()/close() already ran in flush/mergeOneField before the native
        // build. The locality store records were written there too, but its CodecUtil footers are
        // written here (finish() = footers); close() then closes its outputs. Without this the
        // .veqlo/.vemlo files have no footer and fail the reader's integrity checks.
        localityVectorsWriter.finish();
    }

    @Override
    public void close() throws IOException {
        // IOUtils.close is null/already-closed safe; sqFlatVectorsWriter is already closed in
        // flush/mergeOneField.
        IOUtils.close(sqFlatVectorsWriter, localityVectorsWriter);
    }

    @Override
    public long ramBytesUsed() {
        // localityVectorsWriter is a thin helper (not Accountable); its footprint is negligible.
        return SHALLOW_SIZE + sqFlatVectorsWriter.ramBytesUsed() + (fieldWriter != null ? fieldWriter.ramBytesUsed() : 0);
    }

    /**
     * Deletes the SQ flat store's quantized sub-files ({@code .veq}/{@code .vemq}) for this field's
     * segment, leaving the raw {@code .vec}/{@code .vemf} intact.
     *
     * <p>Once the reordered locality store ({@code .veqlo}) holds the quantized codes in physical
     * order, the ordinal-order {@code .veq} is pure duplication: search scores against {@code .veqlo},
     * and merge re-quantizes from the raw {@code .vec}, so nothing reads {@code .veq} again. Deleting
     * through the segment's {@code TrackingDirectoryWrapper} also removes these from the tracked
     * created-file set, so they never become part of the segment (not synced, checksummed, or carried
     * into future merges).
     *
     * <p>Must run only after the reader over these files has been closed — open files cannot be deleted
     * on Windows / under {@code MockDirectoryWrapper}. Files are deleted only if present, so the
     * empty-field paths (which never open the reader) are unaffected.
     */
    private void deleteRedundantSqQuantizedFiles() throws IOException {
        final String quantizedData = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            SQ_QUANTIZED_DATA_EXTENSION
        );
        final String quantizedMeta = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            SQ_QUANTIZED_META_EXTENSION
        );
        final List<String> present = Arrays.asList(segmentWriteState.directory.listAll());
        for (final String name : List.of(quantizedData, quantizedMeta)) {
            if (present.contains(name)) {
                segmentWriteState.directory.deleteFile(name);
            } else {
                log.warn("Redundant SQ quantized file [{}] not found; skipping delete (Lucene extension may have changed)", name);
            }
        }
    }

    /**
     * Opens a {@link FlatVectorsReader} scoped to this single field over the already-written SQ flat
     * store, to extract quantized values for the native build and the reordered copy.
     */
    private FlatVectorsReader openSqFlatVectorsReader() throws IOException {
        final SegmentReadState readState = new SegmentReadState(
            segmentWriteState.directory,
            segmentWriteState.segmentInfo,
            new FieldInfos(new FieldInfo[] { fieldInfo }),
            segmentWriteState.context,
            segmentWriteState.segmentSuffix
        );
        return quantizedFlatVectorsReaderSupplier.apply(readState);
    }
}
