/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import lombok.NonNull;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.packed.DirectWriter;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.knn.index.codec.nativeindex.model.Layer0LocalityOrdering;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
 *    numHubs               (vInt)  search entry-point candidates (JNI -1-padding dropped)
 *    hubOrdinals           (numHubs x vInt)     original ordinals, highest-degree first
 *    hubRecords            (numHubs x recordSize) each hub's quantized record (code + 4 corrections)
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
public final class LocalityOrderedQuantizedVectorsWriter implements Closeable {

    public static final String CODEC_NAME = "LocalityOrderedQuantizedVectors";
    public static final String EXTENSION = "veqlo";
    public static final String METADATA_EXTENSION = "vemlo";
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;

    /** 3 correction floats + 1 correction int, all 4 bytes. */
    static final int CORRECTION_BYTES = Integer.BYTES * 4;

    private final IndexOutput vectordata;
    private final IndexOutput metadata;
    private boolean finished;
    private final QuantizedByteVectorValues.ScalarEncoding scalarEncoding;

    public LocalityOrderedQuantizedVectorsWriter(
        final SegmentWriteState segmentWriteState,
        QuantizedByteVectorValues.ScalarEncoding scalarEncoding
    ) throws IOException {
        this.scalarEncoding = scalarEncoding;

        final String vectorFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            EXTENSION
        );
        final String metaFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            METADATA_EXTENSION
        );
        boolean success = false;
        IndexOutput vectordata = null;
        IndexOutput metadata = null;
        try {
            vectordata = segmentWriteState.directory.createOutput(vectorFileName, segmentWriteState.context);
            metadata = segmentWriteState.directory.createOutput(metaFileName, segmentWriteState.context);
            CodecUtil.writeIndexHeader(
                vectordata,
                CODEC_NAME,
                VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );
            CodecUtil.writeIndexHeader(
                metadata,
                CODEC_NAME,
                VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );
            success = true;
        } finally {
            if (!success && vectordata != null) {
                IOUtils.closeWhileHandlingException(vectordata, metadata);
            }
        }
        this.vectordata = vectordata;
        this.metadata = metadata;
    }

    /**
     * Writes the locality store ({@value #METADATA_EXTENSION} metadata + {@value #EXTENSION} records)
     * for a single field by <b>copying an already-quantized source in a physical (permuted) order</b> —
     * no re-quantization. This is the entry point the Faiss HNSW reordered build uses after the graph
     * is constructed: the source {@code quantizedByteVectorValues} are the codes read back from the SQ
     * flat store, and the records are re-emitted here in locality order.
     *
     * <p>Steps: read the graph-derived permutation from {@code layer0LocalityOrdering} (the FORWARD map
     * {@code ordinalToPhysicalOrdinal[originalOrdinal] = physicalPosition}), invert it into
     * {@code physicalOrdinals[physicalPosition] = originalOrdinal}, write the field metadata (scalars,
     * {@code centroid} taken from the source, and the {@code originalOrdinal -> physicalPosition} map),
     * then copy each source record into its physical slot via
     * {@link #writeVectorDataUsingQuantizedVectorValues} (the source is {@link QuantizedByteVectorValues#copy()
     * copied} so record iteration does not disturb the caller's instance).
     *
     * <p><b>LIMITATION — this currently only handles a DENSE permutation, i.e. the BFS ordering
     * ({@code FaissService.buildOrderingOfVectorsUsingBFS}), NOT the greedy page-aligned ordering
     * ({@code FaissService.buildOrderingOfVectorsUsingIndexStructure}).</b> The inversion below assumes
     * physical positions are a contiguous {@code 0..N-1} (one record per vector, no gaps), which holds
     * for BFS. The greedy strict-page layout instead pads each page to a {@code pageCapacity} boundary,
     * so its physical positions are <b>sparse</b> — values can exceed {@code N-1} and some physical
     * slots map to no vector (padding). Feeding a greedy permutation here would (a) index the
     * {@code N}-sized {@code physicalOrdinals} array out of bounds, and (b) leave no notion of the
     * zero-filled padding slots. Supporting greedy requires sizing the inverse to
     * {@code numPages*pageCapacity}, marking padding slots with a sentinel, emitting zero-filled records
     * for them, and separating the record count (physical slots) from the vector count in
     * {@link #writeMetadata}. Until then, wire the BFS ordering into the build strategy for this path.
     *
     * <p>Only a single field per file is supported (see {@link #writeMetadata}); the caller must not
     * also drive this writer via the flat {@code addField}/{@code flush} path for the same file.
     *
     * @param fieldInfo                  the field being written
     * @param quantizedByteVectorValues  source quantized codes (addressed by original ordinal) plus the
     *                                   centroid; consumed as-is, not re-quantized
     * @param layer0LocalityOrdering     holds the forward permutation {@code [originalOrdinal] =
     *                                   physicalPosition} produced by the native build; must be a DENSE
     *                                   (BFS) permutation — see the limitation above
     */
    public void writeReorderedLocalityStore(
        @NonNull final FieldInfo fieldInfo,
        @NonNull final QuantizedByteVectorValues quantizedByteVectorValues,
        @NonNull final Layer0LocalityOrdering layer0LocalityOrdering
    ) throws IOException {

        final int[] physicalOrdinalsArray = new int[layer0LocalityOrdering.getPhysicalOrdinals().length];
        final int[] ordinalToPhysicalOrdinal = layer0LocalityOrdering.getPhysicalOrdinals();

        // Invert the forward map into physicalOrdinals[physicalPos] = originalOrdinal.
        // DENSE / BFS ONLY: this array is sized to N and assumes every physicalPosition is in [0, N)
        // with no gaps. That holds for the BFS ordering. It does NOT hold for the greedy page-aligned
        // ordering, whose physical positions are sparse (padding between pages) and can exceed N-1 —
        // ordinalToPhysicalOrdinal[i] would then index this array out of bounds, and padding slots
        // would have no record written. See this method's Javadoc for what greedy support needs.
        for (int i = 0; i < ordinalToPhysicalOrdinal.length; i++) {
            physicalOrdinalsArray[ordinalToPhysicalOrdinal[i]] = i;
        }

        writeMetadata(fieldInfo, quantizedByteVectorValues.getCentroid(), physicalOrdinalsArray);

        // Hub entry-point candidates go at the very end of the metadata file (after the
        // ordToPhysicalOrdMap DirectWriter block); see writeHubs.
        writeHubs(layer0LocalityOrdering.getHubs(), quantizedByteVectorValues.copy());

        writeVectorDataUsingQuantizedVectorValues(physicalOrdinalsArray, quantizedByteVectorValues.copy());
    }

    /**
     * Appends the hub section to the end of the metadata ({@value #METADATA_EXTENSION}) file — the
     * search entry-point candidates (highest-degree graph nodes). Layout, immediately after the
     * {@code ordToPhysicalOrdMap} DirectWriter block and before the footer:
     * <pre>
     *   numHubs        (vInt)                      valid hubs (JNI -1-padding dropped)
     *   hubOrdinals    (numHubs x vInt)            original ordinals, highest-degree first
     *   hubRecords     (numHubs x recordSize)      each hub's quantized record: 1-bit code + 4 corrections
     * </pre>
     *
     * <p>Hub records are the <b>quantized</b> codes (not raw float), so entry-point selection scores the
     * query against them with the same ADC path used for the rest of the graph. They are copied from the
     * source at each hub's original ordinal, matching the on-disk record layout in
     * {@link #writeVectorDataUsingQuantizedVectorValues}.
     *
     * @param hubs   hub ordinals from the native build ({@code hubs[rank] = originalOrdinal}), possibly
     *               {@code -1}-padded
     * @param source quantized codes addressed by original ordinal; consumed by random access
     */
    private void writeHubs(final int[] hubs, final QuantizedByteVectorValues source) throws IOException {
        int numHubs = 0;
        for (final int hub : hubs) {
            if (hub >= 0) {
                numHubs++;
            }
        }
        metadata.writeVInt(numHubs);
        // Ordinals first (highest-degree first), then the matching quantized records.
        for (final int hub : hubs) {
            if (hub >= 0) {
                metadata.writeVInt(hub);
            }
        }
        for (final int hub : hubs) {
            if (hub < 0) {
                continue;
            }
            final byte[] code = source.vectorValue(hub);
            metadata.writeBytes(code, code.length);
            final OptimizedScalarQuantizer.QuantizationResult qr = source.getCorrectiveTerms(hub);
            metadata.writeInt(Float.floatToIntBits(qr.lowerInterval()));
            metadata.writeInt(Float.floatToIntBits(qr.upperInterval()));
            metadata.writeInt(Float.floatToIntBits(qr.additionalCorrection()));
            metadata.writeInt(qr.quantizedComponentSum());
        }
    }

    /**
     * Writes one field's metadata to the {@code .vemlo} file: the scalar header, the centroid, and
     * the {@code originalOrdinal -> physicalPosition} map.
     */
    private void writeMetadata(final FieldInfo fieldInfo, final float[] centroid, final int[] physicalOrdinals) throws IOException {
        final int count = physicalOrdinals.length;
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
            ordToPhysicalOrdMap[physicalOrdinals[i]] = i;
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

    private void writeVectorDataUsingQuantizedVectorValues(
        final int[] physicalOrdinals,
        final QuantizedByteVectorValues quantizedByteVectorValues
    ) throws IOException {
        final int count = physicalOrdinals.length;
        for (int p = 0; p < count; p++) {
            // get quantized vector using the original ordinal.
            byte[] vectorToBeWritten = quantizedByteVectorValues.vectorValue(physicalOrdinals[p]);
            vectordata.writeBytes(vectorToBeWritten, vectorToBeWritten.length);
            final OptimizedScalarQuantizer.QuantizationResult quantizationResult = quantizedByteVectorValues.getCorrectiveTerms(
                physicalOrdinals[p]
            );
            vectordata.writeInt(Float.floatToIntBits(quantizationResult.lowerInterval()));
            vectordata.writeInt(Float.floatToIntBits(quantizationResult.upperInterval()));
            vectordata.writeInt(Float.floatToIntBits(quantizationResult.additionalCorrection()));
            vectordata.writeInt(quantizationResult.quantizedComponentSum());
        }
    }

    /** Writes the footer.*/
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        CodecUtil.writeFooter(metadata);
        CodecUtil.writeFooter(vectordata);
    }

    /** Closes the metadata and records outputs. {@link IOUtils#close} is null-safe. */
    @Override
    public void close() throws IOException {
        IOUtils.close(metadata, vectordata);
    }
}
