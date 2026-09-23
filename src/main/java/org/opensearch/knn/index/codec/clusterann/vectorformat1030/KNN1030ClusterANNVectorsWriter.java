/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.write.CentroidsWriter;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidData;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidOffsets;
import org.opensearch.knn.clusterann.format.rotation.Rotation;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.format.rotation.RotationScheme;
import org.opensearch.knn.clusterann.write.postings.ClusterAnnPostingsWriter;
import org.opensearch.knn.clusterann.write.postings.ClusterWriterFactory;
import org.opensearch.knn.clusterann.write.postings.PostingsRegions;
import org.opensearch.knn.clusterann.ClusteringResult;
import org.opensearch.knn.clusterann.clustering.ClusterBuilder;

import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.CENTROIDS_CODEC_NAME;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.CENTROIDS_EXTENSION;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.DEFAULT_BLOCK_SIZE;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.NO_MORE_FIELDS;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.POSTINGS_CODEC_NAME;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.POSTINGS_EXTENSION;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.ROTATION_CODEC_NAME;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.ROTATION_EXTENSION;
import static org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsFormat.VERSION_CURRENT;

/**
 * Per-segment {@link KnnVectorsWriter} for {@link KNN1030ClusterANNVectorsFormat}, mirroring
 * {@code Lucene99HnswVectorsWriter}: the wrapped {@link FlatVectorsWriter} stores the raw vectors
 * ({@code .vec}/{@code .vemf}); {@link #addField} keeps each flat field writer so {@link #flush} can read the
 * buffered vectors back and, per field, cluster them and write the ClusterANN files — {@code .clar} (rotation,
 * EUCLIDEAN only), {@code .clap} (postings), {@code .clac} (centroids), and {@code .clam} (per-field metadata
 * that points into the other three). The four outputs are opened and headed in the constructor and footed in
 * {@link #finish}; each field's {@code .clam} entry is written after its regions, since it records where they
 * landed.
 *
 * <p>On merge, {@link #mergeOneField} merges the raw vectors through the flat delegate (one write to the flat
 * file), reopens a flat reader over them, and clusters from that — so the flat vectors are written once, with
 * no temp copy. It assumes one vector field per segment (reopening the reader closes the flat writer).
 *
 * @see KNN1030ClusterANNVectorsFormat
 * @see KNN1030ClusterANNVectorsReader
 */
@Log4j2
public class KNN1030ClusterANNVectorsWriter extends KnnVectorsWriter {

    /** Sentinel {@code .clam} records for a rotation offset/length that an unrotated field does not have. */
    private static final long NO_ROTATION = -1L;

    private final SegmentWriteState segmentWriteState;
    private final QuantizationParams quantizationParams;
    private final FlatVectorsFormat rawFlatVectorsFormat;
    private final FlatVectorsWriter rawFlatVectorsWriter;
    private final List<FieldEntry> fields = new ArrayList<>();
    private final IndexOutput clam;
    private final IndexOutput clac;
    private final IndexOutput clap;
    /** Opened on the first rotated field; stays {@code null} for an all-unrotated segment, which writes no {@code .clar}. */
    private IndexOutput clar;
    /** On merge, the reopened reader over the just-written flat vectors; {@code null} on the flush path. */
    private FlatVectorsReader flatReader;
    /** Set once the flat writer has been finished+closed (the merge path does this early to reopen the reader). */
    private boolean flatWriterClosed;
    private boolean finished;
    private boolean closed;

    /**
     * Creates a writer for a single segment and opens its cluster files, writing each one's index header and
     * the shared block size into {@code .clam}; field entries and footers follow in {@link #flush}/{@link
     * #finish}.
     *
     * @param segmentWriteState    state describing the segment being written
     * @param rawFlatVectorsFormat delegate flat-vectors format; supplies the writer that persists the raw
     *     vectors and, on merge, the reader reopened over them
     * @param quantizationParams         the encoding (backend + bit width) to quantize each field with
     */
    public KNN1030ClusterANNVectorsWriter(
        final SegmentWriteState segmentWriteState,
        final FlatVectorsFormat rawFlatVectorsFormat,
        final QuantizationParams quantizationParams
    ) throws IOException {
        this.rawFlatVectorsFormat = rawFlatVectorsFormat;
        this.quantizationParams = quantizationParams;
        this.segmentWriteState = segmentWriteState;
        boolean success = false;
        try {
            this.rawFlatVectorsWriter = rawFlatVectorsFormat.fieldsWriter(segmentWriteState);
            clam = openOutput(META_EXTENSION, META_CODEC_NAME);
            clac = openOutput(CENTROIDS_EXTENSION, CENTROIDS_CODEC_NAME);
            clap = openOutput(POSTINGS_EXTENSION, POSTINGS_CODEC_NAME);
            // .clar is opened lazily on the first rotated field, so an all-unrotated segment writes none.
            clam.writeVInt(DEFAULT_BLOCK_SIZE);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    /**
     * Registers a vector field: delegates raw storage to the flat writer and remembers the flat field writer so
     * {@link #flush} can read the buffered vectors back and cluster them.
     */
    @Override
    @SuppressWarnings("unchecked")   // guarded above: FLOAT32 fields are float[]
    public KnnFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        ensureSingleVectorField(fields.size() + 1);
        ensureFloat32(fieldInfo);
        ensureSupportedSimilarity(fieldInfo);
        final FlatFieldVectorsWriter<float[]> flat = (FlatFieldVectorsWriter<float[]>) rawFlatVectorsWriter.addField(fieldInfo);
        fields.add(new FieldEntry(fieldInfo, flat));
        return flat;
    }

    /**
     * Flushes the buffered vectors: raw vectors go to the flat delegate, then each field is clustered and
     * written from the flat writer's in-memory buffer.
     *
     * @param maxDoc the number of documents in the segment
     * @param docMap old&rarr;new doc id mapping when the segment is index-sorted, or {@code null}
     * @throws IOException if writing fails
     */
    @Override
    public void flush(final int maxDoc, final Sorter.DocMap docMap) throws IOException {
        log.debug(
            "flush segment [{}]: maxDoc={} fields={} sorted={}",
            segmentWriteState.segmentInfo.name,
            maxDoc,
            fields.size(),
            docMap != null
        );
        ensureSingleVectorField(fields.size());
        rawFlatVectorsWriter.flush(maxDoc, docMap);
        for (final FieldEntry field : fields) {
            final FieldVectors source = source(field, docMap);
            writeField(field.fieldInfo, source.vectors, source.docsWithField, maxDoc, null); // flush clusters inline
        }
        log.debug("flush segment [{}]: wrote {} field(s)", segmentWriteState.segmentInfo.name, fields.size());
    }

    /**
     * Merges one field: merges its raw vectors through the flat delegate (a single write to the flat file), then
     * reads them back through a reopened flat reader to cluster and write the ClusterANN files — so the flat
     * vectors are written once, with no temp copy for scoring.
     *
     * <p>Assumes one vector field per segment: reopening the reader finishes and closes the flat writer, so a
     * second field would have no open writer to merge into and is rejected.
     *
     * @param fieldInfo  metadata for the field being merged
     * @param mergeState state of the merge
     * @throws IOException if merging, reopening, or writing fails
     */
    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // Reopening the flat reader (below) finishes+closes the flat writer, so this writer can cluster only one
        // field per segment. Detect a multi-vector-field segment up front and fail before doing any work.
        ensureSingleVectorField(vectorFieldCount());
        // Merge the raw vectors once to the flat file, then reopen it for random access — the merged MergeState
        // view is forward-only, but clustering and quantization address vectors by ordinal.
        rawFlatVectorsWriter.mergeOneField(fieldInfo, mergeState);
        ensureFlatReaderOpen();

        final FloatVectorValues vectors = flatReader.getFloatVectorValues(fieldInfo.name);
        // Merge is the heavy path — cluster in parallel on the merge's executor (flush clusters inline).
        writeField(
            fieldInfo,
            vectors,
            docsWithField(vectors),
            mergeState.segmentInfo.maxDoc(),
            new TaskExecutor(mergeState.intraMergeTaskExecutor)
        );
        log.debug("merge segment [{}]: clustered field [{}]", segmentWriteState.segmentInfo.name, fieldInfo.name);
    }

    /**
     * Finishes and closes the flat writer, then reopens a flat reader over the just-written vectors — done once,
     * lazily, so the merged vectors can be read back by ordinal. After this the flat writer is closed
     * ({@link #flatWriterClosed}), so {@link #finish}/{@link #close} must not finish or close it again.
     */
    private void ensureFlatReaderOpen() throws IOException {
        if (flatReader != null) {
            return;
        }
        rawFlatVectorsWriter.finish();
        rawFlatVectorsWriter.close();
        flatWriterClosed = true;
        final SegmentReadState readState = new SegmentReadState(
            segmentWriteState.directory,
            segmentWriteState.segmentInfo,
            segmentWriteState.fieldInfos,
            segmentWriteState.context,
            segmentWriteState.segmentSuffix
        );
        flatReader = rawFlatVectorsFormat.fieldsReader(readState);
    }

    /** The docs that carry this field, in ordinal order, rebuilt by walking the merged vectors' iterator. */
    private static DocsWithFieldSet docsWithField(final FloatVectorValues vectors) throws IOException {
        final DocsWithFieldSet docs = new DocsWithFieldSet();
        final KnnVectorValues.DocIndexIterator it = vectors.iterator();
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            docs.add(doc);
        }
        return docs;
    }

    /**
     * Rejects a segment with more than one vector field: ClusterANN clusters at most one field per segment, so
     * flush, merge, and {@link #addField} all fail fast rather than deferring the error to a later write path.
     */
    private static void ensureSingleVectorField(final int vectorFields) {
        if (vectorFields > 1) {
            throw new UnsupportedOperationException(
                "ClusterANN supports a single vector field per segment, but the segment has " + vectorFields
            );
        }
    }

    /**
     * Rejects a non-{@code FLOAT32} field up front: ClusterANN clusters and quantizes {@code float[]} vectors, so
     * {@link #addField} casts the flat writer to {@code FlatFieldVectorsWriter<float[]>}. Failing here turns what
     * would otherwise be an obscure {@code ClassCastException} at flush into a clear error at field registration.
     */
    private static void ensureFloat32(final FieldInfo fieldInfo) {
        if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new UnsupportedOperationException(
                "ClusterANN supports only FLOAT32 vectors, but field [" + fieldInfo.name + "] has encoding " + fieldInfo.getVectorEncoding()
            );
        }
    }

    private static void ensureSupportedSimilarity(final FieldInfo fieldInfo) {
        if (fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException(
                "ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT. Field [" + fieldInfo.name + "]"
            );
        }
    }

    /** How many of the (merged) segment's fields carry vector values — the merge path clusters at most one. */
    private int vectorFieldCount() {
        int count = 0;
        for (final FieldInfo fieldInfo : segmentWriteState.fieldInfos) {
            if (fieldInfo.hasVectorValues()) {
                count++;
            }
        }
        return count;
    }

    /**
     * Finishes writing: the flat delegate first, then the {@code .clam} end-of-fields terminator and a footer on
     * every cluster file.
     *
     * @throws IOException if finishing or writing fails
     */
    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;

        // On merge the flat writer was already finished+closed to reopen the reader; only the flush path finishes it here.
        if (!flatWriterClosed) {
            rawFlatVectorsWriter.finish();
        }

        clam.writeInt(NO_MORE_FIELDS);
        CodecUtil.writeFooter(clam);
        CodecUtil.writeFooter(clac);
        CodecUtil.writeFooter(clap);
        if (clar != null) {
            CodecUtil.writeFooter(clar);
        }
        log.debug("finish segment [{}]: terminated .clam and footered all cluster files", segmentWriteState.segmentInfo.name);
    }

    /**
     * Writes one field's ClusterANN data from {@code vectors} (in ClusterANN ordinal order), in order:
     * <ol>
     *   <li>an empty field gets a zeroed {@code .clam} entry and nothing else;</li>
     *   <li>choose the rotation — EUCLIDEAN is rotated, inner product / cosine are not;</li>
     *   <li>cluster the vectors;</li>
     *   <li>write {@code .clar}: the rotation matrix (nothing for an unrotated field);</li>
     *   <li>write {@code .clap}: the per-cluster quantized postings;</li>
     *   <li>write {@code .clac}: the centroids and their geometry (rotated centroids only when the field is rotated);</li>
     *   <li>append the {@code .clam} entry that records the field's shape and where the above regions landed.</li>
     * </ol>
     */
    private void writeField(
        final FieldInfo fieldInfo,
        final FloatVectorValues vectors,
        final DocsWithFieldSet docsWithField,
        final int maxDoc,
        final TaskExecutor executor
    ) throws IOException {
        final int dimension = fieldInfo.getVectorDimension();
        final VectorSimilarityFunction metric = fieldInfo.getVectorSimilarityFunction();
        final String name = fieldInfo.name;
        log.debug("field [{}]: read {} vectors (dim={}, metric={})", name, vectors.size(), dimension, metric);

        clam.writeInt(fieldInfo.number); // the field number precedes every .clam entry, empty or not

        // The backend's on-disk .clam code; the write side owns the enum->id mapping, not the upstream enum.
        final int quantizerId = ClusterWriterFactory.quantizerId(quantizationParams.encoding());

        if (vectors.size() == 0) {
            ClusterANNFieldMeta.empty(
                DEFAULT_BLOCK_SIZE,
                dimension,
                metric,
                quantizationParams.docBits(),
                quantizerId,
                clac.getFilePointer(),
                clap.getFilePointer()
            ).write(clam, clap, maxDoc, docsWithField);
            log.debug("field [{}]: empty, wrote zeroed .clam entry", name);
            return;
        }

        final RotationScheme scheme = RotationScheme.forMetric(metric);
        final int rotationId = scheme.code();
        final Rotation rotation = RotationFormats.create(rotationId, dimension);
        final Optional<RotationRegion> rotationRegion = maybeWriteRotation(scheme, rotation);
        log.debug("field [{}]: rotationId={}", name, rotationId);

        final ClusteringResult clusters = ClusterBuilder.build(vectors, metric, executor);
        log.debug("field [{}]: clustered into {} centroid(s)", name, clusters.numCentroids());

        final PostingsRegions postings = new ClusterAnnPostingsWriter(DEFAULT_BLOCK_SIZE, quantizationParams, rotation).write(
            clap,
            clusters,
            vectors,
            metric
        );

        final CentroidOffsets centroids = writeCentroids(clusters, rotation);
        final long clacLength = clac.getFilePointer() - centroids.clacOffset();

        final ClusterANNFieldMeta meta = new ClusterANNFieldMeta(
            DEFAULT_BLOCK_SIZE,
            dimension,
            vectors.size(),
            clusters.numCentroids(),
            metric,
            quantizationParams.docBits(),
            rotationId,
            quantizerId,
            new byte[0],
            centroids.clacOffset(),
            clacLength,
            centroids.clacCentroidsOffset(),
            centroids.clacRotatedCentroidsOffset(),
            postings.clapOffset(),
            postings.clapLength(),
            postings.centroidOffsets(),
            postings.centroidLengths(),
            postings.clusterSizes(),
            rotationRegion.map(RotationRegion::offset).orElse(NO_ROTATION),
            rotationRegion.map(RotationRegion::length).orElse(NO_ROTATION),
            null
        );
        meta.write(clam, clap, maxDoc, docsWithField);
        log.debug("field [{}]: wrote .clam entry", name);
    }

    /**
     * For a rotated field, writes its matrix to {@code .clar} (opening it on first use) and returns the region.
     * EUCLIDEAN is rotated (a random Gaussian rotation, so 1-bit codes stay accurate under L2); inner product and
     * cosine are not. An unrotated field touches no {@code .clar} (so an all-unrotated segment writes none) and
     * returns {@link Optional#empty()}, which {@code .clam} records as {@link #NO_ROTATION} for {@code ROTATION_NONE}.
     */
    private Optional<RotationRegion> maybeWriteRotation(final RotationScheme scheme, final Rotation rotation) throws IOException {
        if (!scheme.rotates()) {
            return Optional.empty();
        }
        final IndexOutput rotationOut = rotationOutput();
        final long offset = rotationOut.getFilePointer();
        final long length = RotationFormats.write(scheme.code(), rotationOut, rotation);
        return Optional.of(new RotationRegion(offset, length));
    }

    /**
     * Writes the field's centroids to {@code .clac} and returns the {@link CentroidOffsets} {@code CentroidsWriter}
     * produced: the region start plus region-relative sub-offsets, the rotated-centroids one already
     * {@link #NO_ROTATION} for an unrotated field. The region's byte length is derived at the call site from the
     * {@code .clac} file pointer.
     */
    private CentroidOffsets writeCentroids(final ClusteringResult clusters, final Rotation rotation) throws IOException {
        final CentroidData centroidData = new CentroidData(clusters.assignments(), clusters.centroids());
        return CentroidsWriter.write(clac, centroidData, rotation);
    }

    /** A rotated field's {@code .clar} region: absolute offset and byte length. Absent for an unrotated field. */
    private record RotationRegion(long offset, long length) {
    }

    /**
     * The field's vectors and doc set in ClusterANN ordinal order. Unsorted is the buffered arrival order the
     * flat writer holds; an index-sorted segment is gathered through {@link KnnVectorsWriter#mapOldOrdToNewOrd}
     * into the flat file's sorted order, so the ordinals match.
     */
    private static FieldVectors source(final FieldEntry field, final Sorter.DocMap docMap) throws IOException {
        final int dimension = field.fieldInfo.getVectorDimension();
        final List<float[]> buffered = field.flat.getVectors();
        if (docMap == null) {
            return new FieldVectors(FloatVectorValues.fromFloats(buffered, dimension), field.flat.getDocsWithFieldSet());
        }

        final int[] newToOldOrd = new int[buffered.size()];
        final DocsWithFieldSet sortedDocs = new DocsWithFieldSet();
        mapOldOrdToNewOrd(field.flat.getDocsWithFieldSet(), docMap, null, newToOldOrd, sortedDocs);

        // Reindex the buffered vectors into sorted-ordinal order as a view, without copying the vectors.
        final List<float[]> sorted = new AbstractList<>() {
            @Override
            public float[] get(int newOrd) {
                return buffered.get(newToOldOrd[newOrd]);
            }

            @Override
            public int size() {
                return newToOldOrd.length;
            }
        };
        return new FieldVectors(FloatVectorValues.fromFloats(sorted, dimension), sortedDocs);
    }

    /** The {@code .clar} output, created and headed on first use so an all-unrotated segment never opens one. */
    private IndexOutput rotationOutput() throws IOException {
        if (clar == null) {
            clar = openOutput(ROTATION_EXTENSION, ROTATION_CODEC_NAME);
        }
        return clar;
    }

    /** Creates a cluster file and writes its {@code CodecUtil} index header. */
    private IndexOutput openOutput(final String extension, final String codecName) throws IOException {
        final String fileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            extension
        );
        final IndexOutput out = segmentWriteState.directory.createOutput(fileName, segmentWriteState.context);
        CodecUtil.writeIndexHeader(out, codecName, VERSION_CURRENT, segmentWriteState.segmentInfo.getId(), segmentWriteState.segmentSuffix);
        return out;
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        if (flatWriterClosed) {
            // The merge path already finished+closed the flat writer to reopen the reader; close the reader here.
            IOUtils.close(flatReader, clam, clac, clap, clar);
        } else {
            IOUtils.close(rawFlatVectorsWriter, clam, clac, clap, clar);
        }
    }

    @Override
    public long ramBytesUsed() {
        return flatWriterClosed ? 0L : rawFlatVectorsWriter.ramBytesUsed();
    }

    /** A field added to this segment: its info and the flat writer holding its buffered vectors. */
    private record FieldEntry(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> flat) {
    }

    /** A field's vectors and doc set in ClusterANN ordinal order. */
    private record FieldVectors(FloatVectorValues vectors, DocsWithFieldSet docsWithField) {
    }
}
