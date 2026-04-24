/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.index.clusterann.ClusterANNVectorValues;
import org.opensearch.knn.index.clusterann.ClusteringResult;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.clusterann.algorithm.IVFIndexBuilder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.*;
import org.opensearch.knn.index.clusterann.codec.*;
import org.opensearch.knn.index.clusterann.prefetch.*;

/**
 * Writer for ClusterANN IVF format v2.
 *
 * <p>Two files:
 * <ul>
 *   <li>{@code .clam} — metadata + centroids + offset table</li>
 *   <li>{@code .clap} — per-centroid: [docIds | ordinals | quantized] columnar</li>
 * </ul>
 *
 * <p>Primary + SOAR posting lists are adjacent per centroid for sequential I/O.
 * Centroids written in spatial order (sorted by first principal component).
 */
@Log4j2
public class ClusterANN1040KnnVectorsWriter extends KnnVectorsWriter {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(ClusterANN1040KnnVectorsWriter.class);

    private final SegmentWriteState state;
    private final FlatVectorsWriter flatVectorsWriter;
    private final byte docBits;
    private final List<FieldWriterInfo> fields = new ArrayList<>();

    private final IndexOutput metaOutput;
    private final IndexOutput postingsOutput;

    private static class FieldWriterInfo {
        final FieldInfo fieldInfo;
        final FlatFieldVectorsWriter<float[]> flatFieldWriter;

        FieldWriterInfo(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> flatFieldWriter) {
            this.fieldInfo = fieldInfo;
            this.flatFieldWriter = flatFieldWriter;
        }
    }

    public ClusterANN1040KnnVectorsWriter(SegmentWriteState state, FlatVectorsWriter flatVectorsWriter, int docBits) throws IOException {
        this.state = state;
        this.flatVectorsWriter = flatVectorsWriter;
        this.docBits = validateDocBits(docBits);

        boolean success = false;
        try {
            metaOutput = createOutput(META_EXTENSION);
            postingsOutput = createOutput(POSTINGS_EXTENSION);
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        @SuppressWarnings("unchecked")
        FlatFieldVectorsWriter<float[]> flatFieldWriter = (FlatFieldVectorsWriter<float[]>) flatVectorsWriter.addField(fieldInfo);
        fields.add(new FieldWriterInfo(fieldInfo, flatFieldWriter));
        return flatFieldWriter;
    }

    // ========== Flush ==========

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);
        for (FieldWriterInfo field : fields) {
            List<float[]> vectors = field.flatFieldWriter.getVectors();
            if (vectors.isEmpty()) {
                writeEmptyMeta(field.fieldInfo);
                continue;
            }
            int dimension = field.fieldInfo.getVectorDimension();
            int[] docIds = collectDocIds(field.flatFieldWriter, vectors.size());
            ClusterANNVectorValues vectorValues = ClusterANNVectorValues.fromList(vectors, docIds, dimension);
            writeIVF(field.fieldInfo, vectorValues, null);
        }
    }

    // ========== Merge ==========

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        float[][] reservoir = new float[4096][];
        ClusterANNVectorValues vectors = ClusterANNVectorValues.fromMergeState(mergeState, fieldInfo, reservoir);

        if (vectors.size() == 0) {
            writeEmptyMeta(fieldInfo);
            return;
        }

        int numCentroids = estimateCentroids(vectors.size());
        int actualReservoir = Math.min(vectors.size(), reservoir.length);
        float[][] initialCentroids = Arrays.copyOf(reservoir, Math.min(numCentroids, actualReservoir));

        writeIVF(fieldInfo, vectors, initialCentroids);
    }

    // ========== Shared write path ==========

    private void writeIVF(FieldInfo fieldInfo, ClusterANNVectorValues vectors, float[][] initialCentroids) throws IOException {
        int numVectors = vectors.size();
        int dimension = fieldInfo.getVectorDimension();
        DistanceMetric metric = toDistanceMetric(fieldInfo.getVectorSimilarityFunction());

        // 1. Cluster
        ClusteringResult result = IVFIndexBuilder.build(vectors, TARGET_CLUSTER_SIZE, metric, SOAR_LAMBDA, initialCentroids, 42L, true);

        int numCentroids = result.numCentroids();
        float[][] centroids = result.centroids();

        // 2. Spatial sort centroids (by first principal component)
        int[] spatialOrder = spatialSort(centroids, dimension);
        // Remap: spatialOrder[sortedIdx] = originalIdx

        // 3. Write posting lists to .clap (in spatial order, primary+soar adjacent)
        postingsOutput.alignFilePointer(SECTION_ALIGNMENT);
        long postingsFieldOffset = postingsOutput.getFilePointer();
        long[] centroidOffsets = new long[numCentroids]; // offset per ORIGINAL centroid index

        int[][] primaryPostings = result.primaryPostingLists();
        int[][] soarPostings = result.soarPostingLists();

        ByteBuffer floatBuf = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);

        try (QuantizedVectorWriter qWriter = new QuantizedVectorWriter(fieldInfo.getVectorSimilarityFunction(), dimension, docBits)) {
            for (int si = 0; si < numCentroids; si++) {
                int origIdx = spatialOrder[si];
                centroidOffsets[origIdx] = postingsOutput.getFilePointer();

                // Primary posting list
                writePostingList(primaryPostings[origIdx], vectors, centroids[origIdx], qWriter);

                // SOAR posting list (adjacent)
                writePostingList(soarPostings[origIdx], vectors, centroids[origIdx], qWriter);
            }
        }

        // 4. Write .clam: meta + centroid stats + centroids + offset table
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(numVectors);
        metaOutput.writeInt(dimension);
        metaOutput.writeInt(numCentroids);
        metaOutput.writeString(metric.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeLong(postingsFieldOffset);

        // Centroid doc counts (primary posting list sizes)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeInt(primaryPostings[c].length);
        }

        // Centroid norms (||c||² for fast ADC correction)
        for (int c = 0; c < numCentroids; c++) {
            float norm = 0f;
            for (int d = 0; d < dimension; d++)
                norm += centroids[c][d] * centroids[c][d];
            metaOutput.writeInt(Float.floatToIntBits(norm));
        }

        // Centroids in original order (bulk write via ByteBuffer)
        for (int c = 0; c < numCentroids; c++) {
            floatBuf.clear();
            floatBuf.asFloatBuffer().put(centroids[c]);
            metaOutput.writeBytes(floatBuf.array(), floatBuf.array().length);
        }

        // Offset table (indexed by original centroid index)
        for (int c = 0; c < numCentroids; c++) {
            metaOutput.writeLong(centroidOffsets[c]);
        }
        log.info(
            "[ClusterANN-WRITE] field={} vectors={} centroids={} dim={} clapSize={}",
            fieldInfo.name,
            numVectors,
            numCentroids,
            dimension,
            postingsOutput.getFilePointer()
        );
    }

    /**
     * Write one posting list: [docIds | ordinals | quantized blocks] columnar.
     */
    private void writePostingList(int[] ordinals, ClusterANNVectorValues vectors, float[] centroid, QuantizedVectorWriter qWriter)
        throws IOException {
        int count = ordinals.length;

        // Convert ordinals to docIds
        int[] docIds = new int[count];
        for (int i = 0; i < count; i++) {
            docIds[i] = vectors.ordToDoc(ordinals[i]);
        }

        // Sort by docId, keep ordinals in sync
        sortParallel(docIds, ordinals, count);

        // Write columns
        PostingListCodec.write(docIds, postingsOutput);     // docIds column (sorted, delta-encoded)
        writeRawInts(ordinals, postingsOutput);              // ordinals column (unsorted, raw VInts)

        // Quantized column — block-columnar for SIMD scoring
        qWriter.writeBlocked(ordinals, count, vectors::vectorValue, centroid, postingsOutput);
    }

    // ========== Lifecycle ==========

    @Override
    public void finish() throws IOException {
        flatVectorsWriter.finish();
        metaOutput.writeInt(END_OF_FIELDS);
        CodecUtil.writeFooter(metaOutput);
        CodecUtil.writeFooter(postingsOutput);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter, metaOutput, postingsOutput);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed();
    }

    // ========== Helpers ==========

    /** Sort centroids by first principal component for spatial locality. */
    private static int[] spatialSort(float[][] centroids, int dimension) {
        int n = centroids.length;
        if (n <= 1) return new int[] { 0 };

        // Compute global mean
        float[] mean = new float[dimension];
        for (float[] c : centroids) {
            for (int d = 0; d < dimension; d++)
                mean[d] += c[d];
        }
        for (int d = 0; d < dimension; d++)
            mean[d] /= n;

        // Find axis of max variance (farthest centroid from mean)
        float[] axis = new float[dimension];
        float maxDist = 0;
        for (float[] c : centroids) {
            float dist = 0;
            for (int d = 0; d < dimension; d++) {
                float diff = c[d] - mean[d];
                dist += diff * diff;
            }
            if (dist > maxDist) {
                maxDist = dist;
                System.arraycopy(c, 0, axis, 0, dimension);
                for (int d = 0; d < dimension; d++)
                    axis[d] -= mean[d];
            }
        }

        // Project each centroid onto axis, sort by projection
        long[] packed = new long[n];
        for (int i = 0; i < n; i++) {
            float proj = 0;
            for (int d = 0; d < dimension; d++) {
                proj += centroids[i][d] * axis[d];
            }
            packed[i] = ((long) Float.floatToIntBits(proj) << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);

        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = (int) packed[i];
        }
        return order;
    }

    /** Sort two parallel arrays by the first (keys). */
    /** Write unsorted ints as count + raw VInts. */
    private static void writeRawInts(int[] vals, IndexOutput out) throws IOException {
        out.writeVInt(vals.length);
        for (int v : vals) {
            out.writeVInt(v);
        }
    }

    /** Sort two parallel arrays by the first (keys). */
    private static void sortParallel(int[] keys, int[] vals, int count) {
        if (count <= 1) return;
        long[] packed = new long[count];
        for (int i = 0; i < count; i++) {
            packed[i] = ((long) keys[i] << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        int[] tk = new int[count];
        int[] tv = new int[count];
        for (int i = 0; i < count; i++) {
            int origIdx = (int) packed[i];
            tk[i] = keys[origIdx];
            tv[i] = vals[origIdx];
        }
        System.arraycopy(tk, 0, keys, 0, count);
        System.arraycopy(tv, 0, vals, 0, count);
    }

    private static int estimateCentroids(int numVectors) {
        return Math.max(2, Math.min(4096, (numVectors + 256) / 512));
    }

    private int[] collectDocIds(FlatFieldVectorsWriter<float[]> writer, int numVectors) throws IOException {
        if (writer.getDocsWithFieldSet() == null) return null;
        DocIdSetIterator iter = writer.getDocsWithFieldSet().iterator();
        int[] docIds = new int[numVectors];
        for (int i = 0; i < numVectors; i++) {
            docIds[i] = iter.nextDoc();
        }
        return docIds;
    }

    private void writeEmptyMeta(FieldInfo fieldInfo) throws IOException {
        metaOutput.writeInt(fieldInfo.number);
        metaOutput.writeInt(0);
        metaOutput.writeInt(0);
        metaOutput.writeInt(0);
        metaOutput.writeString(DistanceMetric.L2.name());
        metaOutput.writeByte(docBits);
        metaOutput.writeLong(0);
    }

    private IndexOutput createOutput(String extension) throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        IndexOutput output = state.directory.createOutput(fileName, state.context);
        CodecUtil.writeIndexHeader(output, CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
        return output;
    }

    private static byte validateDocBits(int bits) {
        if (bits != 1 && bits != 2 && bits != 4) {
            throw new IllegalArgumentException("docBits must be 1, 2, or 4, got: " + bits);
        }
        return (byte) bits;
    }

    private static DistanceMetric toDistanceMetric(VectorSimilarityFunction simFunc) {
        return switch (simFunc) {
            case EUCLIDEAN -> DistanceMetric.L2;
            case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> DistanceMetric.INNER_PRODUCT;
            case COSINE -> DistanceMetric.COSINE;
        };
    }
}
