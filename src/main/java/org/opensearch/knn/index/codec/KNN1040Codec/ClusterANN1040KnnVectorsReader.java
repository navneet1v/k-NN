/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.clusterann.DistanceMetric;
import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.index.warmup.WarmableReader;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Reader for the ClusterANN IVF format. Performs cluster-based approximate nearest neighbor
 * search by probing a subset of centroids and scanning their posting lists.
 *
 * <p>Search flow:
 * <ol>
 *   <li>Find nprobe nearest centroids to the query</li>
 *   <li>For each probed centroid, read primary + SOAR posting lists</li>
 *   <li>Score candidates using {@link ClusterANNScorer} (exact or two-phase ADC)</li>
 *   <li>Collect results into {@link KnnCollector}</li>
 * </ol>
 *
 * <p>Scoring is delegated to {@link ClusterANNScorer} implementations:
 * <ul>
 *   <li>{@link ExactClusterANNScorer} — full-precision via Lucene's RandomVectorScorer</li>
 *   <li>{@link TwoPhaseClusterANNScorer} — ADC first pass + exact rescore</li>
 * </ul>
 */
@Log4j2
public class ClusterANN1040KnnVectorsReader extends KnnVectorsReader implements WarmableReader {

    private final FlatVectorsReader flatVectorsReader;
    private final Map<Integer, ClusterANNFieldState> fieldStates;
    private final Map<String, Integer> fieldNameToNumber;
    private final FieldInfos fieldInfos;

    // Index file inputs (kept open for the lifetime of the reader)
    private final IndexInput centroidsInput;
    private final IndexInput postingsInput;
    private final IndexInput quantizedInput;

    public ClusterANN1040KnnVectorsReader(FlatVectorsReader flatVectorsReader, SegmentReadState state) throws IOException {
        this.flatVectorsReader = flatVectorsReader;

        boolean success = false;
        IndexInput metaIn = null;
        IndexInput centIn = null;
        IndexInput postIn = null;
        IndexInput quantIn = null;
        try {
            metaIn = openInput(state, ClusterANN1040KnnVectorsWriter.META_EXTENSION);
            centIn = openInput(state, ClusterANN1040KnnVectorsWriter.CENTROIDS_EXTENSION);
            postIn = openInput(state, ClusterANN1040KnnVectorsWriter.POSTINGS_EXTENSION);
            quantIn = openInput(state, ClusterANN1040KnnVectorsWriter.QUANTIZED_EXTENSION);

            this.fieldStates = ClusterANNFieldState.readAll(metaIn, state);

            // Build field name → number mapping
            this.fieldNameToNumber = new HashMap<>();
            for (FieldInfo fi : state.fieldInfos) {
                if (fieldStates.containsKey(fi.number)) {
                    fieldNameToNumber.put(fi.getName(), fi.number);
                }
            }

            this.centroidsInput = centIn;
            this.postingsInput = postIn;
            this.quantizedInput = quantIn;
            this.fieldInfos = state.fieldInfos;
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(metaIn, centIn, postIn, quantIn, flatVectorsReader);
            } else if (metaIn != null) {
                metaIn.close(); // meta fully read, no longer needed
            }
        }

        log.info("[ClusterANN] reader created: {} fields with IVF index", fieldStates.size());
    }

    @Override
    public void checkIntegrity() throws IOException {
        flatVectorsReader.checkIntegrity();
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return flatVectorsReader.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return flatVectorsReader.getByteVectorValues(field);
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        Integer fieldNumber = fieldNameToNumber.get(field);
        ClusterANNFieldState fieldState = fieldNumber != null ? fieldStates.get(fieldNumber) : null;

        if (fieldState == null || fieldState.isEmpty()) {
            // No IVF index — fall back to brute force
            bruteForceSearch(field, target, knnCollector, acceptDocs);
            return;
        }

        // Ensure centroids and posting offsets are loaded
        fieldState.ensureCentroidsLoaded(centroidsInput);
        fieldState.ensurePostingOffsetsLoaded(postingsInput);

        int k = knnCollector.k();
        int nprobe = calculateNprobe(fieldState.numCentroids, fieldState.numVectors, k);
        int[] nearestCentroids = findNearestCentroids(target, fieldState.centroids,
            fieldState.numCentroids, fieldState.dimension, fieldState.metric, nprobe);

        // Create scorer
        RandomVectorScorer exactScorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (exactScorer == null) return;

        VectorSimilarityFunction simFunc = getSimFunc(field);
        boolean useADC = fieldState.docBits > 0 && quantizedInput != null;

        ClusterANNScorer scorer;
        TwoPhaseClusterANNScorer twoPhase = null;
        if (useADC) {
            twoPhase = new TwoPhaseClusterANNScorer(
                exactScorer, quantizedInput, fieldState, simFunc, target, k);
            scorer = twoPhase;
        } else {
            scorer = new ExactClusterANNScorer(exactScorer);
        }

        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        Set<Integer> visited = new HashSet<>();

        // Prefetch posting list data for probed centroids
        prefetchPostingLists(fieldState, nearestCentroids, nprobe);

        // Scan posting lists for probed centroids using direct seek
        for (int i = 0; i < nprobe; i++) {
            int centId = nearestCentroids[i];

            // Primary posting list
            postingsInput.seek(fieldState.primaryPostingOffsets[centId]);
            int primarySize = postingsInput.readVInt();
            scanPostingList(primarySize, scorer, twoPhase, fieldState, centId, target,
                acceptBits, visited, knnCollector, useADC);

            // SOAR posting list
            postingsInput.seek(fieldState.soarPostingOffsets[centId]);
            int soarSize = postingsInput.readVInt();
            scanPostingList(soarSize, scorer, twoPhase, fieldState, centId, target,
                acceptBits, visited, knnCollector, useADC);
        }

        // Finish: for two-phase, rescore top ADC candidates
        scorer.finish((docId, score) -> {
            knnCollector.collect(docId, score);
        });
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        // Byte vector search: delegate to brute force via flat reader
        RandomVectorScorer byteScorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (byteScorer == null) return;
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        for (int ord = 0; ord < byteScorer.maxOrd(); ord++) {
            int docId = byteScorer.ordToDoc(ord);
            if (acceptBits != null && !acceptBits.get(docId)) continue;
            knnCollector.collect(docId, byteScorer.score(ord));
            knnCollector.incVisitedCount(1);
        }
    }

    @Override
    public void warmUp(String fieldName) throws IOException {
        // Warm centroids into memory
        Integer fieldNumber = fieldNameToNumber.get(fieldName);
        if (fieldNumber != null) {
            ClusterANNFieldState state = fieldStates.get(fieldNumber);
            if (state != null && !state.isEmpty()) {
                state.ensureCentroidsLoaded(centroidsInput);
            }
        }
        // Warm flat vectors
        FloatVectorValues values = flatVectorsReader.getFloatVectorValues(fieldName);
        if (values != null) {
            WarmupUtil.readAll(values);
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsReader, centroidsInput, postingsInput, quantizedInput);
    }

    // ========== Private: Posting List Scanning ==========

    private void scanPostingList(int listSize, ClusterANNScorer scorer,
                                 TwoPhaseClusterANNScorer twoPhase,
                                 ClusterANNFieldState fieldState, int centroidIdx,
                                 float[] target, Bits acceptBits, Set<Integer> visited,
                                 KnnCollector knnCollector, boolean useADC) throws IOException {
        if (listSize == 0) return;

        if (useADC && twoPhase != null) {
            // ADC path: quantize query per centroid, prefetch then score
            float[] centroid = new float[fieldState.dimension];
            System.arraycopy(fieldState.centroids, centroidIdx * fieldState.dimension,
                centroid, 0, fieldState.dimension);
            TwoPhaseClusterANNScorer.QueryQuantization qQuant = twoPhase.quantizeQuery(centroid);

            // Collect valid ordinals first
            int[] allOrds = new int[listSize];
            for (int i = 0; i < listSize; i++) {
                allOrds[i] = postingsInput.readVInt();
            }

            int[] validOrds = new int[listSize];
            int validCount = 0;
            for (int i = 0; i < listSize; i++) {
                int ord = allOrds[i];
                int doc = scorer.ordToDoc(ord);
                if (!visited.add(doc)) continue;
                if (acceptBits != null && !acceptBits.get(doc)) continue;
                validOrds[validCount++] = ord;
            }

            // Prefetch quantized data for all valid ordinals
            if (validCount > 0) {
                twoPhase.prefetch(validOrds, validCount);

                // Score all prefetched candidates
                for (int i = 0; i < validCount; i++) {
                    twoPhase.scoreADC(validOrds[i], qQuant.transposed, qQuant.ay, qQuant.ly,
                        qQuant.queryComponentSum, qQuant.additionalCorrection);
                }
                knnCollector.incVisitedCount(validCount);
            }
        } else {
            // Exact path: collect ordinals into batch, then bulkScore for prefetch + SIMD
            ExactClusterANNScorer exactScorer = (ExactClusterANNScorer) scorer;
            exactScorer.ensureBatchCapacity(listSize);
            int[] batchOrds = exactScorer.batchOrdinals();

            // Read all ordinals from posting list, filter by acceptDocs
            int batchCount = 0;
            int[] allOrds = new int[listSize];
            for (int i = 0; i < listSize; i++) {
                allOrds[i] = postingsInput.readVInt();
            }

            for (int i = 0; i < listSize; i++) {
                int ord = allOrds[i];
                int doc = exactScorer.ordToDoc(ord);
                if (!visited.add(doc)) continue;
                if (acceptBits != null && !acceptBits.get(doc)) continue;
                batchOrds[batchCount++] = ord;
            }

            if (batchCount > 0) {
                // Bulk score: triggers prefetch + native SIMD in one call
                float[] batchScores = exactScorer.batchScores();
                exactScorer.bulkScore(batchOrds, batchScores, batchCount);

                // Collect results
                for (int i = 0; i < batchCount; i++) {
                    int doc = exactScorer.ordToDoc(batchOrds[i]);
                    knnCollector.collect(doc, batchScores[i]);
                }
                knnCollector.incVisitedCount(batchCount);
            }
        }
    }

    private void skipPostingList(int listSize) throws IOException {
        for (int i = 0; i < listSize; i++) {
            postingsInput.readVInt();
        }
    }

    // ========== Private: Posting List Prefetch ==========

    /**
     * Prefetch posting list byte ranges for probed centroids from .clap.
     * Issues OS-level prefetch hints so data is in page cache before we seek to read.
     */
    private void prefetchPostingLists(ClusterANNFieldState fieldState, int[] nearestCentroids, int nprobe)
        throws IOException {
        // Estimate average posting size for prefetch range calculation
        // Use a conservative estimate: avgVectorsPerCentroid * 5 bytes (VInt avg)
        int avgPostingBytes = Math.max(64, (fieldState.numVectors / fieldState.numCentroids) * 5);

        for (int i = 0; i < nprobe; i++) {
            int centId = nearestCentroids[i];
            long primaryOff = fieldState.primaryPostingOffsets[centId];
            long soarOff = fieldState.soarPostingOffsets[centId];
            postingsInput.prefetch(primaryOff, avgPostingBytes);
            postingsInput.prefetch(soarOff, avgPostingBytes);
        }
    }

    // ========== Private: Centroid Selection ==========

    private static int calculateNprobe(int numCentroids, int numVectors, int k) {
        if (numCentroids <= 10) return numCentroids;
        int nprobe = Math.max(10, (int) Math.sqrt(numCentroids));
        return Math.min(nprobe, numCentroids);
    }

    private static int[] findNearestCentroids(float[] query, float[] centroids,
                                               int numCentroids, int dimension,
                                               DistanceMetric metric, int nprobe) {
        // Partial selection sort: find top-nprobe nearest
        float[] dists = new float[numCentroids];
        int[] indices = new int[numCentroids];
        for (int c = 0; c < numCentroids; c++) {
            dists[c] = metric.distance(query, 0, centroids, c * dimension, dimension);
            indices[c] = c;
        }

        for (int i = 0; i < nprobe; i++) {
            int minIdx = i;
            for (int j = i + 1; j < numCentroids; j++) {
                if (dists[indices[j]] < dists[indices[minIdx]]) {
                    minIdx = j;
                }
            }
            int tmp = indices[i];
            indices[i] = indices[minIdx];
            indices[minIdx] = tmp;
        }

        int[] result = new int[nprobe];
        System.arraycopy(indices, 0, result, 0, nprobe);
        return result;
    }


    // ========== Private: Brute Force Fallback ==========

    private void bruteForceSearch(String field, float[] target, KnnCollector knnCollector,
                                  AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer scorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (scorer == null) return;
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        for (int ord = 0; ord < scorer.maxOrd(); ord++) {
            int docId = scorer.ordToDoc(ord);
            if (acceptBits != null && !acceptBits.get(docId)) continue;
            knnCollector.collect(docId, scorer.score(ord));
            knnCollector.incVisitedCount(1);
        }
    }

    // ========== Private: Helpers ==========

    private IndexInput openInput(SegmentReadState state, String extension) throws IOException {
        String fileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, extension);
        IndexInput input = state.directory.openInput(fileName, state.context);
        CodecUtil.checkIndexHeader(input, ClusterANN1040KnnVectorsWriter.CODEC_NAME,
            ClusterANN1040KnnVectorsWriter.VERSION_START,
            ClusterANN1040KnnVectorsWriter.VERSION_CURRENT,
            state.segmentInfo.getId(), state.segmentSuffix);
        return input;
    }

    private VectorSimilarityFunction getSimFunc(String field) {
        FieldInfo fi = fieldInfos.fieldInfo(field);
        return fi != null ? fi.getVectorSimilarityFunction() : VectorSimilarityFunction.EUCLIDEAN;
    }
}
