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
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.*;
import org.opensearch.knn.index.clusterann.codec.*;
import org.opensearch.knn.index.clusterann.prefetch.*;

/**
 * Reader for ClusterANN IVF format v2.
 *
 * <p>Search uses a composable probe pipeline:
 * {@link NearestProbeScheduler} → {@link OptimizedProbeScheduler}
 * feeding a {@link ClusterANNCentroidScanner}.
 */
@Log4j2
public class ClusterANN1040KnnVectorsReader extends KnnVectorsReader {

    private final FlatVectorsReader flatVectorsReader;
    private final Map<Integer, ClusterANNFieldState> fieldStates;
    private final Map<String, Integer> fieldNameToNumber;
    private final FieldInfos fieldInfos;

    private final IndexInput metaInput;
    private final IndexInput postingsInput;
    private CentroidAssignmentReader filterReader;
    private OffHeapCentroids.Reader centroidReader;
    private Map<Integer, ClipPruningData> clipDataMap;

    /** Segment name, kept for the shard-int8 stash token (must match the query-side leaf token). */
    private final String segmentName;

    // PERF instrumentation (opt-in via -Dclusterann.perf=true): accumulate per-phase nanos + query count.
    private static final boolean PERF = Boolean.getBoolean("clusterann.perf");
    private static final java.util.concurrent.atomic.AtomicLong PERF_PREP_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong PERF_ROUTE_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong PERF_SCAN_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong PERF_QUERIES = new java.util.concurrent.atomic.AtomicLong();

    public ClusterANN1040KnnVectorsReader(FlatVectorsReader flatVectorsReader, SegmentReadState state) throws IOException {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentName = state.segmentInfo.name;

        boolean success = false;
        IndexInput metaIn = null;
        IndexInput postIn = null;
        IndexInput filterIn = null;
        IndexInput centIn = null;
        try {
            metaIn = openInput(state, META_EXTENSION);
            postIn = openInput(state, POSTINGS_EXTENSION);
            filterIn = openInput(state, FILTER_EXTENSION);
            centIn = openInput(state, CENTROIDS_EXTENSION);

            this.fieldStates = ClusterANNFieldState.readAll(metaIn, state);

            // Open filter reader and centroid reader for the first field
            if (!fieldStates.isEmpty()) {
                int firstField = fieldStates.keySet().iterator().next();
                this.filterReader = new CentroidAssignmentReader(filterIn, firstField);
                this.centroidReader = new OffHeapCentroids.Reader(centIn, firstField);
            }

            // Read CLIP pruning data if available
            this.clipDataMap = readClipData(state);

            this.fieldNameToNumber = new HashMap<>();
            for (FieldInfo fi : state.fieldInfos) {
                if (fieldStates.containsKey(fi.number)) {
                    fieldNameToNumber.put(fi.getName(), fi.number);
                }
            }

            this.metaInput = metaIn;
            this.postingsInput = postIn;
            this.fieldInfos = state.fieldInfos;
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(metaIn, postIn, filterIn, centIn, flatVectorsReader);
            }
        }

        log.debug("[ClusterANN] reader created: {} fields with IVF index", fieldStates.size());    }

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

    // ================= Donor-seed merge support =================

    /** Donor data extracted from a source segment for the donor-seed merge optimization. */
    public static final class DonorData {
        public final float[][] centroids;      // [numCentroids][dim] raw centroids
        public final int[] ordToCell;          // per source local-ordinal -> primary cell
        public final int numVectors;
        public final int numCentroids;
        DonorData(float[][] c, int[] o, int nv, int nc) {
            this.centroids = c; this.ordToCell = o; this.numVectors = nv; this.numCentroids = nc;
        }
    }

    /**
     * Extract this segment's centroids and per-ordinal cell assignments, for use as a merge donor.
     * Walks each centroid's posting (primary docs only) and records the cell of every ordinal.
     * Returns null if the field has no ClusterANN state.
     */
    public DonorData extractDonor(String field) throws IOException {
        Integer fieldNumber = fieldNameToNumber.get(field);
        if (fieldNumber == null) return null;
        ClusterANNFieldState fs = fieldStates.get(fieldNumber);
        if (fs == null || fs.numVectors == 0 || fs.numCentroids == 0) return null;
        fs.ensureLoaded(metaInput);

        int dim = fs.dimension;
        int nc = fs.numCentroids;
        // Centroids from the .clac reader.
        float[] flat = new float[nc * dim];
        centroidReader.readAllCentroids(flat);
        float[][] centroids = new float[nc][dim];
        for (int c = 0; c < nc; c++) {
            System.arraycopy(flat, c * dim, centroids[c], 0, dim);
        }

        // Reconstruct ord -> cell by walking each centroid's posting header.
        int[] ordToCell = new int[fs.numVectors];
        java.util.Arrays.fill(ordToCell, -1);
        IndexInput in = postingsInput.clone();
        long[] offsets = fs.centroidOffsets;
        int[] scratchDocs = new int[fs.numVectors > 0 ? Math.min(fs.numVectors, 65536) : 1];
        int[] scratchOrds = new int[scratchDocs.length];
        int assignedCount = 0, maxOrd = -1, oobOrd = 0;
        for (int c = 0; c < nc; c++) {
            in.seek(offsets[c]);   // centroidOffsets are absolute file positions (see scanner)
            int count = in.readVInt();
            if (count == 0) { continue; }
            if (count > scratchDocs.length) { scratchDocs = new int[count]; scratchOrds = new int[count]; }
            org.opensearch.knn.index.clusterann.codec.PostingListCodec.readBody(in, count, scratchDocs);
            int ordCount = in.readVInt();
            if (ordCount > 0) {
                if (ordCount > scratchOrds.length) scratchOrds = new int[ordCount];
                in.readInts(scratchOrds, 0, ordCount);
                for (int i = 0; i < ordCount; i++) {
                    int ord = scratchOrds[i];
                    if (ord > maxOrd) maxOrd = ord;
                    if (ord < 0 || ord >= ordToCell.length) { oobOrd++; continue; }
                    if (ordToCell[ord] < 0) {
                        ordToCell[ord] = c;  // primary cell (first occurrence wins)
                        assignedCount++;
                    }
                }
            }
            // skip the quantized blocks for this posting (we only need assignments)
            // handled by seeking per-centroid via offsets, so no explicit skip needed.
        }
        if (Boolean.getBoolean("clusterann.mergeTrace")) {
            org.apache.logging.log4j.LogManager.getLogger(ClusterANN1040KnnVectorsReader.class).info(
                "[ClusterANN-MERGE-TRACE] extractDonor: nc={} numVectors={} assigned={} maxOrd={} oob={}",
                nc, fs.numVectors, assignedCount, maxOrd, oobOrd);
        }
        return new DonorData(centroids, ordToCell, fs.numVectors, nc);
    }

    private static final int MIN_IVF_VECTORS = 100;

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        Integer fieldNumber = fieldNameToNumber.get(field);
        ClusterANNFieldState fieldState = fieldNumber != null ? fieldStates.get(fieldNumber) : null;

        if (fieldState == null || fieldState.isEmpty()) {
            bruteForceSearch(field, target, knnCollector, acceptDocs);
            return;
        }

        fieldState.ensureLoaded(metaInput);

        // Skip IVF for tiny segments — brute force is faster than IVF overhead
        if (fieldState.numVectors < MIN_IVF_VECTORS) {
            bruteForceSearch(field, target, knnCollector, acceptDocs);
            return;
        }

        // Cross-segment CLIP pruning: skip entire segment if upper bound < threshold
        ClipPruningData clipData = clipDataMap != null ? clipDataMap.get(fieldNumber) : null;
        if (clipData != null && clipData.hasSegmentCentroid()) {
            float sharedThreshold = OptimizedProbeScheduler.getSharedThreshold(knnCollector);
            if (sharedThreshold > Float.NEGATIVE_INFINITY) {
                float segUbIP = clipData.segmentUpperBoundIP(target);
                // Transform raw IP to Lucene MAXIMUM_INNER_PRODUCT score
                float segUbScore = segUbIP >= 0 ? segUbIP + 1.0f : 1.0f / (1.0f - segUbIP);
                if (segUbScore < sharedThreshold) {
                    log.debug("[ClusterANN-SEG] CLIP segment skip: ubScore={} < threshold={}",
                        segUbScore, sharedThreshold);
                    return;
                }
            }
        }

        int k = knnCollector.k();
        long t0 = System.nanoTime();
        log.debug("[ClusterANN-SEARCH] collector.k={}", k);
        IndexInput postingsClone = postingsInput.clone();
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        long filterCost = acceptDocs != null ? acceptDocs.cost() : fieldState.numVectors;

        // Build scorers
        RandomVectorScorer exactScorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (exactScorer == null) return;

        VectorSimilarityFunction simFunc = getSimFunc(field);
        boolean useADC = fieldState.docBits > 0 && fieldState.numVectors > MIN_ADC_VECTORS;

        // Flow B ("IVFaster absolute"): build the thermometer scan reader with the Hadamard-rotated
        // query. The writer used HadamardRotation.create(dim) (deterministic default seed), so the
        // reader rebuilds the same rotation from (dim). UNVERIFIED (written blind).
        org.opensearch.knn.index.clusterann.codec.ThermometerVectorReader thermoReader = null;
        if (fieldState.quantizerId == QUANTIZER_IVFASTER_ABSOLUTE) {
            thermoReader = new org.opensearch.knn.index.clusterann.codec.ThermometerVectorReader(fieldState);
            float[] rotatedQuery = new float[fieldState.dimension];
            org.opensearch.knn.index.clusterann.algorithm.HadamardRotation.create(fieldState.dimension)
                .transform(target, rotatedQuery);
            thermoReader.prepareQuery(rotatedQuery);
            // Prepare the int8 query form if int8 is used for ranking (rerank tier OR int8-only flow).
            if ("int8".equalsIgnoreCase(System.getProperty("clusterann.thermo.rerank"))
                    || Boolean.getBoolean("clusterann.thermo.int8only")) {
                thermoReader.prepareInt8Query(rotatedQuery);
            }
        }

        // Flow C ("ScaNN residual PQ"): load the single global PQ codebook once; the scanner
        // residualizes the query per probed cell and scores via the AH lookup table.
        org.opensearch.knn.index.clusterann.codec.PQScanState pqState = null;
        if (fieldState.quantizerId == QUANTIZER_SCANN_RESIDUAL_PQ && fieldState.pqCodebookLength > 0) {
            IndexInput cbIn = postingsClone.clone();
            cbIn.seek(fieldState.pqCodebookOffset);
            org.opensearch.knn.index.clusterann.codec.PQCodebook codebook =
                org.opensearch.knn.index.clusterann.codec.PQCodebook.read(cbIn);
            pqState = new org.opensearch.knn.index.clusterann.codec.PQScanState(codebook, target, simFunc);
        }

        // Transform query for ADC scoring (randomRotation redistributes variance for better quantization).
        // Normally EUCLIDEAN-only; the flowA.rotateIP POC also rotates IP queries. The write side
        // applied rotation whenever hasRotation() is true, so gate on that + the flag for IP.
        float[] adcTarget = target;
        boolean flowARotateIP = Boolean.getBoolean("clusterann.flowA.rotateIP");
        if (useADC && centroidReader.hasRotation()
                && (simFunc == VectorSimilarityFunction.EUCLIDEAN || flowARotateIP)) {
            adcTarget = new float[target.length];
            centroidReader.transformQuery(target, adcTarget);
        }

        // === Three-tier adaptive filtering ===
        // Tier 1: Strict filter — exact score all matching docs (skip IVF)
        // Tier 2: Moderate filter — density-weighted centroid probing
        // Tier 3: Loose/no filter — normal adaptive nprobe
        long tPrep = System.nanoTime();  // PERF: end of query-prep (rotate/quantize/scorer build)
        if (acceptBits != null && filterCost < fieldState.numVectors) {
            long filterDimProduct = filterCost * (long) fieldState.dimension;
            if (filterDimProduct <= EXACT_FILTER_THRESHOLD) {
                // Tier 1: few matching docs — exact score them all
                int limit = Math.min(fieldState.numVectors, acceptBits.length());
                for (int ord = 0; ord < limit; ord++) {
                    if (acceptBits.get(ord)) {
                        float score = exactScorer.score(ord);
                        knnCollector.collect(ord, score);
                    }
                }
                return;
            }
        }

        // For radial search: two-phase (ADC first pass → exact rescore candidates)
        boolean isRadial = !(knnCollector instanceof org.apache.lucene.search.TopKnnCollector);
        if (isRadial && useADC) {
            // Phase 1: ADC scoring into local candidate buffer
            java.util.ArrayList<int[]> candidates = new java.util.ArrayList<>();
            // Use a collecting scanner that gathers docIds instead of submitting to collector
            QuantizedVectorReader adcReader = new QuantizedVectorReader(exactScorer, postingsClone, fieldState, simFunc, adcTarget, 100);
            // Collect candidates via ADC into a temp top-k collector
            org.apache.lucene.search.TopKnnCollector tempCollector =
                new org.apache.lucene.search.TopKnnCollector(100, Integer.MAX_VALUE);
            adcReader.setCollector(tempCollector);

            BitSet visited = new BitSet(fieldState.numVectors);
            ClusterANNCentroidScanner scanner = new ClusterANNCentroidScanner(
                postingsClone, fieldState, exactScorer, adcReader, target, acceptBits, visited, true, centroidReader, thermoReader
            );
            scanner.setSegmentToken(segmentName.hashCode());
            NearestProbeScheduler nearest = new NearestProbeScheduler(target, fieldState, 100, scanner, centroidReader);
            OptimizedProbeScheduler pipeline = new OptimizedProbeScheduler(
                nearest, scanner, postingsClone, fieldState.centroidDocCounts, fieldState.numVectors, 100, filterCost
            );
            pipeline.execute(tempCollector);
            adcReader.finish(tempCollector);

            // Phase 2: exact rescore candidates and submit to real collector
            org.apache.lucene.search.TopDocs topDocs = tempCollector.topDocs();
            for (org.apache.lucene.search.ScoreDoc sd : topDocs.scoreDocs) {
                float exactScore = exactScorer.score(sd.doc);
                knnCollector.collect(sd.doc, exactScore);
            }
            return;
        }

        QuantizedVectorReader adcReader = null;
        if (useADC) {
            adcReader = new QuantizedVectorReader(exactScorer, postingsClone, fieldState, simFunc, adcTarget, k);
            adcReader.setCollector(knnCollector);
        }

        BitSet visited = new BitSet(fieldState.numVectors);

        ClusterANNCentroidScanner scanner = new ClusterANNCentroidScanner(
            postingsClone,
            fieldState,
            exactScorer,
            adcReader,
            adcTarget,
            acceptBits,
            visited,
            useADC,
            centroidReader,
            thermoReader,
            pqState
        );

        scanner.setSegmentToken(segmentName.hashCode());

        NearestProbeScheduler nearest;
        int[] filterMatchCounts = null;
        if (acceptBits != null && filterReader != null && filterCost < fieldState.numVectors
                ) {
            // Tier 2: moderate filter with few matches — density-weighted probing
            try {
                FixedBitSet acceptedOrds = new FixedBitSet(fieldState.numVectors);
                int limit = Math.min(fieldState.numVectors, acceptBits.length());
                for (int doc = 0; doc < limit; doc++) {
                    if (acceptBits.get(doc)) acceptedOrds.set(doc);
                }
                filterMatchCounts = new int[fieldState.numCentroids];
                FixedBitSet acceptCentroids = filterReader.computeCentroidFilter(acceptedOrds, filterMatchCounts);
                nearest = new NearestProbeScheduler(target, fieldState, k, scanner, acceptCentroids, filterMatchCounts, centroidReader);
            } catch (Exception e) {
                // Fallback to normal probing if .claf is incompatible
                nearest = new NearestProbeScheduler(target, fieldState, k, scanner, centroidReader);
                filterMatchCounts = null;
            }
        } else {
            // Tier 3: no filter or loose filter — normal adaptive nprobe
            nearest = new NearestProbeScheduler(target, fieldState, k, scanner, centroidReader);
        }
        long t1 = System.nanoTime();
        OptimizedProbeScheduler pipeline = new OptimizedProbeScheduler(
            nearest,
            scanner,
            postingsClone,
            fieldState.centroidDocCounts,
            fieldState.centroidNorms,
            fieldState.numVectors,
            k,
            filterCost,
            filterMatchCounts,
            clipData
        );
        pipeline.execute(knnCollector);
        long t2 = System.nanoTime();

        if (PERF) {
            PERF_PREP_NS.addAndGet(tPrep - t0);
            PERF_ROUTE_NS.addAndGet(t1 - tPrep);
            PERF_SCAN_NS.addAndGet(t2 - t1);
            long n = PERF_QUERIES.incrementAndGet();
            // Log a running average every 100 queries so the benchmark's tail is representative.
            if (n % 100 == 0) {
                log.info("[ClusterANN-PERF] queries={} avg_us prep={} route={} scan={} (total={})",
                    n,
                    PERF_PREP_NS.get() / n / 1000,
                    PERF_ROUTE_NS.get() / n / 1000,
                    PERF_SCAN_NS.get() / n / 1000,
                    (PERF_PREP_NS.get() + PERF_ROUTE_NS.get() + PERF_SCAN_NS.get()) / n / 1000);
            }
        }

        if (adcReader != null) {
            adcReader.finish(knnCollector);
        }
        long t3 = System.nanoTime();
        long actualAdcBytes = adcReader != null ? adcReader.getBytesRead() : 0;
        // Accumulate actual bytes (not estimated) into query-level counter
        OptimizedProbeScheduler.addActualBytes(actualAdcBytes);
        log.info(
            "[ClusterANN-SEG] nprobe={} clustersProbed={} vectors={} prep_us={} route_us={} scan_us={} total_us={}",
            nearest.nprobe(),
            OptimizedProbeScheduler.lastClustersProbed(),
            fieldState.numVectors,
            (tPrep - t0) / 1000,
            (t1 - tPrep) / 1000,
            (t2 - t1) / 1000,
            (t3 - t0) / 1000
        );
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer byteScorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (byteScorer == null) return;
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        int maxOrd = byteScorer.maxOrd();
        int[] ords = new int[Math.min(maxOrd, 256)];
        float[] scores = new float[ords.length];
        int count = 0;
        for (int ord = 0; ord < maxOrd; ord++) {
            int docId = byteScorer.ordToDoc(ord);
            if (acceptBits != null && !acceptBits.get(docId)) continue;
            ords[count++] = ord;
            if (count == ords.length) {
                byteScorer.bulkScore(ords, scores, count);
                for (int j = 0; j < count; j++)
                    knnCollector.collect(byteScorer.ordToDoc(ords[j]), scores[j]);
                knnCollector.incVisitedCount(count);
                count = 0;
            }
        }
        if (count > 0) {
            byteScorer.bulkScore(ords, scores, count);
            for (int j = 0; j < count; j++)
                knnCollector.collect(byteScorer.ordToDoc(ords[j]), scores[j]);
            knnCollector.incVisitedCount(count);
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsReader, metaInput, postingsInput, filterReader, centroidReader);
    }

    // ========== Brute Force Fallback ==========

    private void bruteForceSearch(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        RandomVectorScorer scorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (scorer == null) return;
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        int maxOrd = scorer.maxOrd();
        int[] ords = new int[Math.min(maxOrd, 256)];
        float[] scores = new float[ords.length];
        int count = 0;
        for (int ord = 0; ord < maxOrd; ord++) {
            int docId = scorer.ordToDoc(ord);
            if (acceptBits != null && !acceptBits.get(docId)) continue;
            ords[count++] = ord;
            if (count == ords.length) {
                scorer.bulkScore(ords, scores, count);
                for (int i = 0; i < count; i++)
                    knnCollector.collect(scorer.ordToDoc(ords[i]), scores[i]);
                knnCollector.incVisitedCount(count);
                count = 0;
            }
        }
        if (count > 0) {
            scorer.bulkScore(ords, scores, count);
            for (int i = 0; i < count; i++)
                knnCollector.collect(scorer.ordToDoc(ords[i]), scores[i]);
            knnCollector.incVisitedCount(count);
        }
    }

    // ========== Helpers ==========

    private Map<Integer, ClipPruningData> readClipData(SegmentReadState state) {
        // CLIP pruning DISABLED: always return an empty map so no field has CLIP data and every
        // query-path guard (segment-skip in search(), inter-cluster skip in OptimizedProbeScheduler)
        // sees clipData == null and does not prune. Removes the recall loss from CLIP over-pruning.
        return new HashMap<>();
    }

    private Map<Integer, ClipPruningData> readClipDataDISABLED(SegmentReadState state) {
        Map<Integer, ClipPruningData> map = new HashMap<>();
        try {
            String fileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name, state.segmentSuffix, ClipPruningData.EXTENSION);
            if (!Arrays.asList(state.directory.listAll()).contains(fileName)) {
                return map;
            }
            IndexInput clipIn = openInput(state, ClipPruningData.EXTENSION);
            try {
                while (clipIn.getFilePointer() < clipIn.length() - CodecUtil.footerLength()) {
                    int fieldNumber = clipIn.readInt();
                    ClipPruningData data = ClipPruningData.read(clipIn);
                    map.put(fieldNumber, data);
                }
            } finally {
                clipIn.close();
            }
        } catch (IOException e) {
            log.debug("[ClusterANN] CLIP data not available, pruning disabled: {}", e.getMessage());
        }
        return map;
    }

    private IndexInput openInput(SegmentReadState state, String extension) throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        IndexInput input = state.directory.openInput(fileName, state.context);
        CodecUtil.checkIndexHeader(
            input,
            CODEC_NAME,
            ClusterANNFormatConstants.VERSION_START,
            ClusterANNFormatConstants.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix
        );
        return input;
    }

    private VectorSimilarityFunction getSimFunc(String field) {
        FieldInfo fi = fieldInfos.fieldInfo(field);
        return fi != null ? fi.getVectorSimilarityFunction() : VectorSimilarityFunction.EUCLIDEAN;
    }
}
