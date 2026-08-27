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
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.CENTROIDS_EXTENSION;
import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.CODEC_NAME;
import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.META_EXTENSION;
import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.POSTINGS_EXTENSION;
import static org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants.ROTATION_EXTENSION;

import org.opensearch.knn.index.clusterann.codec.CentroidVectorValues;
import org.opensearch.knn.index.clusterann.codec.ClipPruningData;
import org.opensearch.knn.index.clusterann.codec.ClusterANNFieldState;
import org.opensearch.knn.index.clusterann.codec.ClusterANNFormatConstants;
import org.opensearch.knn.index.clusterann.codec.ClusterFactory;
import org.opensearch.knn.index.clusterann.codec.ClusterSearcher;
import org.opensearch.knn.index.clusterann.codec.Clusters;
import org.opensearch.knn.index.clusterann.codec.ScanParams;
import org.opensearch.knn.index.clusterann.prefetch.CentroidProbePlanner;

/**
 * Reader for ClusterANN IVF format v2.
 *
 * <p>Search uses a composable probe pipeline:
 * {@link CentroidProbePlanner} → {@link ClusterSearcher} walk (which owns level-1 prefetch)
 *
 */
@Log4j2
public class ClusterANN1040KnnVectorsReader extends KnnVectorsReader {

    private final FlatVectorsReader flatVectorsReader;
    private final Map<Integer, ClusterANNFieldState> fieldStates;
    private final Map<String, Integer> fieldNameToNumber;
    private final FieldInfos fieldInfos;

    private final IndexInput metaInput;
    private final IndexInput postingsInput;
    private IndexInput centroidsInput;
    private IndexInput rotationInput;
    private Map<Integer, ClipPruningData> clipDataMap;
    // Per-field search structures are built once at reader open (single-threaded ctor) and never mutated
    // after, so the search path reads them from these final, immutable maps with no locking — safely
    // published to concurrent-search threads by the final fields. Empty fields are absent.
    //   clacRegion1Map: .clac region 1 (ordToDoc DISI config + ordToCentroid).
    //   clustersMap:    the field's persistent cluster structure (HnswGraph analogue) + its .clap slice.
    private final Map<Integer, ClacRegion1> clacRegion1Map;
    private final Map<Integer, Clusters> clustersMap;
    // Stateless search algorithm — all per-query state lives in the ClusterSearchContext passed to
    // search(), so one shared instance serves every query/thread.
    private static final int ORD_TO_DOC_BLOCK_SHIFT = 16;

    public ClusterANN1040KnnVectorsReader(FlatVectorsReader flatVectorsReader, SegmentReadState state) throws IOException {
        this.flatVectorsReader = flatVectorsReader;

        boolean success = false;
        IndexInput metaIn = null;
        IndexInput postIn = null;
        IndexInput centIn = null;
        IndexInput rotIn = null;
        try {
            metaIn = openInput(state, META_EXTENSION);
            postIn = openInput(state, POSTINGS_EXTENSION);
            centIn = openInput(state, CENTROIDS_EXTENSION);
            rotIn = openInput(state, ROTATION_EXTENSION);
            this.centroidsInput = centIn;
            this.rotationInput = rotIn;

            this.fieldStates = ClusterANNFieldState.readAll(metaIn, state);

            this.fieldNameToNumber = new HashMap<>();
            for (FieldInfo fi : state.fieldInfos) {
                if (fieldStates.containsKey(fi.number)) {
                    fieldNameToNumber.put(fi.getName(), fi.number);
                }
            }

            this.metaInput = metaIn;
            this.postingsInput = postIn;
            this.fieldInfos = state.fieldInfos;

            // Build the per-field search structures eagerly (single-threaded here), so the search path
            // reads immutable maps with no synchronization. Empty fields are skipped (never searched).
            Map<Integer, ClacRegion1> region1 = new HashMap<>();
            Map<Integer, Clusters> clusters = new HashMap<>();
            for (Map.Entry<Integer, ClusterANNFieldState> entry : fieldStates.entrySet()) {
                ClusterANNFieldState fs = entry.getValue();
                if (fs.isEmpty()) continue;
                int fieldNumber = entry.getKey();
                region1.put(fieldNumber, buildClacRegion1(fs));
                FieldInfo fi = state.fieldInfos.fieldInfo(fieldNumber);
                VectorSimilarityFunction sim =
                    fi != null ? fi.getVectorSimilarityFunction() : VectorSimilarityFunction.EUCLIDEAN;
                ClusterFactory factory = new ClusterFactory(fs.quantizerType, fs.docBits, fs.dimension, sim);
                clusters.put(fieldNumber, new Clusters(postIn, centIn, rotIn, fs, factory));
            }
            this.clacRegion1Map = region1;
            this.clustersMap = clusters;
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(metaIn, postIn, centIn, rotIn, flatVectorsReader);
            }
        }

        log.debug("[ClusterANN] reader created: {} fields with IVF index", fieldStates.size());
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

    private static final int MIN_IVF_VECTORS = 100;

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        Integer fieldNumber = fieldNameToNumber.get(field);
        ClusterANNFieldState fieldState = fieldNumber != null ? fieldStates.get(fieldNumber) : null;

        // The codec is uniform-ADC only — there is no full-precision brute-force fallback (that
        // would emit exact scores into a shared multi-segment collector alongside ADC scores).
        // Reaching here with no/empty ClusterANN data is unexpected in the POC; fail fast.
        if (fieldState == null || fieldState.isEmpty()) {
            throw new IllegalStateException(
                "ClusterANN: search invoked for field '" + field + "' with no/empty ClusterANN data");
        }
        assert fieldState.docBits > 0 : "Cannot have docBits <= 0";

        int k = knnCollector.k();
        long t0 = System.nanoTime();
        log.debug("[ClusterANN-SEARCH] collector.k={}", k);
        // Ord-space filter for the searcher/cluster (doc→ord already resolved). null = accept all.
        Bits acceptedOrds = buildAcceptedOrds(fieldNumber, fieldState, acceptDocs);
        long filterCost = acceptDocs != null ? acceptDocs.cost() : fieldState.numVectors;

        VectorSimilarityFunction simFunc = getSimFunc(field);
        // The raw centroid region carries the trailing ‖c‖² only for L2 (ranking norm-trick).
        boolean rawHasNorm = simFunc == VectorSimilarityFunction.EUCLIDEAN;

        Clusters clusters = clustersMap.get(fieldNumber); // built at open; present for non-empty fields
        // Project the query into the space this field's codes were written in (once), then bundle it with
        // this query's per-query facts. The walk forwards the bundle to each cluster untouched, so varying
        // e.g. the query-side quantization width never reaches the search algorithm.
        float filterSelectivity = fieldState.numVectors > 0 ? (float) filterCost / fieldState.numVectors : 1.0f;
        ScanParams scanParams = ScanParams.of(clusters.prepareQuery(target), Math.min(filterSelectivity, 1.0f));

        // ord→doc translation is the collector's job: the searcher collects by ordinal, and this wrapper
        // maps ordinals to docids (Lucene's OrdinalTranslatedKnnCollector, same as HNSW). It's a
        // KnnCollector.Decorator, so incVisitedCount / minCompetitiveSimilarity / topDocs delegate to the
        // real collector.
        LongValues ordToDoc = getOrdToDoc(fieldNumber);
        KnnCollector collector = new OrdinalTranslatedKnnCollector(knnCollector, ord -> (int) ordToDoc.get(ord));

        // Per-query raw-centroid values view for probe ranking (hides .clac I/O behind FloatVectorValues).
        // rawHasNorm is a storage-layout flag (does each centroid carry a trailing ‖c‖²), not a metric.
        CentroidVectorValues centroidValues = new CentroidVectorValues(
                centroidsInput,
                fieldState.clacCentroidOffset,
                fieldState.numCentroids,
                fieldState.dimension,
                rawHasNorm
        );

        // Tier-2 density-weighted probing is temporarily disabled during the .claf → .clac
        // region-1 migration. Filtered queries use normal nprobe; the scanner still applies the
        // per-vector acceptBits filter, so results are correct — just without density weighting.
        // TODO: rewire matchCount from .clac region 1 (OrdToDocDISI doc→ord + ordToCentroid).
        long t1 = System.nanoTime();
        // Plan the probes (rank centroids closest-first + knee), then walk them. The walk owns the
        // prefetch window too, since only it knows the probe order and which clusters it will skip.
        int[] probes = CentroidProbePlanner.planProbes(centroidValues, target, fieldState.metric, k);
        int clustersProbed = ClusterSearcher.search(clusters, probes, scanParams, collector, acceptedOrds);
        long t2 = System.nanoTime();

        long t3 = System.nanoTime();
        log.info(
            "[ClusterANN-SEG] nprobe={} clustersProbed={} vectors={} centroidRank={}ms scan={}ms total={}ms",
            probes.length,
            clustersProbed,
            fieldState.numVectors,
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t3 - t0) / 1_000_000
        );
    }


    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        // Byte/binary vectors are not supported by ClusterANN in the POC.
        throw new UnsupportedOperationException("ClusterANN does not support byte/binary vector search");
    }

    @Override
    public void close() throws IOException {
        // Per-query centroid readers only hold clones of centroidsInput (never closed); the shared
        // input is owned and closed here.
        IOUtils.close(flatVectorsReader, metaInput, postingsInput, centroidsInput, rotationInput);
    }

    // ========== Helpers ==========

    /**
     * Per-field .clac region 1: the OrdToDocDISI config (doc↔ord) plus the ordToCentroid array.
     * Heap-loaded from the self-contained region for POC.
     */
    private static final class ClacRegion1 {
        final OrdToDocDISIReaderConfiguration config;
        final byte[] disiData;                 // input for getIndexedDISI / getDirectMonotonicReader
        final LongValues ordToDoc;             // ord → doc (identity when the field is dense)
        final int[] ordToCentroid;             // ord → primary centroid (for filter matchCount)

        ClacRegion1(OrdToDocDISIReaderConfiguration config, byte[] disiData,
                    LongValues ordToDoc, int[] ordToCentroid) {
            this.config = config;
            this.disiData = disiData;
            this.ordToDoc = ordToDoc;
            this.ordToCentroid = ordToCentroid;
        }
    }

    /** Build a field's .clac region 1. Called once per field at reader open; result is cached. */
    private ClacRegion1 buildClacRegion1(ClusterANNFieldState fieldState) throws IOException {
        IndexInput in = centroidsInput.clone();
        in.seek(fieldState.clacRegion1Offset);
        int numValues = in.readVInt();
        int metaLen = in.readVInt();
        byte[] metaBytes = new byte[metaLen];
        in.readBytes(metaBytes, 0, metaLen);
        int dataLen = in.readVInt();
        byte[] disiData = new byte[dataLen];
        in.readBytes(disiData, 0, dataLen);
        int[] ordToCentroid = new int[numValues];
        for (int i = 0; i < numValues; i++) {
            ordToCentroid[i] = in.readShort() & 0xFFFF;
        }

        ByteArrayIndexInput metaIn = new ByteArrayIndexInput("clacR1Meta", metaBytes);
        OrdToDocDISIReaderConfiguration config = OrdToDocDISIReaderConfiguration.fromStoredMeta(metaIn, numValues);
        // Dense/empty fields carry no ord→doc monotonic map (ord == doc); only sparse fields do.
        LongValues ordToDoc = (config.isDense() || config.isEmpty())
            ? LongValues.IDENTITY
            : config.getDirectMonotonicReader(new ByteArrayIndexInput("clacR1Data", disiData));

        return new ClacRegion1(config, disiData, ordToDoc, ordToCentroid);
    }

    /** ord → doc reader for a field (from .clac region 1, built at open); identity when dense. */
    private LongValues getOrdToDoc(int fieldNumber) {
        return clacRegion1Map.get(fieldNumber).ordToDoc;
    }

    /**
     * Translate the query's doc-space filter into an ord-space {@link Bits} the searcher/cluster can
     * test directly (the cluster deals only in ordinals). Returns {@code null} for "accept all" so the
     * hot path pays nothing when unfiltered.
     *
     * <p><b>Dense fields</b> have {@code ord == doc}, so the doc-space {@code Bits} is already ord-space and
     * is returned as-is.
     *
     * <p><b>Sparse fields</b> get a view that translates per test — mapping {@code ord} through
     * {@code ordToDoc} and asking the doc filter — rather than materializing a bitset up front. The
     * alternative, driving the jump-table {@link IndexedDISI} by the accepted docs to set a bit per accepted
     * ordinal, buys a cheaper per-test read at a cost of {@code O(#accepted)} and a bitset the width of the
     * field. That trade only pays in a narrow middle band, and loses badly outside it: a loose filter would
     * walk most of the field through the DISI on every query, while a very selective one has its clusters
     * dropped wholesale by the expected-match guard before many tests happen at all. Translating on demand
     * is within a small factor in the band where materializing is best, and far cheaper everywhere else.
     *
     * <p>Either way the tests are ord-keyed random access into heap structures, so neither disturbs the
     * sequential file access the scan depends on.
     */
    private Bits buildAcceptedOrds(int fieldNumber, ClusterANNFieldState fieldState, AcceptDocs acceptDocs)
        throws IOException {
        if (acceptDocs == null) return null;
        Bits docBits = acceptDocs.bits();
        if (docBits == null) return null; // match-all
        ClacRegion1 region = clacRegion1Map.get(fieldNumber); // built at open
        if (region.config.isDense() || region.config.isEmpty()) {
            return docBits; // ord == doc
        }
        LongValues ordToDoc = region.ordToDoc;
        int numVectors = fieldState.numVectors;
        // Safe to share: ordToDoc is a monotonic reader over a heap byte array, so it carries no file cursor
        // — unlike a normal IndexInput — and is already read concurrently by the ord→doc collector wrapper.
        return new Bits() {
            @Override
            public boolean get(int ord) {
                return docBits.get((int) ordToDoc.get(ord));
            }

            @Override
            public int length() {
                return numVectors;
            }
        };
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
