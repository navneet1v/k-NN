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

import java.io.IOException;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.codec.KNN1040Codec.clusterann.ClusterANNFormatConstants.*;
import org.opensearch.knn.index.codec.KNN1040Codec.clusterann.*;
import org.opensearch.knn.index.codec.KNN1040Codec.clusterann.prefetch.*;

/**
 * Reader for ClusterANN IVF format v2.
 *
 * <p>Search uses a composable probe pipeline:
 * {@link NearestProbeIterator} → {@link SequentialProbeIterator} → {@link PrefetchingProbeIterator}
 * feeding a {@link ClusterANNPostingVisitor}.
 */
@Log4j2
public class ClusterANN1040KnnVectorsReader extends KnnVectorsReader {

    private final FlatVectorsReader flatVectorsReader;
    private final Map<Integer, ClusterANNFieldState> fieldStates;
    private final Map<String, Integer> fieldNameToNumber;
    private final FieldInfos fieldInfos;

    private final IndexInput metaInput;
    private final IndexInput postingsInput;

    public ClusterANN1040KnnVectorsReader(FlatVectorsReader flatVectorsReader, SegmentReadState state) throws IOException {
        this.flatVectorsReader = flatVectorsReader;

        boolean success = false;
        IndexInput metaIn = null;
        IndexInput postIn = null;
        try {
            metaIn = openInput(state, META_EXTENSION);
            postIn = openInput(state, POSTINGS_EXTENSION);

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
            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(metaIn, postIn, flatVectorsReader);
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
            bruteForceSearch(field, target, knnCollector, acceptDocs);
            return;
        }

        fieldState.ensureLoaded(metaInput);

        int k = knnCollector.k();
        IndexInput postingsClone = postingsInput.clone();
        Bits acceptBits = acceptDocs != null ? acceptDocs.bits() : null;
        long filterCost = acceptDocs != null ? acceptDocs.cost() : fieldState.numVectors;

        // Build probe pipeline: Nearest → Filter → Sequential → Prefetch → Budget
        ProbeIterator probes = new NearestProbeIterator(target, fieldState, k);
        probes = new FilterAwareProbeIterator(probes, fieldState.centroidDocCounts, fieldState.numVectors, filterCost);
        probes = new SequentialProbeIterator(probes);
        probes = new PrefetchingProbeIterator(probes, postingsClone);
        BudgetedProbeIterator budgeted = new BudgetedProbeIterator(probes, knnCollector, fieldState.numVectors, k);

        // Build scorers
        RandomVectorScorer exactScorer = flatVectorsReader.getRandomVectorScorer(field, target);
        if (exactScorer == null) return;

        VectorSimilarityFunction simFunc = getSimFunc(field);
        boolean useADC = fieldState.docBits > 0 && fieldState.numVectors > MIN_ADC_VECTORS;

        QuantizedVectorReader adcReader = null;
        if (useADC) {
            adcReader = new QuantizedVectorReader(exactScorer, postingsClone, fieldState, simFunc, target, k);
        }

        BitSet visited = new BitSet(fieldState.numVectors);

        PostingVisitor visitor = new ClusterANNPostingVisitor(
            postingsClone,
            fieldState,
            exactScorer,
            adcReader,
            target,
            acceptBits,
            visited,
            useADC
        );

        // Visit loop with budget tracking
        while (budgeted.hasNext()) {
            ProbedCentroid probe = budgeted.next();
            visitor.reset(probe);
            int scored = visitor.visit(knnCollector);
            budgeted.recordVisit(fieldState.centroidDocCounts[probe.centroidIdx()], scored);
            if (knnCollector.earlyTerminated()) break;
        }

        if (adcReader != null) {
            adcReader.finish(knnCollector);
        }
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
        IOUtils.close(flatVectorsReader, metaInput, postingsInput);
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
