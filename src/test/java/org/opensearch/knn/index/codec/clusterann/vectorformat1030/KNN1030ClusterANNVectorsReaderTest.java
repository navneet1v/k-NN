/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.store.MockDirectoryWrapper;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.read.ClusterANNFieldMetaEncoder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Isolated unit tests for {@link KNN1030ClusterANNVectorsReader}: Mockito for the delegate, and a real
 * in-memory {@code .clam} file for the metadata the constructor parses.
 *
 * <p>The reader takes its {@link FlatVectorsReader} delegate as a constructor argument, so the exact-search
 * branches (accept-doc filtering and competitive skipping) can be driven with mocks. The constructor itself
 * needs real bytes, since it reads the metadata file of the segment it is given — so each case builds its own
 * one-field segment in a {@link ByteBuffersDirectory}, wrapped so that an input left open is a failure.
 */
class KNN1030ClusterANNVectorsReaderTest {

    private static final String FIELD = "float_field";
    private static final String SEGMENT = "_0";
    private static final int FIELD_NUMBER = 0;
    private static final int DIMENSION = 8;
    private static final int BLOCK_SIZE = 32;
    private static final int MAX_DOC = 100;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

    private static final byte ROTATION_NONE = 0;
    private static final byte ROTATION_RANDOM_GAUSSIAN = 1;

    /** Stored code width, matching {@code ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE} — one the ADC scorer supports. */
    private static final byte DOC_BITS = 1;

    /** Long enough for every region the default entry describes, so a slice never runs off the end. */
    private static final int DATA_FILE_BYTES = 4096;

    private final FlatVectorsReader raw = mock(FlatVectorsReader.class);
    private final List<Directory> directories = new ArrayList<>();
    private final List<KNN1030ClusterANNVectorsReader> readers = new ArrayList<>();

    private KNN1030ClusterANNVectorsReader reader;

    @BeforeEach
    void setUp() throws IOException {
        reader = openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry());
    }

    /**
     * Readers first, then the directories they read through. A reader holds inputs onto its directory for as long as it
     * is open, and a {@link MockDirectoryWrapper} refuses to close while any of them is — so releasing them in the wrong
     * order, or not at all, fails here rather than silently outliving the case that opened it.
     */
    @AfterEach
    void tearDown() throws IOException {
        IOUtils.close(readers);
        IOUtils.close(directories);
    }

    // ---------------------------------------------------------------- constructor

    /**
     * A whole metadata file opens: header, block size, one field entry, terminator and footer. Getting to the
     * footer check without a checksum failure is itself the assertion — it can only happen if the entry was
     * consumed to exactly the byte the writer stopped at.
     */
    @Test
    void constructor_readsMetaFile_whenEntryMatchesFieldInfo() throws Exception {
        openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry());
    }

    @Test
    void constructor_readsMetaFile_whenFieldIsRotated() throws Exception {
        // given / when / then — a rotated field is the only kind with a .clar file to slice, and reaching the footer
        // check without a checksum failure means the entry and that slice were both consumed to the byte
        openReader(
            floatVectorField(DIMENSION, SIMILARITY),
            validEntry().rotationId(ROTATION_RANDOM_GAUSSIAN).clarOffset(512L).clarLength(64L)
        );
    }

    /**
     * Nothing writes {@code .clar} for a segment whose fields are all unrotated, so the reader must not go looking
     * for one. The file's absence is asserted rather than assumed: opening it unconditionally would fail here.
     */
    @Test
    void constructor_doesNotOpenRotation_whenNoFieldIsRotated() throws Exception {
        // given / when — opening at all is half the assertion: an unconditional .clar open would throw here
        openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry());

        // then
        String rotationFile = IndexFileNames.segmentFileName(SEGMENT, "", KNN1030ClusterANNVectorsFormat.ROTATION_EXTENSION);
        assertFalse(List.of(lastDirectory().listAll()).contains(rotationFile), "the segment has no rotation file at all");
    }

    /**
     * A field can be indexed without ever getting a vector. Its entry is still written, so it still has to be
     * read to the byte — the reader just keeps nothing for it, since there is no cluster data to point at.
     */
    @Test
    void constructor_readsMetaFile_whenFieldIsEmpty() throws Exception {
        openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry().vectorCount(0).centroidCount(0));
    }

    @Test
    void constructor_throws_whenEntryNamesAFieldTheSegmentDoesNotHave() {
        final int unknown = FIELD_NUMBER + 7;
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry(), unknown)
        );
        assertTrue(e.getMessage().contains("Invalid field number: " + unknown), e.getMessage());
    }

    @Test
    void constructor_throws_whenDimensionDisagreesWithFieldInfo() {
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry().dimension(DIMENSION + 1))
        );
        assertTrue(e.getMessage().contains("has dimension " + DIMENSION), e.getMessage());
        assertTrue(e.getMessage().contains("entry says " + (DIMENSION + 1)), e.getMessage());
    }

    @Test
    void constructor_throws_whenSimilarityDisagreesWithFieldInfo() {
        // FieldInfo says L2; the entry says cosine.
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(
                floatVectorField(DIMENSION, VectorSimilarityFunction.EUCLIDEAN),
                validEntry().similarityFunction(ClusterANNFieldMetaEncoder.SIMILARITY_COSINE)
            )
        );
        assertTrue(e.getMessage().contains("uses " + VectorSimilarityFunction.EUCLIDEAN), e.getMessage());
        assertTrue(e.getMessage().contains("entry says " + VectorSimilarityFunction.COSINE), e.getMessage());
    }

    @Test
    void constructor_throws_whenFieldIsByteEncoded() {
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(vectorField(DIMENSION, VectorEncoding.BYTE, SIMILARITY), validEntry())
        );
        assertTrue(e.getMessage().contains("is encoded as " + VectorEncoding.BYTE), e.getMessage());
    }

    /** A field with an entry but no vectors at all: the entry cannot be describing this field. */
    @Test
    void constructor_throws_whenFieldHasNoVectorValues() {
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(vectorField(0, VectorEncoding.FLOAT32, SIMILARITY), validEntry())
        );
        assertTrue(e.getMessage().contains("carries no vector values"), e.getMessage());
    }

    /**
     * A vector field is single-valued, so more vectors than documents cannot be true. The reader is the only
     * place to catch it: the entry on its own has no idea how big the segment is.
     */
    @Test
    void constructor_throws_whenThereAreMoreVectorsThanDocuments() {
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry().vectorCount(MAX_DOC + 1))
        );
        assertTrue(e.getMessage().contains("has " + (MAX_DOC + 1) + " vectors"), e.getMessage());
        assertTrue(e.getMessage().contains("only " + MAX_DOC + " documents"), e.getMessage());
    }

    /** A corrupt entry surfaces as itself, not as the footer failure that follows it. */
    @Test
    void constructor_throws_whenEntryIsCorrupt() {
        final CorruptIndexException e = assertThrows(
            CorruptIndexException.class,
            () -> openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry().rotationId((byte) 9))
        );
        assertTrue(e.getMessage().contains("Unknown rotation: 9"), e.getMessage());
    }

    /** A constructor that fails must not leak the delegate it was handed. */
    @Test
    void constructor_closesDelegate_whenMetaCannotBeRead() throws Exception {
        assertThrows(
            CorruptIndexException.class,
            () -> openReader(floatVectorField(DIMENSION, SIMILARITY), validEntry().dimension(DIMENSION + 1))
        );
        verify(raw).close();
    }

    // ---------------------------------------------------------------- delegation

    @Test
    void checkIntegrity_delegates() throws Exception {
        reader.checkIntegrity();
        verify(raw).checkIntegrity();
    }

    @Test
    void getFloatVectorValues_delegates() throws Exception {
        final FloatVectorValues values = mock(FloatVectorValues.class);
        when(raw.getFloatVectorValues(FIELD)).thenReturn(values);
        assertSame(values, reader.getFloatVectorValues(FIELD));
    }

    @Test
    void getByteVectorValues_delegates() throws Exception {
        final ByteVectorValues values = mock(ByteVectorValues.class);
        when(raw.getByteVectorValues(FIELD)).thenReturn(values);
        assertSame(values, reader.getByteVectorValues(FIELD));
    }

    @Test
    void close_delegates() throws Exception {
        reader.close();
        verify(raw).close();
    }

    /**
     * The reader holds three files open for a rotated field, and every one of them has to be let go — a
     * {@link MockDirectoryWrapper} refuses to close while any is still open, so the leak is the assertion.
     */
    @Test
    void close_thenReleasesEveryDataFileItOpened() throws Exception {
        // given — its own directory, closed here rather than in tearDown, since that is what is being asserted
        MockDirectoryWrapper directory = new MockDirectoryWrapper(new Random(), new ByteBuffersDirectory());
        KNN1030ClusterANNVectorsReader rotated = openReader(
            directory,
            floatVectorField(DIMENSION, SIMILARITY),
            validEntry().rotationId(ROTATION_RANDOM_GAUSSIAN).clarOffset(512L).clarLength(64L),
            FIELD_NUMBER
        );

        // when
        rotated.close();

        // then
        directory.close();
    }

    @Test
    void searchByteVector_isUnsupported() {
        assertThrows(
            UnsupportedOperationException.class,
            () -> reader.search(FIELD, new byte[] { 1 }, mock(KnnCollector.class), mock(AcceptDocs.class))
        );
    }

    // ---------------------------------------------------------------- search

    /**
     * Every argument a search needs is refused rather than dereferenced. Asserts are off in production, so a null would
     * otherwise surface as an NPE from wherever it was first touched rather than naming what the caller left out.
     *
     * <p>{@code acceptDocs} is included: a query always has one, even for a match-all, because it carries the live docs.
     * That is not the same as its {@code bits()} being null, which <em>is</em> how match-all is expressed — see
     * {@code search_whenTheFilterAcceptsEverything_thenStillSearches}.
     */
    @Test
    void search_whenAnArgumentIsNull_thenThrows() {
        // given
        final KnnCollector collector = mock(KnnCollector.class);
        final AcceptDocs acceptDocs = mock(AcceptDocs.class);

        // when / then
        assertThrows(IllegalArgumentException.class, () -> reader.search(null, new float[] { 0.1f }, collector, acceptDocs));
        assertThrows(IllegalArgumentException.class, () -> reader.search(FIELD, (float[]) null, collector, acceptDocs));
        assertThrows(IllegalArgumentException.class, () -> reader.search(FIELD, new float[] { 0.1f }, null, acceptDocs));
        assertThrows(IllegalArgumentException.class, () -> reader.search(FIELD, new float[] { 0.1f }, collector, null));
    }

    /**
     * A match-all query arrives as an {@link AcceptDocs} whose {@code bits()} is null — the documented way Lucene says
     * "every document is accepted", and what an unfiltered query on a segment with no deletions produces. It has to
     * search, not be refused: the null argument check above must not reach through into the filter itself.
     */
    @Test
    void search_whenTheFilterAcceptsEverything_thenStillSearches() throws Exception {
        // given — bits() is null, which mock(AcceptDocs.class) returns by default
        final KnnCollector collector = mock(KnnCollector.class);

        // when / then — reaching the end without throwing is the assertion. The query is the field's own width: the
        // rotation is applied before the walk, and one of the wrong length is rejected before the filter is reached.
        reader.search(FIELD, new float[DIMENSION], collector, mock(AcceptDocs.class));
        verify(raw, never()).getRandomVectorScorer(any(String.class), any(float[].class));
    }

    /**
     * A field with no ClusterANN data is not searchable, and the reader says so by leaving the collector alone
     * rather than by throwing: an empty field is a valid field, it just has no vectors to offer.
     */
    @Test
    void search_whenTheFieldHasNoClusters_thenLeavesTheCollectorAlone() throws Exception {
        // given — an entry with no vectors, which the reader keeps nothing for
        KNN1030ClusterANNVectorsReader empty = openReader(
            floatVectorField(DIMENSION, SIMILARITY),
            validEntry().vectorCount(0).centroidCount(0)
        );
        final KnnCollector collector = mock(KnnCollector.class);

        // when
        empty.search(FIELD, query(), collector, mock(AcceptDocs.class));

        // then
        verifyNoInteractions(collector);
    }

    @Test
    void search_whenTheFieldIsNotInTheSegment_thenLeavesTheCollectorAlone() throws Exception {
        // given
        final KnnCollector collector = mock(KnnCollector.class);

        // when
        reader.search("no_such_field", query(), collector, mock(AcceptDocs.class));

        // then
        verifyNoInteractions(collector);
    }

    /**
     * A field that has clusters is walked end to end: plan, scan, collect. The hits themselves are not asserted —
     * this segment's data files are blank, so the ordinals and scores are whatever zeroed codes decode to — but
     * reaching the collector at all is what says the reader is wired to the planner and the searcher rather than
     * quietly returning nothing, which is indistinguishable from a query that matched no documents.
     */
    @Test
    void search_thenWalksTheProbedClustersIntoTheCollector() throws Exception {
        // given — the default reader's field does have clusters
        final KnnCollector collector = mock(KnnCollector.class);

        // when
        reader.search(FIELD, query(), collector, mock(AcceptDocs.class));

        // then
        verify(collector, atLeastOnce()).collect(anyInt(), anyFloat());
        verify(collector, atLeastOnce()).incVisitedCount(anyInt());
    }

    /**
     * Searching goes through the clusters, never around them: the flat delegate holds the full-precision vectors
     * for reranking and exact access, and quietly brute-forcing it instead would turn a broken ANN path into a
     * slow-but-passing one.
     */
    @Test
    void search_thenDoesNotBruteForceTheFlatDelegate() throws Exception {
        // given
        final KnnCollector collector = mock(KnnCollector.class);

        // when
        reader.search(FIELD, query(), collector, mock(AcceptDocs.class));

        // then
        verify(raw, never()).getRandomVectorScorer(any(String.class), any(float[].class));
    }

    // ---------------------------------------------------------------- helpers

    /**
     * A query of the field's own dimension. The values do not matter to these cases, but the width does: anything
     * narrower is rejected before the search gets as far as the behaviour being asserted.
     */
    private static float[] query() {
        float[] query = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            query[i] = 0.1f * (i + 1);
        }
        return query;
    }

    /**
     * An entry that agrees with {@link #floatVectorField} on everything the reader cross-checks.
     *
     * <p>{@code docBits} is set rather than left to the encoder's default, which is not a width any
     * {@code ScalarEncoding} has — fine for the cases that only parse the entry, but a search resolves the
     * encoding and would be refused before reaching what it means to assert.
     */
    private static ClusterANNFieldMetaEncoder validEntry() {
        return new ClusterANNFieldMetaEncoder().dimension(DIMENSION)
            .similarityFunction(ClusterANNFieldMetaEncoder.SIMILARITY_L2)
            .rotationId(ROTATION_NONE)
            .docBits(DOC_BITS);
    }

    private KNN1030ClusterANNVectorsReader openReader(FieldInfo info, ClusterANNFieldMetaEncoder entry) throws IOException {
        return openReader(info, entry, FIELD_NUMBER);
    }

    /**
     * Builds a one-field segment whose metadata file holds {@code entry} under {@code entryFieldNumber}, then
     * opens a reader over it. The entry's field number is separate from the {@link FieldInfo}'s own so a case
     * can write an entry for a field the segment does not have.
     */
    private KNN1030ClusterANNVectorsReader openReader(FieldInfo info, ClusterANNFieldMetaEncoder entry, int entryFieldNumber)
        throws IOException {
        // Wrapped rather than raw so an input the reader forgets to release is caught: a ByteBuffersDirectory closes
        // happily with files still open, which is what would let a leak go unnoticed.
        final MockDirectoryWrapper directory = new MockDirectoryWrapper(new Random(), new ByteBuffersDirectory());
        directory.setCheckIndexOnClose(false);   // these segments are hand-built metadata, not an index to check
        directories.add(directory);
        return openReader(directory, info, entry, entryFieldNumber);
    }

    /** The directory of the segment {@link #openReader} built last, for a case that asserts on the files themselves. */
    private Directory lastDirectory() {
        return directories.get(directories.size() - 1);
    }

    private KNN1030ClusterANNVectorsReader openReader(
        Directory directory,
        FieldInfo info,
        ClusterANNFieldMetaEncoder entry,
        int entryFieldNumber
    ) throws IOException {
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            null,
            SEGMENT,
            MAX_DOC,
            false,
            false,
            null,
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null
        );
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { info });
        final SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        writeMeta(state, entry, entryFieldNumber);

        // The reader slices each field's region out of these for any field that has clusters, so they have to be
        // long enough to hold the regions the entry describes. Nothing here reads them — Clusters only holds them
        // and clones per access — so blank bytes are enough.
        writeBlank(state, KNN1030ClusterANNVectorsFormat.POSTINGS_EXTENSION);
        writeBlank(state, KNN1030ClusterANNVectorsFormat.CENTROIDS_EXTENSION);

        // .clar exists only for a rotated field, which is what lets the unrotated cases assert the reader does not
        // go looking for it.
        if (entry.hasRotation()) {
            writeBlank(state, KNN1030ClusterANNVectorsFormat.ROTATION_EXTENSION);
        }

        // Registered rather than returned bare: a reader holds an input per data file for as long as it is open, and
        // tearDown is what releases them. Every case goes through here, so no case has to remember. A case that closes
        // it itself may still do so — closing twice is harmless.
        final KNN1030ClusterANNVectorsReader opened = new KNN1030ClusterANNVectorsReader(state, raw);
        readers.add(opened);
        return opened;
    }

    /**
     * A data file of blank bytes, closed with a real codec footer. The bytes themselves are never read — the reader
     * only slices regions out of these — but the footer is: {@code checkIntegrity} checksums each file end to end, so a
     * fixture without one would fail the check for the shape of the fixture rather than for anything the reader did.
     */
    private static void writeBlank(SegmentReadState state, String extension) throws IOException {
        String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        try (IndexOutput out = state.directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(new byte[DATA_FILE_BYTES], 0, DATA_FILE_BYTES);
            CodecUtil.writeFooter(out);
        }
    }

    private static void writeMeta(SegmentReadState state, ClusterANNFieldMetaEncoder entry, int entryFieldNumber) throws IOException {
        final String name = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN1030ClusterANNVectorsFormat.META_EXTENSION
        );

        try (IndexOutput out = state.directory.createOutput(name, IOContext.DEFAULT)) {
            CodecUtil.writeIndexHeader(
                out,
                KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                KNN1030ClusterANNVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            out.writeVInt(BLOCK_SIZE);
            out.writeInt(entryFieldNumber);
            entry.write(out);
            out.writeInt(-1);
            CodecUtil.writeFooter(out);
        }
    }

    private static FieldInfo floatVectorField(int dimension, VectorSimilarityFunction similarity) {
        return vectorField(dimension, VectorEncoding.FLOAT32, similarity);
    }

    private static FieldInfo vectorField(int dimension, VectorEncoding encoding, VectorSimilarityFunction similarity) {
        return new FieldInfo(
            FIELD,
            FIELD_NUMBER,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            dimension,
            encoding,
            similarity,
            false,
            false
        );
    }
}
