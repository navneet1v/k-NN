/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.reader;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.reader.block.scalar.SQScanContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ClusterFactoryTests {

    private static final int DIMENSION = 64;
    private static final int BLOCK_SIZE = 4;
    private static final int CENTROID_COUNT = 3;

    /** Cluster sizes per centroid: deliberately different, so a cluster taking the wrong one is visible. */
    private static final int[] CLUSTER_SIZES = { 10, 6, 3 };

    /** Postings start here within the field's region, and run for this many bytes each. */
    private static final long[] CLAP_CENTROID_OFFSETS = { 0L, 1000L, 2000L };
    private static final int[] CENTROID_LENGTHS = { 1000, 1000, 1000 };

    /**
     * Where this field's postings begin in the whole {@code .clap} file. Non-zero, and deliberately not applied to
     * the input below: the factory is handed the region already cut at this offset, so a factory adding it a second
     * time would slice past the postings it was given.
     */
    private static final long CLAP_OFFSET = 512L;

    /** Length of the field's region — the three postings and nothing else. */
    private static final int CLAP_LENGTH = 3000;

    /** Where each centroid region begins within the field's {@code .clac} region, the rotated one after the raw. */
    private static final long RAW_CENTROIDS_OFFSET = 64L;
    private static final long ROTATED_CENTROIDS_OFFSET = RAW_CENTROIDS_OFFSET + (long) CENTROID_COUNT * (DIMENSION + 1) * Float.BYTES;

    /** Norms that name their region, so a centroid read from the wrong one carries the wrong number. */
    private static final float RAW_NORM_BASE = 100f;
    private static final float ROTATED_NORM_BASE = 200f;

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- dispatch

    /**
     * The stored code width selects the encoding, so a field carrying 1 or 2 bits gets a cluster it can score.
     * Nothing exposes the resolved encoding, so what is observable here is that a cluster is produced at all.
     */
    @ParameterizedTest(name = "docBits {0}")
    @ValueSource(ints = { 1, 2 })
    void testCreate_whenQuantizerIsScalar_thenBuildsAClusterForThatWidth(int docBits) throws IOException {
        // given / when
        Cluster cluster = factory(fieldMeta(ClusterFactory.QUANTIZER_SQ, docBits)).create(0);

        // then
        assertEquals(CLUSTER_SIZES[0], cluster.size());
        assertEquals(0, cluster.ordinal());
    }

    /** Only scalar quantization exists so far; the other ids are in the layout but unimplemented. */
    @ParameterizedTest(name = "quantizerId {0}")
    @ValueSource(ints = { 1, 2, 7, -1 })
    void testCreate_whenQuantizerIsNotScalar_thenThrows(int quantizerId) {
        // given / when
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> factory(fieldMeta(quantizerId, 1)).create(0));

        // then
        assertTrue(e.getMessage().contains("Unsupported quantizerId: " + quantizerId), e.getMessage());
    }

    /**
     * A width no encoding uses cannot be served. Resolved by bit count rather than by a separate id, so there is no
     * way for the two to disagree — but a corrupt width still has to be rejected rather than mapped to a neighbour.
     */
    @ParameterizedTest(name = "docBits {0}")
    @ValueSource(ints = { 0, 3, 5, 9, -1 })
    void testCreate_whenCodeWidthHasNoEncoding_thenThrows(int docBits) {
        // given / when
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> factory(fieldMeta(ClusterFactory.QUANTIZER_SQ, docBits)).create(0)
        );

        // then
        assertTrue(e.getMessage().contains("Unsupported docBits: " + docBits), e.getMessage());
    }

    // ---------------------------------------------------------------- per-cluster inputs

    /**
     * A cluster's size comes from the field's own {@code clusterSizes}, indexed by the ordinal — the caller does not
     * pass it, so a cluster reading the wrong entry would parse its neighbour's header length.
     */
    @ParameterizedTest(name = "cluster {0}")
    @CsvSource({ "0, 10", "1, 6", "2, 3" })
    void testCreate_thenTakesTheClusterSizeFromTheFieldMetadata(int ordinal, int expectedSize) throws IOException {
        // given / when
        Cluster cluster = factory(fieldMeta(ClusterFactory.QUANTIZER_SQ, 1)).create(ordinal);

        // then
        assertEquals(expectedSize, cluster.size());
        assertEquals(ordinal, cluster.ordinal());
    }

    /**
     * Creating a cluster reads no centroid — the factory slices the region and hands out a supplier to resolve later,
     * and only a scan resolves it. Counted rather than assumed, since slicing a region and reading one look alike
     * from the outside.
     */
    @Test
    void testCreate_thenReadsNoCentroid() throws IOException {
        // given
        CountingInput centroids = new CountingInput(clac(), new int[1]);
        ClusterFactory factory = new ClusterFactory(fieldMeta(ClusterFactory.QUANTIZER_SQ, 1), clap(), centroids, null);

        // when
        Cluster cluster = factory.create(0);

        // then
        assertEquals(CLUSTER_SIZES[0], cluster.size());
        assertEquals(0, centroids.reads(), "a cluster that is never scanned must not read its centroid");
    }

    /**
     * The centroid region is sliced, so a {@code .clac} too short to hold the centroids the metadata claims fails
     * here — at open, naming the file — rather than at the first scan that seeks past its end.
     */
    @Test
    void testCreate_whenTheCentroidRegionRunsPastTheFile_thenThrows() throws IOException {
        // given — the region needs RAW_CENTROIDS_OFFSET + 3 records of 65 floats; the file stops well short
        IndexInput shortClac = open("short-clac", 128);

        // when / then
        assertThrows(
            Exception.class,
            () -> new ClusterFactory(fieldMeta(ClusterFactory.QUANTIZER_SQ, 1), clap(), shortClac, null),
            "a centroid region running past the end of the file cannot be sliced"
        );
    }

    // ---------------------------------------------------------------- slicing

    /**
     * The posting is sliced here, from the cluster's own offset within the region it was handed — so a factory that
     * added the field's {@code clapOffset} on top, or indexed the offsets by anything other than the ordinal, would
     * hand the cluster the wrong bytes. Asked for one byte past the last posting, the slice has to fail rather than
     * read on.
     */
    @Test
    void testCreate_whenTheFileEndsBeforeThePosting_thenThrows() throws IOException {
        // given — the region stops halfway through the third posting's span
        IndexInput shortClap = open("short-clap", 2500);

        // when / then
        assertThrows(
            Exception.class,
            () -> new ClusterFactory(fieldMeta(ClusterFactory.QUANTIZER_SQ, 1), shortClap, clac(), null).create(2),
            "a posting running past the end of the file cannot be sliced"
        );
    }

    /** Each cluster gets its own length, so a posting too short for its header fails rather than reading on. */
    @Test
    void testCreate_whenThePostingIsTooShortForItsHeader_thenThrows() throws IOException {
        // given — cluster 0 holds 10 entries, whose header alone needs 82 bytes
        ClusterANNFieldMeta fieldMeta = fieldMeta(ClusterFactory.QUANTIZER_SQ, 1, new int[] { 8, 1000, 1000 });

        // when / then
        assertThrows(Exception.class, () -> factory(fieldMeta).create(0));
    }

    // ---------------------------------------------------------------- centroid region

    /**
     * A rotated field scores against its centroids in the rotated space; an unrotated one has no such region, so the
     * raw centroids are the only ones there are. Both are located by the metadata, so the region a cluster reads is
     * decided by the rotation and nothing else — the norms name their region, so reading the wrong one is visible.
     */
    @ParameterizedTest(name = "rotated={0}")
    @ValueSource(booleans = { false, true })
    void testCreate_thenReadsCentroidsFromTheRegionTheRotationImplies(boolean rotated) throws IOException {
        // given
        ClusterFactory factory = new ClusterFactory(
            fieldMeta(ClusterFactory.QUANTIZER_SQ, 1, CENTROID_LENGTHS, rotated),
            clap(),
            centroidsWithBothRegions(),
            rotated ? open("clar", 64) : null
        );

        // when — preparing a scan is what reads the centroid, and it carries the norm it read
        SQScanContext context = (SQScanContext) factory.create(0).prepareScan(ScanParams.of(new float[DIMENSION]));

        // then
        float expectedNorm = rotated ? ROTATED_NORM_BASE : RAW_NORM_BASE;
        assertEquals(expectedNorm, context.centroidNormSq(), "centroid 0 came from the wrong region");
    }

    // ---------------------------------------------------------------- helpers

    /** One factory per field, as the reader builds it: it owns the data inputs and the centroid cursor. */
    private ClusterFactory factory(ClusterANNFieldMeta fieldMeta) throws IOException {
        return new ClusterFactory(fieldMeta, clap(), clac(), null);
    }

    /** Centroids are only read when a cluster is scanned, so an empty region is enough to build one. */
    private IndexInput clac() throws IOException {
        return open("clac", 4096);
    }

    /**
     * A {@code .clac} region holding the raw centroids and then the rotated ones, each record carrying a norm that
     * names its region — so a cursor pointed at the wrong one comes back with the other region's number.
     */
    private IndexInput centroidsWithBothRegions() throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput("clac", IOContext.DEFAULT)) {
            out.writeBytes(new byte[(int) RAW_CENTROIDS_OFFSET], 0, (int) RAW_CENTROIDS_OFFSET);
            writeCentroids(out, RAW_NORM_BASE);
            writeCentroids(out, ROTATED_NORM_BASE);
        }
        return directory.openInput("clac", IOContext.DEFAULT);
    }

    /** One region: {@code CENTROID_COUNT} records of {@code DIMENSION} floats, each followed by its norm. */
    private static void writeCentroids(IndexOutput out, float normBase) throws IOException {
        for (int ordinal = 0; ordinal < CENTROID_COUNT; ordinal++) {
            for (int i = 0; i < DIMENSION; i++) {
                out.writeInt(Float.floatToIntBits(ordinal + i / 100f));
            }
            out.writeInt(Float.floatToIntBits(normBase + ordinal));
        }
    }

    private static ClusterANNFieldMeta fieldMeta(int quantizerId, int docBits) {
        return fieldMeta(quantizerId, docBits, CENTROID_LENGTHS);
    }

    private static ClusterANNFieldMeta fieldMeta(int quantizerId, int docBits, int[] centroidLengths) {
        return fieldMeta(quantizerId, docBits, centroidLengths, false);
    }

    private static ClusterANNFieldMeta fieldMeta(int quantizerId, int docBits, int[] centroidLengths, boolean rotated) {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            DIMENSION,
            19,                                     // vectorCount, the sum of CLUSTER_SIZES
            CENTROID_COUNT,
            VectorSimilarityFunction.EUCLIDEAN,
            docBits,
            rotated ? ClusterANNFieldMeta.ROTATION_RANDOM_GAUSSIAN : ClusterANNFieldMeta.ROTATION_NONE,
            quantizerId,
            new byte[0],                            // quantizerParams, unused by a scalar field
            0L,                                     // clacOffset
            4096L,                                  // clacLength
            RAW_CENTROIDS_OFFSET,                   // clacCentroidsOffset
            rotated ? ROTATED_CENTROIDS_OFFSET : ClusterANNFieldMeta.NO_ROTATION,
            CLAP_OFFSET,
            3000L,                                  // clapLength
            CLAP_CENTROID_OFFSETS,
            centroidLengths,
            CLUSTER_SIZES,
            rotated ? 0L : ClusterANNFieldMeta.NO_ROTATION,   // clarOffset
            rotated ? 64L : ClusterANNFieldMeta.NO_ROTATION,  // clarLength
            null                                    // ordToDoc, which the factory never consults
        );
    }

    /** The field's region of {@code .clap}, as the reader slices it — long enough for the three postings. */
    private IndexInput clap() throws IOException {
        return open("clap", CLAP_LENGTH);
    }

    private IndexInput open(String name, int bytes) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(new byte[bytes], 0, bytes);
        }
        return directory.openInput(name, IOContext.DEFAULT);
    }

    /** Counts reads, so "create() reads nothing" is asserted rather than assumed. Clones and slices share the counter. */
    private static final class CountingInput extends FilterIndexInput {

        private final int[] reads;

        private CountingInput(IndexInput in, int[] reads) {
            super("counting(" + in + ")", in);
            this.reads = reads;
        }

        private int reads() {
            return reads[0];
        }

        @Override
        public byte readByte() throws IOException {
            reads[0]++;
            return in.readByte();
        }

        @Override
        public void readBytes(byte[] bytes, int offset, int length) throws IOException {
            reads[0]++;
            in.readBytes(bytes, offset, length);
        }

        @Override
        public void readFloats(float[] floats, int offset, int length) throws IOException {
            reads[0]++;
            in.readFloats(floats, offset, length);
        }

        @Override
        public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
            return new CountingInput(in.slice(sliceDescription, offset, length), reads);
        }

        @Override
        public CountingInput clone() {
            return new CountingInput(in.clone(), reads);
        }
    }
}
