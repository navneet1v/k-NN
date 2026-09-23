/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.util.HashMap;
import java.util.Map;

/**
 * Flush-path integration tests for {@link KNN1030ClusterANNVectorsWriter}. Unlike the component test (which
 * force-merges and asserts search through the flat delegate), these keep the flushed segment intact
 * ({@link NoMergePolicy}) and verify what the flush actually wrote: the four sidecars exist and are
 * checksum-valid, and the {@code .clam} carries a well-formed entry for the segment's single vector field —
 * with rotation recorded for EUCLIDEAN and absent for inner product. Also covers an index-sorted segment
 * (flush with a doc map) and rejection of a multi-vector-field segment.
 */
public class KNN1030ClusterANNVectorsWriterFlushTest extends LuceneTestCase {

    private static final String L2_FIELD = "l2_field";
    private static final String IP_FIELD = "ip_field";
    private static final int DIMENSION = 4;
    private static final int NUM_DOCS = 40;
    /** The {@code .clam} sentinel offset/length for an unrotated field (mirrors {@code ClusterANNFieldMeta}'s private value). */
    private static final long NO_ROTATION = -1L;

    /** A EUCLIDEAN field is rotated and lands as a valid {@code .clam} entry pointing into the sidecars. */
    public void testFlush_writesSidecars_andClamEntry() throws Exception {
        try (Directory dir = newDirectory()) {
            indexField(dir, /* sorted */ false, L2_FIELD, VectorSimilarityFunction.EUCLIDEAN);

            final String segment = baseName(dir, ".clam");
            assertSidecarIntact(dir, segment + ".clac");
            assertSidecarIntact(dir, segment + ".clap");
            assertSidecarIntact(dir, segment + ".clar");

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FieldInfos fieldInfos = reader.leaves().get(0).reader().getFieldInfos();
                final int l2Number = fieldInfos.fieldInfo(L2_FIELD).number;

                final Map<Integer, ClusterANNFieldMeta> entries = readClam(dir, segment + ".clam");
                assertEquals("one entry for the segment's single vector field", 1, entries.size());

                final ClusterANNFieldMeta l2 = entries.get(l2Number);
                assertEquals(DIMENSION, l2.dimension());
                assertEquals(NUM_DOCS, l2.vectorCount());
                assertEquals(VectorSimilarityFunction.EUCLIDEAN, l2.similarityFunction());
                assertTrue("below threshold clusters into a single centroid", l2.centroidCount() >= 1);
                assertEquals("centroid bookkeeping is sized by centroidCount", l2.centroidCount(), l2.clusterSizes().length);
                assertTrue("EUCLIDEAN field is rotated", l2.hasRotation());
                assertTrue("rotated field records a .clar offset", l2.clarOffset() >= 0);
                assertTrue("rotated field records rotated centroids", l2.clacRotatedCentroidsOffset() >= 0);

                // Follow the .clam offsets into the actual regions to prove the orchestrator threaded them correctly.
                assertOrdToCentroidRegion(dir, segment + ".clac", l2);
                assertClapRegionInBounds(dir, segment + ".clap", l2);
            }
        }
    }

    /** An inner-product field is not rotated: its {@code .clam} entry records no rotation and no {@code .clar} is written. */
    public void testFlush_innerProductField_notRotated() throws Exception {
        try (Directory dir = newDirectory()) {
            indexField(dir, /* sorted */ false, IP_FIELD, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);

            final String segment = baseName(dir, ".clam");
            assertSidecarIntact(dir, segment + ".clac");
            assertSidecarIntact(dir, segment + ".clap");
            assertSidecarAbsent(dir, segment + ".clar");

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FieldInfos fieldInfos = reader.leaves().get(0).reader().getFieldInfos();
                final Map<Integer, ClusterANNFieldMeta> entries = readClam(dir, segment + ".clam");

                final ClusterANNFieldMeta ip = entries.get(fieldInfos.fieldInfo(IP_FIELD).number);
                assertEquals(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, ip.similarityFunction());
                assertFalse("inner-product field is not rotated", ip.hasRotation());
                assertEquals("no rotation matrix for an unrotated field", NO_ROTATION, ip.clarOffset());
                assertEquals(NO_ROTATION, ip.clacRotatedCentroidsOffset());

                assertOrdToCentroidRegion(dir, segment + ".clac", ip);
                assertClapRegionInBounds(dir, segment + ".clap", ip);
            }
        }
    }

    /** An index-sorted segment flushes through the doc-map remap path and still writes a valid entry. */
    public void testFlush_indexSortedSegment_writesEntry() throws Exception {
        try (Directory dir = newDirectory()) {
            indexField(dir, /* sorted */ true, L2_FIELD, VectorSimilarityFunction.EUCLIDEAN);

            final String segment = baseName(dir, ".clam");
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FieldInfos fieldInfos = reader.leaves().get(0).reader().getFieldInfos();
                final Map<Integer, ClusterANNFieldMeta> entries = readClam(dir, segment + ".clam");

                final ClusterANNFieldMeta l2 = entries.get(fieldInfos.fieldInfo(L2_FIELD).number);
                assertEquals("every doc's vector is clustered, regardless of sort order", NUM_DOCS, l2.vectorCount());
                assertTrue(l2.centroidCount() >= 1);
            }
        }
    }

    /** ClusterANN does not support DOT_PRODUCT, and says so at field registration rather than at scoring time. */
    public void testFlush_dotProductField_throwsUnsupported() throws Exception {
        try (Directory dir = newDirectory()) {
            final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setUseCompoundFile(false)
                .setMaxBufferedDocs(NUM_DOCS + 1);
            final Throwable thrown = expectThrows(Throwable.class, () -> {
                try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(IP_FIELD, vector(0), VectorSimilarityFunction.DOT_PRODUCT));
                    writer.addDocument(doc);
                    writer.commit();
                }
            });
            assertTrue("a DOT_PRODUCT field is rejected as unsupported", hasCause(thrown, UnsupportedOperationException.class));
        }
    }

    /** ClusterANN clusters at most one vector field per segment; a second field is rejected as unsupported. */
    public void testFlush_multipleVectorFields_throwsUnsupported() throws Exception {
        try (Directory dir = newDirectory()) {
            final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setUseCompoundFile(false)
                .setMaxBufferedDocs(NUM_DOCS + 1);
            final Throwable thrown = expectThrows(Throwable.class, () -> {
                try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(L2_FIELD, vector(0), VectorSimilarityFunction.EUCLIDEAN));
                    doc.add(new KnnFloatVectorField(IP_FIELD, vector(0), VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
                    writer.addDocument(doc);
                    writer.commit();
                }
            });
            assertTrue("a multi-vector-field segment is rejected as unsupported", hasCause(thrown, UnsupportedOperationException.class));
        }
    }

    /**
     * Merging exercises the merge clustering path: the raw vectors are merged through the flat delegate (one
     * write), then read back through a reopened flat reader and clustered — so a merged segment gets a full
     * {@code .clam} entry, not just an empty frame. Uses a single vector field (the merge path assumes one) and
     * two source segments so {@code forceMerge(1)} actually merges.
     */
    public void testMerge_singleField_clustersMergedVectors() throws Exception {
        // A plain directory and a serial merge scheduler keep this deterministic: MockDirectoryWrapper's random
        // faults and the concurrent merge scheduler's threads would otherwise flake the reopen-during-merge path.
        try (Directory dir = new ByteBuffersDirectory()) {
            final Codec codec = new ClusterANN1030TestCodec();
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler());
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(L2_FIELD, vector(i), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                    if (i == NUM_DOCS / 2) {
                        writer.commit(); // a second segment, so the forceMerge below has something to merge
                    }
                }
                writer.commit();
                writer.forceMerge(1);
            }

            final String segment = baseName(dir, ".clam");
            assertSidecarIntact(dir, segment + ".clac");
            assertSidecarIntact(dir, segment + ".clap");
            assertSidecarIntact(dir, segment + ".clar");

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("force-merged into one segment", 1, reader.leaves().size());
                final FieldInfos fieldInfos = reader.leaves().get(0).reader().getFieldInfos();
                final Map<Integer, ClusterANNFieldMeta> entries = readClam(dir, segment + ".clam");

                final ClusterANNFieldMeta l2 = entries.get(fieldInfos.fieldInfo(L2_FIELD).number);
                assertEquals("every doc is clustered on merge, across both source segments", NUM_DOCS, l2.vectorCount());
                assertTrue("below threshold clusters into at least one centroid", l2.centroidCount() >= 1);
                assertTrue("EUCLIDEAN field is rotated on the merge path too", l2.hasRotation());
                assertOrdToCentroidRegion(dir, segment + ".clac", l2);
                assertClapRegionInBounds(dir, segment + ".clap", l2);
            }
        }
    }

    private void indexField(final Directory dir, final boolean sorted, final String field, final VectorSimilarityFunction similarity)
        throws Exception {
        final Codec codec = new ClusterANN1030TestCodec();
        // Keep the sidecars as separate files (LuceneTestCase randomizes compound files, which would pack them into .cfs),
        // and buffer all docs into a single flushed segment (randomized maxBufferedDocs could otherwise split them,
        // and NoMergePolicy would leave the read below inspecting just one partial segment).
        IndexWriterConfig iwc = newIndexWriterConfig().setCodec(codec)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(NUM_DOCS + 1);
        if (sorted) {
            iwc = iwc.setIndexSort(new Sort(new SortField("sortkey", SortField.Type.LONG)));
        }
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(field, vector(i), similarity));
                if (sorted) {
                    doc.add(new NumericDocValuesField("sortkey", NUM_DOCS - i)); // reverse of insertion order
                }
                writer.addDocument(doc);
            }
            writer.commit();
        }
    }

    /** True if {@code throwable} or any of its causes is an instance of {@code type}. */
    private static boolean hasCause(final Throwable throwable, final Class<? extends Throwable> type) {
        for (Throwable cause = throwable; cause != null; cause = cause.getCause()) {
            if (type.isInstance(cause)) {
                return true;
            }
        }
        return false;
    }

    /** The shared base name of the single flushed segment, found from its {@code .clam} file. */
    private static String baseName(final Directory dir, final String extension) throws Exception {
        for (final String file : dir.listAll()) {
            if (file.endsWith(extension)) {
                return file.substring(0, file.length() - extension.length());
            }
        }
        throw new AssertionError("no " + extension + " file was written");
    }

    /**
     * Follows {@code .clam}'s {@code clacOffset} into {@code .clac} region 1 (the per-ordinal {@code ordToCentroid}
     * map) and checks it holds exactly {@code vectorCount} ids, each a valid centroid — proving the orchestrator's
     * {@code .clac} offset points where the reader would look, and that region 2 begins after region 1. The
     * centroid sub-offsets are region-relative (the reader slices the field's region at {@code clacOffset} and
     * reads them within it), so region 2 sits at {@code vectorCount} ints past region 1's start.
     */
    private static void assertOrdToCentroidRegion(final Directory dir, final String file, final ClusterANNFieldMeta meta) throws Exception {
        try (IndexInput in = dir.openInput(file, IOContext.DEFAULT)) {
            assertEquals(
                "region 2 (centroids) follows region 1 (ordToCentroid) within the field's region",
                (long) meta.vectorCount() * Integer.BYTES,
                meta.clacCentroidsOffset()
            );
            assertTrue("centroids sub-offset within the clac region", meta.clacCentroidsOffset() < meta.clacLength());
            in.seek(meta.clacOffset());
            for (int ord = 0; ord < meta.vectorCount(); ord++) {
                final int centroid = in.readInt();
                assertTrue(
                    "ordToCentroid[" + ord + "]=" + centroid + " in [0," + meta.centroidCount() + ")",
                    centroid >= 0 && centroid < meta.centroidCount()
                );
            }
        }
    }

    /** Checks {@code .clam}'s postings region sits inside {@code .clap} and the region-relative per-centroid offsets ascend from 0 within it. */
    private static void assertClapRegionInBounds(final Directory dir, final String file, final ClusterANNFieldMeta meta) throws Exception {
        try (IndexInput in = dir.openInput(file, IOContext.DEFAULT)) {
            assertTrue("clap region within file", meta.clapOffset() >= 0 && meta.clapOffset() + meta.clapLength() <= in.length());
            long previous = 0L;
            for (final long centroidOffset : meta.clapCentroidOffsets()) {
                assertTrue(
                    "region-relative centroid offset ascends within the clap region",
                    centroidOffset >= previous && centroidOffset < meta.clapLength()
                );
                previous = centroidOffset;
            }
        }
    }

    /** Opens a sidecar and verifies its CodecUtil footer checksum over the whole file. */
    private static void assertSidecarIntact(final Directory dir, final String file) throws Exception {
        try (IndexInput in = dir.openInput(file, IOContext.READONCE)) {
            assertTrue(file + " must be non-empty", in.length() > 0);
            CodecUtil.checksumEntireFile(in);
        }
    }

    /** Asserts the segment wrote no such sidecar (e.g. an all-unrotated segment writes no {@code .clar}). */
    private static void assertSidecarAbsent(final Directory dir, final String file) throws Exception {
        for (final String existing : dir.listAll()) {
            assertNotEquals("expected no " + file, file, existing);
        }
    }

    /** Parses the {@code .clam} frame the reader consumes: header, block size, field entries, terminator, footer. */
    private static Map<Integer, ClusterANNFieldMeta> readClam(final Directory dir, final String file) throws Exception {
        final Map<Integer, ClusterANNFieldMeta> entries = new HashMap<>();
        try (ChecksumIndexInput meta = dir.openChecksumInput(file)) {
            CodecUtil.checkIndexHeader(
                meta,
                KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                KNN1030ClusterANNVectorsFormat.VERSION_START,
                KNN1030ClusterANNVectorsFormat.VERSION_CURRENT,
                readSegmentId(dir, file),
                ""
            );
            final int blockSize = meta.readVInt();
            for (int fieldNumber = meta.readInt(); fieldNumber != KNN1030ClusterANNVectorsFormat.NO_MORE_FIELDS; fieldNumber = meta
                .readInt()) {
                entries.put(fieldNumber, ClusterANNFieldMeta.read(meta, blockSize));
            }
            CodecUtil.checkFooter(meta);
        }
        return entries;
    }

    /** The segment id from the sidecar's own index header, so {@link CodecUtil#checkIndexHeader} can be satisfied. */
    private static byte[] readSegmentId(final Directory dir, final String file) throws Exception {
        try (ChecksumIndexInput in = dir.openChecksumInput(file)) {
            CodecUtil.checkHeader(
                in,
                KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                KNN1030ClusterANNVectorsFormat.VERSION_START,
                KNN1030ClusterANNVectorsFormat.VERSION_CURRENT
            );
            final byte[] id = new byte[org.apache.lucene.util.StringHelper.ID_LENGTH];
            in.readBytes(id, 0, id.length);
            return id;
        }
    }

    private static float[] vector(final int seed) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = (seed + 1) * 0.1f + i * 0.01f;
        }
        return vector;
    }
}
