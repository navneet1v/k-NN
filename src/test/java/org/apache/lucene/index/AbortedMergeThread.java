/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

/** Runs test code on a real {@code ConcurrentMergeScheduler} merge thread whose merge has already been aborted. */
public final class AbortedMergeThread {

    /** Work to run on the merge thread. */
    public interface Body {
        void run() throws Exception;
    }

    private AbortedMergeThread() {}

    /**
     * Runs {@code body} on an aborted merge thread and rethrows whatever it threw.
     *
     * @throws Exception what {@code body} threw, if anything
     */
    public static void run(final Body body) throws Exception {
        final MergePolicy.OneMerge merge = oneMerge();
        merge.setAborted();
        final AtomicReference<Throwable> thrown = new AtomicReference<>();
        try (ConcurrentMergeScheduler scheduler = new ConcurrentMergeScheduler()) {
            final Thread thread = scheduler.new MergeThread(new NoMergeSource(), merge) {
                @Override
                public void run() {
                    try {
                        body.run();
                    } catch (final Throwable t) {
                        thrown.set(t);
                    }
                }
            };
            thread.start();
            thread.join();
        }
        final Throwable t = thrown.get();
        if (t instanceof Exception e) {
            throw e;
        }
        if (t instanceof Error e) {
            throw e;
        }
    }

    private static MergePolicy.OneMerge oneMerge() throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig())) {
                final Document doc = new Document();
                doc.add(new StringField("id", "1", Field.Store.NO));
                writer.addDocument(doc);
                writer.commit();
            }
            return new MergePolicy.OneMerge(SegmentInfos.readLatestCommit(dir).asList());
        }
    }

    private static final class NoMergeSource implements MergeScheduler.MergeSource {
        @Override
        public MergePolicy.OneMerge getNextMerge() {
            return null;
        }

        @Override
        public void onMergeFinished(final MergePolicy.OneMerge merge) {}

        @Override
        public boolean hasPendingMerges() {
            return false;
        }

        @Override
        public void merge(final MergePolicy.OneMerge merge) {}
    }
}
