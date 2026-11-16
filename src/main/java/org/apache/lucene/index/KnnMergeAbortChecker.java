/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Field;

/**
 * Utility class for checking if a Lucene segment merge operation has been aborted.
 * Uses reflection to access the merge state from {@link ConcurrentMergeScheduler.MergeThread}.
 *
 * <p>This is used in faiss interrupt callback mechanism to be able to abort faiss index creation
 * when merge process is aborted by Lucene's {@link IndexWriter}.
 *
 * @since 1.0
 */
@Log4j2
public final class KnnMergeAbortChecker {

    // Lives in org.apache.lucene.index because ConcurrentMergeScheduler.MergeThread is protected; the k-NN plugin's
    // native (Faiss) index build and the ClusterANN codec writer both poll it from a merge thread.

    /** The reflected field for accessing merge state from MergeThread */
    private static Field mergeField;

    /** The name of the merge field in ConcurrentMergeScheduler.MergeThread */
    private static final String MERGE_FIELD_NAME = "merge";

    private static boolean hasMergeField;
    static {
        try {
            mergeField = ConcurrentMergeScheduler.MergeThread.class.getDeclaredField(MERGE_FIELD_NAME);
            mergeField.setAccessible(true);
            hasMergeField = true;
        } catch (NoSuchFieldException e) {
            hasMergeField = false;
            log.error("Not find merge field in MergeThread", e);
        }
    }

    private KnnMergeAbortChecker() {}

    /**
     * Checks if the current thread is a merge thread and if its merge operation has been aborted.
     *
     * <p>This method uses reflection to access the private merge field from
     * {@link ConcurrentMergeScheduler.MergeThread} and checks if the associated
     * {@link MergePolicy.OneMerge} has been aborted.
     *
     * @return {@code true} if the current thread is a {@link ConcurrentMergeScheduler.MergeThread}
     *         and its merge is aborted, {@code false} otherwise (including when not running
     *         in a merge thread context or when reflection access fails)
     *
     * @see ConcurrentMergeScheduler.MergeThread
     * @see MergePolicy.OneMerge#isAborted()
     */
    public static boolean isMergeAborted() {
        if (!hasMergeField) {
            return false;
        }
        Thread mergeThread = Thread.currentThread();
        if (mergeThread instanceof ConcurrentMergeScheduler.MergeThread) {
            try {
                MergePolicy.OneMerge merge = (MergePolicy.OneMerge) mergeField.get(mergeThread);
                boolean aborted = merge.isAborted();
                log.debug("Current Thread {} is aborted: {}", mergeThread.getName(), aborted);
                return aborted;
            } catch (RuntimeException e) {
                log.error("Runtime exception while checking merge abort status", e);
                return false;
            } catch (IllegalAccessException e) {
                log.error("IllegalAccessException for reflection merge field access", e);
                return false;
            }
        }
        return false;
    }

    /**
     * Stops the work running on this thread when its merge has been aborted. Off a merge thread this never throws.
     *
     * @throws MergePolicy.MergeAbortedException if the current thread's merge has been aborted
     */
    public static void ensureMergeNotAborted() throws MergePolicy.MergeAbortedException {
        if (isMergeAborted()) {
            throw new MergePolicy.MergeAbortedException("merge aborted");
        }
    }
}
