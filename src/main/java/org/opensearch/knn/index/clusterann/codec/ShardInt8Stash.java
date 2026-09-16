/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.clusterann.codec;

import java.util.concurrent.ConcurrentHashMap;

/**
 * POC shard-level int8 rerank stash.
 *
 * <p>During phase 1 (per-segment coarse-Hamming shortlist), each segment stashes the int8 code +
 * corrections for the docs it emits, keyed by global docId. After the query layer merges segments
 * to a global top-k, phase 2 rescopes those docs with int8 by reading the stash — no random
 * cross-segment file access needed, since the codes were carried at scan time.
 *
 * <p>POC ONLY: a per-query static map, cleared at query start. Not production-safe under concurrent
 * queries on the same JVM (fine for the single-threaded benchmark). Enabled via
 * {@code -Dclusterann.thermo.scope=shard}.
 */
public final class ShardInt8Stash {

    public static final boolean SHARD_SCOPE = "shard".equalsIgnoreCase(System.getProperty("clusterann.thermo.scope"));

    /**
     * Global key -> int8 code. The key is {@code (segmentToken << 32) | localDocId} so the SAME
     * segment-local docId in two different segments does not collide (the earlier POC keyed on the
     * bare local docId and, with dozens of segments at 1M, every segment overwrote the others'
     * codes — recall collapsed to ~0.18). Phase 1 (scanner) and phase 2 (query) MUST build the key
     * with the identical segmentToken so the write and the read land on the same entry.
     */
    private static final ConcurrentHashMap<Long, byte[]> CODES = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Long, float[]> CORR = new ConcurrentHashMap<>();

    private ShardInt8Stash() {}

    /** Compose the global stash key from a per-segment token and the segment-local docId. */
    public static long key(int segmentToken, int localDocId) {
        return ((long) segmentToken << 32) | (localDocId & 0xFFFFFFFFL);
    }

    public static void clear() {
        CODES.clear();
        CORR.clear();
    }

    public static void put(long key, byte[] code, float scale, int sum, float norm) {
        byte[] copy = new byte[code.length];
        System.arraycopy(code, 0, copy, 0, code.length);
        CODES.put(key, copy);
        CORR.put(key, new float[] { scale, (float) sum, norm });
    }

    public static byte[] code(long key) {
        return CODES.get(key);
    }

    public static float scale(long key) {
        float[] c = CORR.get(key);
        return c == null ? 0f : c[0];
    }

    public static int sum(long key) {
        float[] c = CORR.get(key);
        return c == null ? 0 : (int) c[1];
    }

    public static float norm(long key) {
        float[] c = CORR.get(key);
        return c == null ? 0f : c[2];
    }
}
