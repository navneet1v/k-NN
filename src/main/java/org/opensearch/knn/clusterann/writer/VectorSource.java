package org.opensearch.knn.clusterann.writer;

import java.io.IOException;
import java.util.List;

/**
 * The vectors of one field, addressed by ordinal.
 *
 * <p>Random access rather than iteration, because a posting is written in distance order and a vector may appear in
 * two postings: the order bytes are written in is not the order the vectors arrive in. Where the vectors actually
 * live — a flush buffer on the heap, a temporary file during a merge — stays behind this interface, so nothing
 * downstream has to care which it is.
 *
 * <p>Ordinals run {@code 0 .. size()-1} and are assigned in document order, which is what makes
 * {@link #docId(int)} non-decreasing and lets {@code ord → doc} be stored as monotonic deltas.
 */
public interface VectorSource {

    int size();

    int dimension();

    /**
     * The vector at {@code ord}. The array may be reused between calls, so consume it before asking for another.
     */
    float[] vector(int ord) throws IOException;

    /** The document this ordinal belongs to. Non-decreasing in {@code ord}. */
    int docId(int ord);

    /**
     * A source over vectors already on the heap — the flush case, where they are buffered anyway.
     *
     * @param docIds one per vector, ascending; {@code null} when ordinals are document ids
     */
    static VectorSource fromList(List<float[]> vectors, int[] docIds, int dimension) {
        return new VectorSource() {
            @Override
            public int size() {
                return vectors.size();
            }

            @Override
            public int dimension() {
                return dimension;
            }

            @Override
            public float[] vector(int ord) {
                return vectors.get(ord);
            }

            @Override
            public int docId(int ord) {
                return docIds == null ? ord : docIds[ord];
            }
        };
    }
}
