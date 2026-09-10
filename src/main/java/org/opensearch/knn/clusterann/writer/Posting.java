package org.opensearch.knn.clusterann.writer;

import org.apache.lucene.util.FixedBitSet;

/**
 * One cluster's members, in the order they will be written.
 *
 * <p><b>Ordered by {@code ‖c−v‖}, ascending.</b> That is not presentation — it is what the read side's geometric
 * pruning rests on. A scan walks a posting outwards from the centroid and stops when the nearest remaining vector
 * cannot beat the threshold; if the order were anything else, stopping early would silently drop true neighbours.
 * Nothing on the read side checks it, so this is where it has to be true.
 *
 * @param centroidOrdinal the cluster these vectors belong to
 * @param ordinals global vector ordinals, ascending by {@code distances}
 * @param distances {@code ‖c−v‖} for each entry, ascending, parallel to {@code ordinals}
 * @param soar one bit per entry, set when this entry is a SOAR copy rather than a primary member
 */
public record Posting(int centroidOrdinal, int[] ordinals, float[] distances, FixedBitSet soar) {

    public Posting {
        if (ordinals.length != distances.length) {
            throw new IllegalArgumentException("ordinals has " + ordinals.length + ", distances has " + distances.length);
        }
        if (soar.length() < ordinals.length) {
            throw new IllegalArgumentException("soar holds " + soar.length() + " bits for " + ordinals.length + " entries");
        }
        for (int i = 1; i < distances.length; i++) {
            if (distances[i] < distances[i - 1]) {
                throw new IllegalArgumentException(
                    "distances must ascend: distances[" + i + "]=" + distances[i] + " < " + distances[i - 1]
                );
            }
        }
    }

    public int size() {
        return ordinals.length;
    }

    /** Bytes this posting's header occupies, which follows from its size alone — the reader derives it the same way. */
    public long headerBytes() {
        return headerBytes(size());
    }

    /**
     * Bytes the header of a posting of {@code size} entries occupies: the ordinals, the SOAR bitset, and the
     * distances column.
     */
    public static long headerBytes(int size) {
        return (long) size * Integer.BYTES + (size + 7) / 8 + (long) size * Float.BYTES;
    }
}
