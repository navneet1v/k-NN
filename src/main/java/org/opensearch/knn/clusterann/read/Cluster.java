package org.opensearch.knn.clusterann.read;

import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.read.orchestration.ScanBuffers;
import org.opensearch.knn.clusterann.read.orchestration.ScanContext;

import javax.annotation.Nullable;
import java.io.IOException;

/**
 * One IVF cluster, hiding how its vectors are stored and scored. Callers see only ordinals and scores;
 * the storage and scoring scheme never surface. It hands out a {@link PostingScorer} over its own
 * vectors, and scoring stays inside that scorer because it is inseparable from how the vectors are
 * stored. Query-varying inputs arrive as data in {@link ScanParams}.
 *
 * <p>Scanning is two steps: {@link #prepareScan} turns a query into the form this cluster scores against,
 * and {@link #scorer} walks the posting with it. Splitting them lets the prepared form be reused across
 * scorers and lets a scan be abandoned before paying for preparation. Nothing is read until one of the
 * two is called. One instance per query; not thread-safe.
 *
 * <p>{@link Accountable} because cost depends on whether the cluster was scanned.
 */
public interface Cluster extends Accountable {

    /** This cluster's centroid ordinal within the field (identity; useful for bookkeeping/logging). */
    int ordinal();

    /** Number of vectors in this cluster (primary + SOAR). */
    int size();

    /**
     * A cursor over this cluster's postings that scores as it advances.
     *
     * <p>{@code scanBuffers} belong to the scan, not to this cluster, so the scorer returned here must be drained
     * before another cluster of the same scan is scored — a scan visits its clusters one at a time (see
     * {@link org.opensearch.knn.clusterann.read.orchestration.ClusterScan}).
     *
     * @param scanContext a query prepared by this cluster's own {@link #prepareScan}; a context from
     *     another cluster family is rejected rather than misread.
     * @param acceptedOrds ordinals to score, or {@code null} for all; honoured before scoring, not after.
     * @param scanBuffers buffers for this field, from
     *     {@link org.opensearch.knn.clusterann.read.ClusterFactory#newScanBuffers()}, reused by every
     *     cluster of the scan; buffers from another family are rejected rather than misread. {@code null} for a
     *     family that keeps none.
     */
    PostingScorer scorer(ScanContext scanContext, Bits acceptedOrds, @Nullable ScanBuffers scanBuffers) throws IOException;

    /**
     * Turn a query into the form this cluster scores against.
     *
     * @param scanParams the query and per-query knobs to honour
     * @return a context for {@link #scorer}, valid only for this cluster unless the context itself reports
     *     {@link ScanContext#isQueryScoped()}
     */
    ScanContext prepareScan(ScanParams scanParams) throws IOException;
}
