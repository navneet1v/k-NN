/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.write.postings;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

/**
 * Writes one cluster's posting region into {@code .clap}. An implementation owns the cluster-level layout for
 * one storage family: the member header (ordinals, the secondary-assignment bitset, the parallel distances)
 * and how the member vectors are laid down after it. The concrete payload byte format — the code blocks —
 * stays inside the payload writer the implementation composes, so turning the block format into a row format
 * never touches this layout.
 *
 * <p>One instance per cluster, chosen by a factory keyed on the {@code .clam} quantizer scheme — the
 * write-side analogue of the read-side cluster factory — and bound to its centre at construction. The output
 * is passed to {@link #write} rather than held, so an implementation touches the shared {@code .clap} output
 * only for the duration of that one call. One cluster's region is a single {@link #write} call.
 */
public interface ClusterWriter {

    /**
     * Writes this cluster's posting region to {@code out} at its current position and returns where it landed.
     *
     * @param out     the {@code .clap} output this cluster's region is appended to
     * @param vectors the member vectors, read by position; position {@code i} is {@code members.ordinals()[i]}
     * @param members the member ordinals, distances, and secondary flags, written in array order
     * @return the region's start offset and byte length in {@code .clap}
     */
    ClusterRegion write(IndexOutput out, FloatVectorValues vectors, ClusterMembers members) throws IOException;

    /**
     * One cluster's members — its primary members and its SOAR (spill) members combined into a single run — as
     * three parallel arrays indexed by member position: {@code ordinals[i]} at squared-L2 {@code distances[i]}
     * to the centre, with {@code secondary[i]} marking it a spill (non-primary) member — the {@code .clap}
     * secondary bitset. A cluster with no spill-ins has an all-{@code false} {@code secondary}.
     *
     * @param ordinals  member vector ordinals, the cluster's primary and SOAR (spill) members combined
     * @param distances squared-L2 distance of each member to the centre, parallel to {@code ordinals}
     * @param secondary whether each member is a spill (non-primary) member, parallel to {@code ordinals}
     */
    record ClusterMembers(int[] ordinals, float[] distances, boolean[] secondary) {
        public ClusterMembers {
            if (distances.length != ordinals.length || secondary.length != ordinals.length) {
                throw new IllegalArgumentException(
                    "ordinals, distances, and secondary must be parallel (equal length), got "
                        + ordinals.length
                        + ", "
                        + distances.length
                        + ", "
                        + secondary.length
                );
            }
        }
    }

    /**
     * Where one cluster's posting region landed in {@code .clap}.
     *
     * @param startOffset the region's absolute start offset
     * @param length      the region's byte length
     */
    record ClusterRegion(long startOffset, long length) {
    }
}
