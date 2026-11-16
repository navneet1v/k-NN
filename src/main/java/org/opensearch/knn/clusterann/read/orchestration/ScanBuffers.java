/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

/**
 * Scratch one scan reuses across every cluster it probes. Family-specific and opaque, like {@link ScanContext}: the
 * cluster casts it to the type its own storage needs and rejects anything else.
 *
 * <p>Everything in here is sized from the field's geometry, which all of a field's clusters share, so one instance
 * serves a whole probe list. That is the point: a probe would otherwise allocate a block's worth of buffers, which
 * dominated the search path's allocation.
 *
 * <p>Mutable and single-threaded, which is sound only because one scan visits its clusters one at a time — see
 * {@link ClusterScan}. Each query gets its own instance, so concurrency across queries is unaffected.
 */
public interface ScanBuffers {}
