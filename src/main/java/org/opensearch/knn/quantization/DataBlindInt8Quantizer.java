/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization;

import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * Data-blind absolute int8 quantization (the IVFaster {@code Int8Quantizer} scheme), shared by the write and
 * read paths so a query is encoded exactly like the codes it is scored against — the same guarantee
 * {@link ScalarQuantizers} gives the optimized-scalar family.
 *
 * <p><b>Data-blind.</b> The quantizer is not trained or calibrated on the corpus and does not depend on any
 * centroid: a vector is quantized <em>absolutely</em>, in the rotated space the caller supplies. The only
 * per-vector quantity is a trivial {@code maxAbs/127} scale, not a learned parameter. (Variance across
 * dimensions is flattened upstream by a Hadamard rotation, which is what lets one fixed grid serve every
 * dimension; this class assumes its input is already rotated.)
 *
 * <p><b>Encoding.</b> For a rotated vector {@code v}: {@code scale = maxAbs/127}; each coordinate is
 * {@code q = clip(round(v[d]/scale), -127, 127)} stored as the unsigned byte {@code q + 128}. Three
 * corrections travel with the code: the {@code scale}, the signed component sum {@code Σq} (for the offset
 * correction below), and {@code ‖v‖²} (for the Euclidean transform).
 *
 * <p><b>Scoring.</b> With a query encoded the same way, the unsigned-code dot {@code Σ(qᵤ·dᵤ)} is corrected
 * back to the signed dot of the original rotated vectors:
 * <pre>
 *   signedDot = Σ(qᵤ·dᵤ) − OFFSET·Σq_query − OFFSET²·dim − OFFSET·Σq_doc
 *   dot       = signedDot · qScale · dScale                       (≈ ⟨v_query, v_doc⟩, rotation-preserved)
 * </pre>
 * then the metric transform. {@code OFFSET=128}, so {@code OFFSET²·dim = 16384·dim}. No centroid term appears
 * — the scheme is absolute, and the centroid is used only to route to this cluster, never to score within it.
 *
 * <p>Stateless and immutable; one instance may serve every field, cluster and query.
 */
public final class DataBlindInt8Quantizer {

    /** Unsigned-byte offset: a signed code {@code q ∈ [-127,127]} is stored as {@code q + 128 ∈ [1,255]}. */
    public static final int OFFSET = 128;

    private static final int MAX_SIGNED = 127;
    private static final long OFFSET_SQ = (long) OFFSET * OFFSET; // 16384

    private final int dimension;

    public DataBlindInt8Quantizer(final int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive, got " + dimension);
        }
        this.dimension = dimension;
    }

    public int dimension() {
        return dimension;
    }

    /**
     * The corrections stored alongside one vector's code: the per-vector {@code scale}, the signed component
     * sum {@code Σq}, and the squared norm {@code ‖v‖²} of the (rotated) vector.
     *
     * @param scale  {@code maxAbs/127}
     * @param sum    {@code Σ q} over signed codes
     * @param sqNorm {@code ‖v‖²} of the rotated vector
     */
    public record Int8Code(float scale, int sum, float sqNorm) {
    }

    /**
     * The query-only terms of the score, computed once per query and reused for every candidate in the scan.
     * The offset residual {@code OFFSET·Σq_query + OFFSET²·dim} and the query scale and squared norm do not
     * depend on the doc, so hoisting them here turns the per-candidate work into a single multiply-add plus the
     * metric transform.
     *
     * @param scale          the query's {@code maxAbs/127}
     * @param sqNorm         the query's {@code ‖q‖²} (Euclidean transform only)
     * @param offsetResidual {@code OFFSET·Σq_query + OFFSET²·dim}, the query-dependent part of the signed-dot
     *     correction
     */
    public record QueryTerms(float scale, float sqNorm, long offsetResidual) {
    }

    /**
     * Precompute the {@link QueryTerms} for one query from its code corrections and the field dimension. Call
     * once per query; pass the result to {@link #score} for every candidate.
     */
    public QueryTerms queryTerms(final Int8Code query) {
        final long offsetResidual = (long) OFFSET * query.sum() + OFFSET_SQ * dimension;
        return new QueryTerms(query.scale(), query.sqNorm(), offsetResidual);
    }

    /**
     * Quantize an already-rotated vector into {@code codeOut} (one unsigned byte per dimension) and return its
     * corrections. {@code codeOut} must have length ≥ {@link #dimension()}.
     *
     * <p>A zero vector (or all-zero after rotation) has no scale; it is encoded at the offset (every code
     * {@code = OFFSET}, i.e. signed 0) with {@code scale = 1} and {@code sum = 0}, which the scorer handles
     * without a special case.
     */
    public Int8Code encode(final float[] rotated, final byte[] codeOut) {
        float maxAbs = 0f;
        double sqNorm = 0;
        for (int d = 0; d < dimension; d++) {
            final float x = rotated[d];
            final float a = Math.abs(x);
            if (a > maxAbs) {
                maxAbs = a;
            }
            sqNorm += (double) x * x;
        }
        if (maxAbs == 0f) {
            for (int d = 0; d < dimension; d++) {
                codeOut[d] = (byte) OFFSET;
            }
            return new Int8Code(1f, 0, (float) sqNorm);
        }
        final float scale = maxAbs / MAX_SIGNED;
        final float inv = MAX_SIGNED / maxAbs;
        int sum = 0;
        for (int d = 0; d < dimension; d++) {
            int q = Math.round(rotated[d] * inv);
            if (q > MAX_SIGNED) {
                q = MAX_SIGNED;
            } else if (q < -MAX_SIGNED) {
                q = -MAX_SIGNED;
            }
            sum += q;
            codeOut[d] = (byte) (q + OFFSET);
        }
        return new Int8Code(scale, sum, (float) sqNorm);
    }

    /**
     * Similarity estimate of a query against one doc code, from the unsigned-code dot, the query's precomputed
     * {@link QueryTerms}, and the doc's corrections. No centroid term — the scheme is absolute. Static and
     * allocation-free: everything query-invariant was hoisted into {@code query}, so the per-candidate cost is
     * one multiply-add plus the metric transform.
     *
     * @param unsignedDot {@code Σ(qᵤ·dᵤ)} from {@link Int8DotProduct#unsignedDot}
     * @param query       the query's precomputed terms from {@link #queryTerms}
     * @param docScale    the doc's {@code scale}
     * @param docSum      the doc's signed component sum {@code Σq}
     * @param docSqNorm   the doc's {@code ‖v‖²}
     * @param sim         the field's similarity function
     * @return the metric-appropriate similarity (higher is nearer), matching Lucene's score ranges
     * @throws IllegalStateException if {@code sim} is {@code DOT_PRODUCT} (ClusterANN uses
     *     {@code MAXIMUM_INNER_PRODUCT}), matching the optimized-scalar scorer
     */
    public static float score(
        final long unsignedDot,
        final QueryTerms query,
        final float docScale,
        final int docSum,
        final float docSqNorm,
        final VectorSimilarityFunction sim
    ) {
        final long signedDot = unsignedDot - query.offsetResidual() - (long) OFFSET * docSum;
        final double dot = (double) signedDot * query.scale() * docScale;
        switch (sim) {
            case EUCLIDEAN: {
                double s = query.sqNorm() + docSqNorm - 2.0 * dot;
                if (s < 0) {
                    s = 0;
                }
                return (float) (1.0 / (1.0 + s));
            }
            case MAXIMUM_INNER_PRODUCT:
                return dot >= 0 ? (float) (dot + 1.0) : (float) (1.0 / (1.0 - dot));
            case COSINE:
                return (float) Math.max(0.0, (1.0 + dot) / 2.0);
            case DOT_PRODUCT:
            default:
                throw new IllegalStateException("ClusterANN does not support " + sim + "; use MAXIMUM_INNER_PRODUCT");
        }
    }
}
