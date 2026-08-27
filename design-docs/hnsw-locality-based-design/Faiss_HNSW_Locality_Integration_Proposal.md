# Proposal: Integrating Faiss HNSW Graph Construction with the Locality-Ordered Quantized Store

**Status:** Draft for review
**Date:** 2026-08-22
**Scope:** Faiss SQ 1-bit HNSW only. Builds on Milestone 0 (the `.veqlo`/`.vemlo` writer + reader) and the Option-C scorer decision in `Implementation_Plan_SQ1bit_Locality.md` §1.6.1.

---

## 1. Goal

Wire the locality-ordered quantized store (`.veqlo`/`.vemlo`) into a real Faiss-HNSW vector format so that:
- the HNSW graph is built by native Faiss (as today), and
- the quantized records are physically laid out in a **graph-derived locality permutation** (neighbors co-located on 32 KB pages), and
- search scores against the permuted store via the translating scorer (Option C), with full-precision rescoring from the raw `.vec`.

Non-goals for this phase: multi-field support, non-SQ-1-bit encodings, the greedy page-assignment algorithm itself (assumed available as a separate component; a placeholder permutation is acceptable while wiring).

---

## 2. Current State

### 2.1 Milestone 0 (done)
- `LocalityOrderedQuantizedVectorsWriter` / `Reader` — writes/reads the two-file store (`.vemlo` metadata + `.veqlo` permuted records); `ordToPhysicalOrdMap` DirectWriter-packed; docId resolved via the raw flat reader.
- `LocalityOrderedQuantizedByteVectorValues` — original-ordinal addressed (maps to physical), plus a `null`-map identity (physical-addressed) view for the scorer.
- `PhysicalOrdinalTranslatingScorer` — bridges an original-ordinal HNSW graph to the physically-permuted store (Option C). Validated by byte-parity + SIMD/non-SIMD score-parity tests against Lucene104.
- The writer currently permutes with a **random shuffle placeholder** at `flush()` time.

### 2.2 Existing Faiss SQ format (reference)
`Faiss1040ScalarQuantizedKnnVectorsFormat` → `Faiss1040ScalarQuantizedKnnVectorsWriter`:
1. `flatVectorsWriter.flush() + finish() + close()` writes the flat quantized store (`.veq`/`.vemq` + raw `.vec`) via `KNN1040ScalarQuantizedVectorsFormat` (quantizes once, ordinal order).
2. Reopens a flat reader, `extractQuantizedByteVectorValues(...)`, and native Faiss builds the HNSW graph (`.faiss`) from those quantized values.
3. Merge follows the same pattern.

Graph nodes therefore correspond to **original (insertion-order) ordinals** in the flat store.

---

## 3. The Sequencing Constraint (the key design driver)

The `.veqlo` order **is** the greedy page-assignment permutation derived from the HNSW graph adjacency. So the permutation does not exist until the graph is built. The true build order is:

1. **Quantize** (once) → ordinal-order quantized store.
2. **Build the Faiss graph** from that store (native) — nodes are original ordinals.
3. **Extract adjacency → greedy page assignment → `physicalOrdinals`** permutation.
4. **Write `.veqlo`/`.vemlo`** by copying the already-quantized codes in permuted order.

Consequences:
- The `.veqlo` write is a **post-graph step orchestrated by the Faiss writer**, not a standalone `FlatVectorsWriter.flush()` with a random shuffle. The Milestone-0 standalone-flat-writer shape does not fit the real pipeline.
- Because we use **Option C** (original-ordinal graph + translating scorer), **the graph is not remapped** — no change to the `.faiss` graph ordinals. The reader translates original→physical at score time.

---

## 4. Proposed Architecture

A new `KnnVectorsFormat` (working name `Faiss1040LocalityScalarQuantizedKnnVectorsFormat`) mirroring `Faiss1040ScalarQuantizedKnnVectorsFormat`.

### 4.1 Writer flow (`flush`)
1. `addField` collects float vectors (via the KNN1040 flat writer, as today).
2. On `flush`:
   a. Write the **ordinal-order quantized store to a temp artifact** (see §5.2) via the KNN1040 flat writer — this store is needed to build the graph and to source the codes. Quantize **once**.
   b. Reopen a flat reader over that store; **build the Faiss graph** natively from the quantized values (reuse the existing `doBuildAndWriteIndex` path). → `.faiss`.
   c. Extract graph adjacency; run **greedy page assignment** → `physicalOrdinals` (placeholder permutation acceptable while wiring).
   d. Write `.veqlo`/`.vemlo` by **copying the quantized codes from the reopened flat reader in permuted order** (no re-quantization — see §5.1).
   e. Ensure exactly **one raw `.vec`** is kept for rescoring (see §5.3).
3. `merge` follows the same order: merged quantized store → graph → permutation → permuted copy.

### 4.2 Reader
- `LocalityOrderedQuantizedVectorsReader` (translating scorer) for the quantized store, plus
- the `.faiss` graph reader (original-ordinal), plus
- the raw `.vec` reader for full-precision rescoring (delegated, keyed on docId).

---

## 5. Key Decisions

### 5.1 Single quantization pass; locality writer copies a pre-quantized source
The locality writer should copy an already-quantized `QuantizedByteVectorValues` **in permuted order** (its original Milestone-0 source-based mode), rather than re-quantizing internally. Quantization is deterministic (byte-parity proven), so copying is exact and avoids a second pass.

### 5.2 Intermediate quantized store is a *temp* artifact, not write-then-delete
The ordinal-order quantized store is genuinely needed (graph build precedes the permutation), but it is **not** a final segment file. Produce it with `directory.createTempOutput(...)` (Lucene's scratch mechanism, as merge uses) so it is never tracked in `SegmentInfo`/CFS and is cleaned up — rather than writing real segment files and `deleteFile`-ing them (which fights Lucene's file tracking / integrity model).

### 5.3 Exactly one raw `.vec`
Both the KNN1040 flat writer and our locality writer wrap a `Lucene99FlatVectorsFormat` that writes a raw `.vec`. Running both naively **collides** on the same filename. Keep a single raw `.vec` (for rescore); the locality writer must not emit a second one when the pipeline already has one.

### 5.4 Option C — no graph remap
The graph stays in original-ordinal space; `PhysicalOrdinalTranslatingScorer` translates original→physical at score time. This preserves page locality (records co-located) without touching the `.faiss` graph, and keeps `ordToDoc`/iteration in original-ordinal space.

---

## 6. Final Segment File Set
- `.vemlo` — locality metadata (scalars, centroid, `ordToPhysicalOrdMap`).
- `.veqlo` — permuted quantized records.
- `.faiss` — native HNSW graph (original-ordinal).
- raw `.vec` — full-precision vectors (original order) for rescore.
- (temp) intermediate quantized store — created and discarded during build.

---

## 7. Open Questions / To Decide Before Coding
1. **Merge orchestration details** — same quantize→graph→permutation→permuted-copy order; confirm the reopen/extract points and temp-file lifecycle for merge.
2. **Where the permutation component plugs in** — resolved in §10: Java-allocated `int[N*M0]` filled by JNI → `computeOrdering(...)` (locked algorithm) → `physicalOrdinals` → permuted-copy step. Adjacency transfer is decided (§10.5); only the `LocalityPageOrdering` class boundary remains.
3. **Reuse vs. fork of `Faiss1040ScalarQuantizedKnnVectorsWriter`** — subclass/compose to reuse the native build, or a parallel writer? Prefer composition to avoid divergence.
4. **`count == 0` / empty-field** handling through the new pipeline.
5. **Rescore path** confirmation — docId join through the raw `.vec`, unchanged from today.

---

## 8. Risks
- **File collision** on raw `.vec` if two flat writers run (mitigated by §5.3).
- **Write-then-delete fragility** if the intermediate isn't a temp output (mitigated by §5.2).
- **Sequencing bugs** — writing `.veqlo` before the graph/permutation exists (the Milestone-0 flush shape must be replaced by post-graph orchestration).
- **Merge complexity** — the full pipeline runs per merged field; temp lifecycle and reopen must be correct.

---

## 9. Alternatives Considered
- **Run KNN1040 flat writer + locality writer as two independent writers, delete Lucene's files.** Rejected: double quantization, raw `.vec` collision, and write-then-delete of tracked segment files. The refined flow (§4, §5) keeps the single-quantization + graph-build reuse the idea intended, without those drawbacks.
- **Option A (physical-ordinal graph, remap `.faiss`).** Deferred: larger reader rework (`ordToDoc`/iterator in physical space) and a graph remap step. Option C delivers locality without remapping; A can supersede C later if search moves fully native (see `Implementation_Plan_SQ1bit_Locality.md` §1.6.1).

---

## 10. Ordering Flow — deriving the permutation from the graph

This is the core of the phase: given the built HNSW graph, produce the bijection
`physicalOrdinals[physicalPos] = originalOrdinal` over `[0, N)`. Nothing writes or scores here.

### 10.1 Locked decisions (validated by `Simulation_Results_Summary.md`, ~52% page reduction @ 1M)
- **Single-layer graph** (`assign_probas = {1.0}`); order over **layer 0** (the locality-relevant layer).
- **Affinity = shared-neighbor count** (a candidate's affinity to a page = number of its neighbors already on the page).
- **Greedy page-growing** with a **bucket queue** keyed by affinity (the `< 5 s @ 1M` variant; no full re-scan per pick).
- **Seeds = highest-degree unassigned node** (follows from single-layer — there are no upper-level hubs to seed from).
- **Expansion when 1-hop candidates are exhausted = `TWO_HOP`** (neighbors-of-neighbors); default and only strategy needed for v1.
- **Deterministic:** tie-break by node id so the ordering is reproducible.

### 10.2 Stages
1. **Extract layer-0 adjacency (Java-allocated, JNI-filled).** Java first learns `N` and the layer-0 max degree `M0 = 2*M` (small prior JNI call or from the index build params), allocates a single flat `int[N * M0]`, and passes it to JNI. Native copies each node's layer-0 neighbors into its `[node*M0, node*M0 + M0)` slot, padding empty slots with `-1`. Java owns the array lifetime (no native alloc/free). No CSR conversion — the greedy iterates the fixed-stride array directly, stopping at `-1`.
2. **Page capacity** — `pageCapacity = floor(32768 / recordSize)`, `recordSize = ceil(dim/8) + 16`.
3. **Greedy page assignment (pure Java)** — highest-degree seed → grow page by max shared-neighbor affinity (bucket queue) → `TWO_HOP` frontier expansion on exhaustion → assign physical positions sequentially per page → append disconnected / degree-0 nodes (all-`-1` slots) to trailing pages.
4. **Return** `physicalOrdinals` (physical→original). Writer consumes it; reader stores the inverse (`ordToPhysicalOrdMap`).

### 10.3 Component contract (the clean, testable seam)
```java
// Pure Java; no Faiss / JNI / codec dependency.
int[] computeOrdering(
    int[] neighbors,                  // flat layer-0 adjacency, fixed stride: node i's neighbors at [i*twoM, i*twoM+twoM), -1 = empty
    int twoM,                         // layer-0 max degree (M0 = 2*M), the stride
    int N, int pageCapacity)          // -> physicalOrdinals[physicalPos] = originalOrd (bijection over [0, N))
```
- Lives as a standalone class (working name `LocalityPageOrdering`) that the JNI-filled array feeds.
- **Invariants asserted in isolation** (synthetic fixed-stride adjacency, no Faiss): (1) output is a **bijection over `[0, N)`** (every ordinal placed exactly once — critical, or the writer/reader permutation breaks); (2) **intra-page neighbor ratio beats identity ordering** (the locality win, mirroring the sim's metric).

### 10.4 Where it runs
Both `flush` and `merge` build a graph, so Stages 1–3 run in both paths.

### 10.5 Adjacency transfer — decided
- **Java allocates `int[N * M0]`; JNI fills it** (per-node layer-0 neighbors into the `M0`-stride slot, `-1` for empties). The greedy consumes this array directly — **no CSR**. Java-owned memory, single bulk JNI fill (`SetIntArrayRegion` or `GetPrimitiveArrayCritical`).
- **Prerequisite:** a way to obtain `N` and `M0 = 2*M` before allocating (tiny JNI accessor or index params).
- **Memory note:** footprint is `N * M0 * 4` bytes, transient (≈128 MB at `N=1M, M=16`). Fixed stride always reserves `M0` per node even at lower actual degree; if `N`/`M` grow large, revisit with an off-heap `ByteBuffer`/`MemorySegment` or a compacted CSR. Heap `int[]` is fine for v1.
- Confirm the standalone `LocalityPageOrdering` class boundary and package.
