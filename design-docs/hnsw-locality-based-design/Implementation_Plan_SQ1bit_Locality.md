# Locality-Aware Physical Layout for Faiss SQ 1-bit HNSW

## Context

**Problem:** The Faiss SQ 1-bit path stores quantized vectors (Lucene `.veq` file) in insertion order (ordinal 0, 1, 2, ...). HNSW graph neighbors are scattered across the file. When search traverses the graph, each hop reads from `base + ordinal * recordSize`, potentially hitting a different 32KB SSD page each time — high read amplification.

**Solution:** Physically order the quantized vector file so graph neighbors co-reside on the same 32KB page. Combined with flat single-layer search (CAGRA-style), this produces disk-friendly ANN with minimal page faults. Simulation on Cohere 768D/1M shows **~52% fewer pages touched** with unchanged recall (see `Simulation_Results_Summary.md`).

**Scope:** Faiss SQ 1-bit only. 32x compression. 32KB pages.

---

## Architecture

**Chosen approach: own the quantized write path + remap graph ordinals; no runtime indirection.**

The build produces the quantized vector file **once, already in permuted order** (rather than writing insertion-order first and reordering later). After the HNSW graph is built:
1. Extract Layer 0 adjacency lists + levels from native
2. Run greedy page-growing algorithm → permutation `perm[oldOrd] = newOrd` (and inverse `inv[newOrd] = oldOrd`)
3. Remap the native graph with the same permutation (`permute_entries` + permute id_map)
4. Write the quantized vector file in permuted order (physical position `newOrd` = old ordinal `inv[newOrd]`)
5. Write `.faiss` (single-layer graph)

No indirection at search time: `base + ordinal * recordSize` still works because graph ordinals **are** the physical positions in the vector file. A single invariant — the inverse permutation `inv` — ties the graph, the vector file, and the id_map together.

> The detailed write-path design (why we own it, the format constraint that forces it, and the data flow) is in §1.4–§1.6 below, and expanded in `Vector_File_Reordering_DeepDive.md`.

---

## Milestone 0 (first): Ordering-Parameterized Quantized Vector Store — ✅ IMPLEMENTED

Build and prove the write/read path **in isolation**, before any graph/page-assignment work. This de-risks the format and the ordering mechanics: once "given any ordering, write and read quantized vectors correctly" works, the real page permutation is just a different ordering array.

**Status:** Done. Three production classes + a round-trip proof test (3 tests passing).

**Delivered classes** (package `org.opensearch.knn.index.codec.locality`):
- `LocalityOrderedQuantizedVectorsWriter` — writes the `.veqo` file given a source + an `order` array.
- `LocalityOrderedQuantizedVectorsReader` — opens/verifies, exposes the values view + `docId(p)` + `originalOrdinal(p)`.
- `LocalityOrderedQuantizedByteVectorValues` — a `QuantizedByteVectorValues` over the record slice; pure `ord * recordSize` access; `getSlice()` returns the record region so the existing SIMD scorer path works unchanged.

**What was reused vs. new:**
- Reused: Lucene's already-quantized `QuantizedByteVectorValues` as the data source (no re-quantization), `CodecUtil` (header/footer + CRC), `IndexOutput`/`IndexInput`, `IndexFileNames`. (`MemorySegmentAddressExtractorUtil` is used later on the search side, not needed for the round-trip test.)
- New: the ordering-aware record layout + the in-file docId mapping + the values view.

**Ordering:** input array `order[physicalPos] = originalOrdinal`. For this milestone it's a **random shuffle** — the test vehicle that showcases docIds landing in shuffled physical positions and still resolving correctly. Later it becomes the greedy page permutation with no class changes.

**`.veqo` file layout (metadata first, then vectors):**
```
[CodecUtil index header]
[metadata: fieldNumber, dimension, count, codeLength, recordSize, centroidDp, similarityOrdinal,
           centroid (dim f32), docId map (count i32)]
[records (permuted): for physicalPos p -> record of order[p]]
           record = [packed code (ceil(dim/8) B)][lower f32][upper f32][addlCorr f32][compSum i32]  (LE)
[CodecUtil footer + CRC32]
```
Metadata precedes the records, so the reader parses everything in a single sequential pass from the header; the record region is `[dataOffset, fileLength − footerLength)`. No trailer/offset directory.

**Reader exposes:** `getQuantizedValues()` (→ `vectorValue`/`getCorrectiveTerms`/`getCentroid`/`getCentroidDP`/`ordToDoc`), `docId(physicalPos)`, `size()`.

**Proof test** (`LocalityOrderedQuantizedVectorsRoundTripTests`, uses a REAL Lucene-produced source): write with a shuffled ordering, read back, and for every physical position `p` assert the code is byte-identical to the source's `order[p]`, all four correction factors match, and `docId(p) == source.ordToDoc(order[p])`. Cases: shuffled/128-dim, identity/128-dim, shuffled/56-dim (non-multiple-of-8).

**Why first:** decouples format + ordering + docId correctness from the native graph work (§1.2/§1.3) and the page-assignment algorithm (§1.1). Those plug in afterward by supplying the real permutation as the `order` array.

**Deltas from the final design (intentional, for isolation):** docId mapping is stored **in-file** here (self-contained reader); the final design (§1.6) delegates docId resolution to the permuted FAISS `id_map`. Quantization is Lucene's (reused as source); owning quantization to drop the dead-weight `.veq` (§1.4/§1.5) is a later step.

---

## Phase 1 (MVP): Page Assignment + Permuted Vectors + Flat Graph

### 1.1 Greedy Page-Growing Algorithm

**Input:**
- Layer 0 adjacency lists (flat int[] + uniform stride)
- Per-node degree (derived from adjacency; used for seed ordering)
- `pageCapacity = 32768 / recordSize` (~292 for 768-dim)

**Output:** `int[] permutation` where `permutation[oldOrd] = newOrd`

**Algorithm:**
```
1. Identify seeds: highest-degree nodes first (single-layer graph → seed by degree, not level)
2. For each seed (hub), grow a page:
   a. page = {seed}, mark assigned
   b. candidate set = unassigned neighbors of page members
   c. While page not full AND candidates available:
      - For each candidate: affinity = |neighbors(candidate) ∩ page|
      - Pick max-affinity candidate, add to page, expand candidate set
      - If candidate set exhausted: apply expansion strategy (see below)
   d. Assign new ordinals sequentially
3. For remaining unassigned nodes:
   - Pick highest-degree unassigned node as seed
   - Repeat same page-growing logic
4. Return permutation
```

**Affinity function (simple start):** Count of shared neighbors between candidate and nodes already on the page.

**Candidate Expansion Strategies (implement all three, configurable):**

| Strategy | Behavior | Tradeoff |
|----------|----------|----------|
| `TWO_HOP` (default) | When 1-hop candidates exhausted, add 2-hop neighbors (neighbors-of-neighbors) to candidate set | Good locality, moderate build cost |
| `PARTIAL_PAGE` | Close page when no unassigned 1-hop neighbors remain, even if not full | Fastest build, may waste page capacity |
| `BFS_EXPAND` | BFS-style frontier expansion until page full or no reachable unassigned nodes | Best page utilization, highest build cost |

Start implementation with `TWO_HOP`, benchmark all three.

### 1.2 Graph Extraction from Native

After the graph is built, read the Layer 0 adjacency from the native `FaissSQHnsw.hnsw` (public `neighbors`, `offsets`, `levels`) and copy to Java for the page assigner. Single-layer means uniform stride (`2*M` slots/node), so the flat neighbor array + node count is sufficient.

### 1.3 Graph Remapping

Apply the permutation to the native graph using Faiss's built-in `HNSW::permute_entries(inv)` — it reorders per-node neighbor blocks, remaps neighbor values through the inverse mapping, and fixes the entry point in one pass. Additionally permute the `IndexBinaryIDMap.id_map` (`newIdMap[newOrd] = oldIdMap[inv[newOrd]]`) so result ordinals resolve to the correct docIds. (Confirmed feasible in `Faiss_SQ_Integration_Feasibility.md`; this replaces the originally-planned custom neighbor-rewrite.)

### 1.4 The Vector File Write Path (owned, write-once, permuted)

**Why we own the write path.** Lucene's SQ format cannot represent a locality-reordered file: its ordinal→docId mapping must be **monotonic** (dense `ord == doc`, or a sparse `DirectMonotonic` over increasing docIds). A locality permutation deliberately breaks docId ordering, so it cannot be encoded as a valid Lucene `.veq`/`.vemq`. We also cannot use Lucene's `sortMap` flush path — the permutation isn't known until after the graph is built, and that path still produces a monotonic mapping. Therefore we write our **own** quantized vector file, purely ordinal-indexed, with docId resolution delegated to the FAISS `id_map` (which we permute anyway).

**Owning quantization is safe and self-contained.** `OptimizedScalarQuantizer` is public and fully deterministic: given `(vector, centroid, similarityFunction, bits)` it produces byte-identical packed code + the four correction factors, with no hidden state. Write **order** does not affect **values** — only which ordinal a record lands at — so writing in permuted order is inherently safe. Critically, **the centroid does not need to match Lucene's**: it only has to be *consistent* between data quantization (build) and query quantization (search). Since we own both the write path and the metadata the query side reads (`getCentroid`/`getCentroidDP`), we compute a simple mean (L2-normalized for COSINE), store it, and use it on both sides. This removes any need to replicate Lucene's two-branch merge-centroid logic.

**File layout (our format — metadata first, then vectors; see Milestone 0 for the implemented layout):**
```
[CodecUtil index header — KNN codec name, version, segmentId, segmentSuffix]
[metadata: dimension, count, recordSize, centroid (dim × f32), centroidDp, similarityFunction]
[records, contiguous, in PERMUTED order: record at ordinal newOrd occupies newOrd * recordSize]
   record = [ packed 1-bit code (ceil(dim/8) B) ]
            [ lowerInterval (4B f32 LE) ]
            [ upperInterval (4B f32 LE) ]
            [ additionalCorrection (4B f32 LE) ]
            [ quantizedComponentSum (4B i32 LE) ]           recordSize = ceil(dim/8) + 16
[CodecUtil footer + CRC32]
```
Record layout is byte-identical to Lucene's `.veq` record, so the native SIMD offset arithmetic (`base + ord * recordSize`) is unchanged. The file is ordinal-indexed only — no ordToDoc inside (docId resolution comes from the FAISS `id_map`). Metadata precedes the records so the reader parses in one sequential pass and the record region is `[dataOffset, fileLength − footerLength)`.

### 1.5 Build Pipeline (owned write-once flow)

Float vectors are buffered in memory during indexing (the raw-float writer also produces the `.vec` full-precision file in original order). At flush / merge:

1. **Compute centroid** over the field's vectors (arithmetic mean; L2-normalize for COSINE). `centroidDp = dot(centroid, centroid)`.
2. **Quantize in memory** — for each vector (on a copy, since the quantizer subtracts the centroid in place), call `OptimizedScalarQuantizer.scalarQuantize(copy, scratch, bits=1, centroid)` then `packAsBinary` → packed code + 4 corrections.
3. **Build the native HNSW graph** from the in-memory quantized values (existing pass/add path, fed from our values instead of a Lucene reader). Single-layer (§1.7).
4. **Extract graph** (neighbors + levels) to Java.
5. **Compute permutation** via the greedy page assigner → `perm`, `inv`.
6. **Remap the native graph** with `inv` (`permute_entries`) and permute the `id_map` (`newIdMap[newOrd] = oldIdMap[inv[newOrd]]`).
7. **Write `.faiss`** — serialize the permuted single-layer graph (existing skip-storage write path).
8. **Write the quantized vector file once**, records emitted in permuted order (`for newOrd: emit record of inv[newOrd]`), followed by centroid/metadata and the inverse permutation.

No insertion-order quantized file is ever written — a single write, already permuted. The `.vec` full-precision file remains in original order for rescore (§1.7).

**Threshold:** apply locality reordering only when `totalLiveDocs > 10000` (same pattern as remote index build); below that, fall back to the existing insertion-order SQ path.

### 1.6 Search Path

- The searcher opens our quantized vector file, verifies the CodecUtil envelope, slices the record region, and extracts the base address via the existing `MemorySegmentAddressExtractorUtil` — exactly as it does for `.veq` today.
- **Scoring hot path is unchanged** — native SIMD does `base + ordinal * recordSize`; only the file/base pointer differs. The query is quantized with the centroid read from our metadata (consistent with build).
- **ordToDoc** uses the permuted FAISS `id_map`, already the mechanism in the memory-opt path (`OrdinalTranslatedKnnCollector` + `scorer::ordToDoc`).
- **Entry points:** Phase 1 uses CAGRA-style random entry (existing `RandomEntryPointsKnnSearchStrategy`); Phase 2 replaces with hub-based entry.
- **Backward compatibility:** pre-locality segments (standard Lucene `.veq`/`.vemq`) route through the existing path unchanged; the codec detects which format a segment used and dispatches accordingly.

#### 1.6.1 Graph ordinal space vs. scorer addressing — options (decision record)

**Problem.** The `.veqo` records are stored in **physical (permuted)** order for page locality, but the SIMD scorer (`KNN1040ScalarQuantizedVectorScorer` → `BulkSimdRandomVectorScorer`) addresses the record region by **ordinal arithmetic** — native `score(id)`/`bulkScore(ids)` compute `base + id * recordSize` from the slice base and never route through `LocalityOrderedQuantizedByteVectorValues.vectorValue()` or `ordToPhysicalOrdMap`. So the scorer implicitly assumes **graph ordinal == physical record position**. The HNSW graph's ordinal space and the scorer's addressing must therefore agree, or scores read the wrong records.

Three ways to reconcile this:

| Option | Graph ordinal space | Reader addressing | Scorer | Trade-off |
|---|---|---|---|---|
| **A. Physical-ordinal graph** | physical (graph remapped to physical positions) | reader addressed by **physical** ordinal: `ordToDoc(physicalOrd)` needs physical→docId; `iterator()` yields ascending docs with `index()=physicalOrd` (needs `sortedOrds`) | stock SIMD, no translation (`ordToPhysicalOrdMap` disappears from the read path) | Canonical locality design (this is what §1.3 "Graph Remapping" and the Faiss-native §1.6 assume). Fastest scoring, no per-hop indirection. **Cost:** largest reader rework + a graph remap step. |
| **B. Original-ordinal graph, no SIMD** | original | reader addressed by **original** ordinal (as built) | fallback scorer that scores via `vectorValue(originalOrd)` (which maps through `ordToPhysicalOrdMap`) | Minimal reader changes, correct. **Cost:** gives up the native bulk-SIMD fast path — real latency hit; defeats the perf goal. |
| **C. Original-ordinal graph + translating scorer** ✅ chosen (Lucene-HNSW path) | original | reader addressed by **original** ordinal (as built); `ordToDoc`/`iterator` unchanged | stock SIMD wrapped by a thin translator that maps each graph ordinal `originalOrd → ordToPhysicalOrdMap[originalOrd]` before the native `score`/`bulkScore` | Reader untouched, keeps SIMD. **Cost:** one `int[]` lookup per scored node (cheap), a small wrapper class, and bulk prefetch is not yet optimal for the permuted layout (see note). |

**Chosen: Option C** for the current search path (Lucene HNSW over the flat reader — see §1.6 answers: Q1=Lucene HNSW, Q2=original-ordinal graph, Q4=no merge scorer supplier). Rationale: it delivers page locality **and** the SIMD fast path without reworking the reader's `ordToDoc`/iterator (which stay in original-ordinal space, matching a standard, unremapped Lucene HNSW graph). Implemented as `LocalityOrderedQuantizedVectorsReader.PhysicalOrdinalTranslatingScorer` (translates `score`/`bulkScore` ordinals only; `ordToDoc`/`maxOrd`/`getAcceptOrds` inherit original-ordinal semantics from the values).

**Prefetch (C) is correct.** The delegate's bulk path prefetches via `PrefetchableVectorValuesHelper.doPrefetch`, which addresses `getSlice()` by **raw ordinal arithmetic** (`ord × stride`, the same space as the native SIMD read) — it does **not** route through the mapping `LocalityOrderedQuantizedByteVectorValues.prefetch`. Because the wrapper passes it the already-translated **physical** ordinals, prefetch warms exactly the records the native `score(physical)` reads. (Requires the values to be a `HasIndexSlice`, which it is via `QuantizedByteVectorValues.getSlice()`; the per-record stride is `getVectorByteLength()`, matching Lucene's own `OffHeapScalarQuantizedVectorValues`.)

**Relationship to Option A:** the Faiss-native design in §1.3/§1.6 (graph remapped to physical, `base + ordinal*recordSize` directly) is Option A. If the search path later moves to Faiss-native, A supersedes C and `ordToPhysicalOrdMap` moves from a runtime scorer indirection to a build-time graph remap.

### 1.7 Full-Precision Rescoring (`.vec`)

Rescoring is a **separate flow keyed on docIds, not ordinals** — the ANN pass emits docIds (physical ordinal → docId via the id_map), and the rescore step re-scores those docIds against full-precision vectors. So the locality reordering does not touch the rescore path at all: `.vec` stays in **original** order (written by the raw-float writer) and is read by docId, exactly as today. No inverse permutation is stored — the vector reordering and the rescore flow are decoupled by the docId join key. This also avoids reordering the large full-precision `.vec` (≈3 KB/vector at 768-dim), which buys nothing since rescore reads are scattered regardless.

### 1.8 Single-Layer Graph

The graph is built single-layer (all nodes at level 0) by forcing the level distribution at construction (`assign_probas = {1.0}`), so every node reserves exactly one `2*M` neighbor block (uniform stride). This aligns with the existing `FaissCagraHNSW` read path:
- `maxLevel = 1`, `entryPoint = -1`
- `RandomEntryPointsKnnSearchStrategy` provides entry points (Phase 1)

At read time the searcher uses the existing flat-search path. See `Faiss_SQ_Integration_Feasibility.md` for the native details (single-layer forcing, `permute_entries`, id_map permute).

---

## Phase 2: Hub-Based Entry Points

### 2.1 Hub Selection at Build Time

After page assignment, identify K hubs (highest-degree nodes in the single-layer graph). Store their:
- New ordinals (positions in the permuted vector file)
- Original float vectors (for query-to-hub distance)

### 2.2 Hub Storage

Append a trailer to the `.faiss` file after the standard HNSW section:
```
"LHUB" (4B magic)
int numHubs
int[] hubNewOrdinals (numHubs entries)
float[][] hubVectors (numHubs * dimension floats)
```

### 2.3 Hub-Based Entry Point Strategy

**New class:** `org.opensearch.knn.memoryoptsearch.faiss.locality.HubBasedEntryPointStrategy`

At query time:
1. Compute distance from query to each hub vector (tiny — e.g., 32 hubs)
2. Pick top-K closest hubs as entry points
3. Return as `KnnSearchStrategy.Seeded` (replaces `RandomEntryPointsKnnSearchStrategy`)

---

## Phase 3: Boundary Replication (SOAR-Inspired)

### 3.1 Identify Boundary Nodes

After page assignment, for each node:
```
externalRatio = |neighbors NOT on same page| / |total neighbors|
```
Nodes with `externalRatio > 0.6` are candidates.

### 3.2 Replicate into Neighboring Pages

For each boundary node B on page P:
- Find page Q with most neighbors of B
- If Q has capacity: replicate B's record into Q
- Assign replica ordinal (N, N+1, ...) in the vector file after all primary vectors
- Rewrite graph: edges from Q's nodes pointing to B now point to replica ordinal

### 3.3 ID Map Extension

Replica ordinals N..N+R-1 map to the same Lucene doc IDs as their originals. Results are de-duplicated at collection time.

---

## Component Summary (design-level; class breakdown deferred)

**Java build side**
- Page-growing assigner (produces `perm`/`inv`) + expansion-strategy option (TWO_HOP / PARTIAL_PAGE / BFS_EXPAND)
- Quantization orchestrator — centroid + `OptimizedScalarQuantizer` over buffered floats (§1.5)
- Owned quantized vector-file writer — permuted records + centroid/meta + inverse permutation + CodecUtil envelope (§1.4)
- Codec wiring — route SQ-locality fields to the owned writer/reader; keep the raw-float writer for `.vec`

**Native / JNI**
- Extract graph data (neighbors + levels) from `FaissSQHnsw.hnsw`
- Remap graph — thin wrapper over built-in `HNSW::permute_entries(inv)` **plus** permute the `IndexBinaryIDMap.id_map`
- Force single-layer at `initFaissSQIndex` (`assign_probas = {1.0}`)

**Search side**
- Owned vector-file reader — CodecUtil verify, slice + base address via `MemorySegmentAddressExtractorUtil`; exposes `getCentroid`/`getCentroidDP`
- Format detection/dispatch — locality file vs. legacy Lucene `.veq`/`.vemq`
- Hub-based entry point selection (Phase 2)

> Concrete class names and signatures are intentionally deferred until the design is locked.

---

## Verification Plan

1. **Page-assignment unit test:** small synthetic graph (100 nodes, M=16) — all ordinals assigned, permutation bijective, intra-page edge ratio measured.

2. **Quantization parity test:** for a set of vectors, confirm our owned quantization (centroid + `OptimizedScalarQuantizer`) yields the same codes/corrections whether written in original or permuted order, and that query-side quantization uses the same stored centroid (ADC score parity).

3. **End-to-end integration test:** build an SQ 1-bit locality index, verify:
   - Quantized vector file written once, in permuted order (no insertion-order file left behind)
   - Graph neighbor IDs valid (within `[0, N)`) and consistent with the vector file (spot-check node → its vector)
   - Search recall matches the non-locality baseline
   - Rescore returns correct full-precision vectors via the inverse permutation

4. **Page-locality metric:** average distinct pages touched per query (before vs. after) — expect the ~52% reduction seen in simulation.

5. **Benchmarks:** Cohere-1M / SIFT-1M with SQ 1-bit — Recall@10/100 (unchanged), page reads per query, build-time overhead.

---

## Related Work

### DiskANN / Vamana
Separates a memory-resident routing structure from disk-resident full-precision vectors. Uses a single flat graph (Vamana) with long-range edges added during construction. Relevant parallel: their flat-graph + entry-point design mirrors our single-layer + hub approach. Difference: we derive the flat graph from a standard HNSW layer 0 rather than building a purpose-designed Vamana graph.

### SOAR (Google ScaNN)
Selective replication of boundary points to reduce cross-partition crossings. Directly inspires our Phase 3 boundary replication — replicate a small fraction of high-external-edge nodes into neighboring pages.

### d-HNSW: A High-performance Vector Search Engine on Disaggregated Memory (arXiv 2603.13591)
Targets RDMA disaggregated memory (network round-trips), not SSD/disk. Its primary problem differs from ours, but two techniques map onto our design:

1. **Gap reservation for in-place updates (future enhancement).**
   d-HNSW reserves *internal gaps* and *shared overflow gaps* so a partition can still be fetched in one RDMA_READ after many inserts. If we later move to a strict-page-padded layout (deferred — padding is incompatible with the dense-bijection `permute_entries`; MVP uses **dense packing**), those padded slots could be **reserved for future inserts into that page's neighborhood** rather than dead zero-fill — a new vector in page P's cluster occupies P's reserved slots instead of forcing a full rebuild. This is a concrete answer to Design Critique #4 (segment merge invalidates all locality work). **Status: future enhancement, tied to the strict-page-padded / co-located-page format phase.**

2. **Meta-index routing over partition centroids (Phase 2 fallback).**
   d-HNSW builds a small "meta-HNSW" over partition centroids to route a query to the correct partition. This is a more principled version of our hub-based entry point. Our simulation shows a single degree-selected hub already matches Faiss-native recall (0.92) at 1M, so this is an *optimization*, not a requirement. **Status: keep in reserve for Phase 2 if single-hub routing degrades at 10M+.**

**What we explicitly did NOT adopt from d-HNSW:**
- **Balanced clustering partitioning (K-Means++ + capacity queue):** This is a *vector-space* partition (co-locate embedding-space neighbors), not a *graph-traversal-locality* partition (co-locate nodes visited together). Our design philosophy — from the original RFC — is that graph structure drives layout, not external clustering. This is the "graph partitioning" approach we explicitly rejected.
- **RDMA doorbell batching, pipelined RDMA/compute overlap, epoch shadow-copy rebuild:** All specific to disaggregated memory; not relevant to SSD paging.