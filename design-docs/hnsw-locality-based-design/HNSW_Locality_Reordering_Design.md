# HNSW Locality-Aware Vector Reordering for On-Disk k-NN

**Status:** Draft · **Feature branch:** `hnsw_locality`
**Scope:** Faiss Scalar-Quantized (SQ) on-disk k-NN in OpenSearch. Reorder the quantized
vector store so that graph-adjacent vectors are co-located on the same storage pages, cutting
read amplification and per-query page faults during on-disk search — with no change to recall.

---

## 1. TL;DR

On-disk k-NN keeps the quantized vectors and the HNSW graph on disk (memory-mapped) and pages
them in on demand during search. HNSW search is a graph walk: each hop reads the neighbor's
vector. If graph-adjacent vectors are scattered across the file, almost every hop faults a new
page, so a query touches far more distinct pages than the data it actually uses — **read
amplification**.

This feature computes, at index-build time, a **permutation of the vectors derived from the graph
structure** (a greedy, page-filling clustering seeded from graph hubs) and writes the quantized
store in that order. Graph neighbors land on the same or adjacent pages, so a query's working set
collapses onto far fewer pages.

**Results (100M vectors, 1024 dimensions; details in §10.2.1):**
- **Recall is unchanged** (recall@100 0.6644 vs 0.6647). Only the on-disk layout changes.
- **A typical query touches 37% fewer pages** (median across 10,000 queries), and every one of the 10,000 queries
  touched fewer pages. Read amplification falls from 141× to 82×.
- **When the cache is too small and pages keep getting evicted, the reordered index completes 50% more queries**,
  with 37% lower p99 latency and 22% fewer bytes read per query.
- **With everything already cached it is not slower.** Latency differences in that case were within the
  run-to-run variance between hosts.
- If the cache already holds every page a workload touches, total storage reads don't change: the saving is per
  query, so it matters when pages would otherwise have to be read again.
- **Indexing overhead negligible** (≤1%).

The benefit acts on the **graph/quantized read path only**. Full-precision rescore reads are
layout-invariant by design and are not (and cannot be) improved by this change.

---

## 2. Motivation & Problem

### 2.1 On-disk k-NN and the page-fault cost
In on-disk (memory-optimized) mode, the SQ-quantized vectors and the HNSW graph live in files that
are `mmap`ed; the OS/engine pages them in as the search touches them. The storage/cache system
moves data in fixed-size **pages/blocks** (e.g. 8 KB or 32 KB), never single records. A quantized
SQ 1-bit record is small — ~112 bytes for a 768-dim vector (binary code + correction factors), or
~144 bytes for 1024-dim — so a page holds hundreds of records.

### 2.2 Why the naïve layout is bad
HNSW search is a **beam walk over graph edges**. Starting from an entry point, it repeatedly visits
a node, reads its neighbors' vectors, computes distances, and moves toward the query. The set of
vectors a query reads is therefore the set of **graph nodes it traverses** — which are, by
construction, graph-adjacent.

But the default physical order of vectors is the **document/insertion order**, which is
uncorrelated with graph adjacency (a document's id has nothing to do with which vectors are its
nearest neighbors). So graph-adjacent nodes are scattered uniformly across the file. Each graph hop
lands on a random page → each hop faults a fresh page → the query reads many pages while *using*
only a few records from each.

### 2.3 Read amplification, defined
```
read amplification = dataReadBytes / dataUsedBytes
                   = (distinct pages touched × page size) / (distinct records used × record size)
```
With scattered layout, `distinct pages touched ≈ distinct records used` (each record on its own
page from the query's perspective), so amplification ≈ `page size / record size` — for 112-byte
records on 32 KB pages that is a ~290× ceiling. In practice a query visiting ~100k vectors touches
tens of thousands of distinct pages and reads hundreds of MB to use a few MB.

**Goal:** make graph-adjacent vectors share pages, so one page fault serves many of the hops a
query will actually take → fewer distinct pages per query → lower amplification, smaller working
set, fewer cold faults, lower tail latency.

---

## 3. Key Idea

Reorder the physical position (ordinal) of each vector in the quantized store so that **vectors
that are neighbors in the HNSW graph get nearby ordinals**, and hence share 32 KB pages. The
reordering is derived from the graph itself at build time and baked into a dedicated on-disk store;
the graph and query results are unchanged.

Two properties make this safe and cheap:
- **Distance-preserving:** reordering only relabels physical positions; the vectors, the graph
  edges, and therefore the search results are identical. Recall is unaffected.
- **Build-time only:** the permutation is computed once per segment during flush/merge, off the
  query hot path. Query time only pays a cheap ordinal translation.

---

## 4. Background

### 4.1 SQ on-disk storage layout
The Faiss SQ flat store holds all quantized data in one contiguous, ordinal-ordered buffer. Each
record is fixed-stride:
```
[ quantized code | lowerInterval f32 | upperInterval f32 | additionalCorrection f32 | quantizedComponentSum i32 ]
```
= `ceil(dim/8) + 16` bytes for SQ 1-bit (112 B @768-dim, 144 B @1024-dim). Vector `i` lives at
`base + i × recordSize`. This is the store the reordering permutes.

### 4.2 Full-precision `.vec` (rescore) is separate
On-disk k-NN optionally keeps the **full-precision** vectors (`.vec`) to *rescore* the top
candidates from the quantized search. These records are large (dim × 4 bytes: 3072 B @768-dim,
4096 B @1024-dim). **The `.vec` file is deliberately left in original order and is not reordered**
(see `Rejected_Ideas.md` for the analysis showing it is not worth reordering). Rescore reads are therefore
layout-invariant — a fact that matters for interpreting end-to-end latency (§8.4, §10.4).

### 4.3 Page/block model
The engine faults data in fixed pages. We analyze two sizes: **32 KB** (the block size assumed in
the codec's grouping) and **8 KB**. Smaller pages amplify less on scattered layouts (fewer wasted
bytes per fault) but cost more faults — both are reported.

---

## 5. Design Overview

Reordering touches two places: **indexing**, where we decide the order of the vectors on disk, and **search**,
where we read them back in that order. The graph, the document ids, the full-precision vectors and the query API
don't change.

### 5.1 A small example first

Say a segment has six vectors, A to F, and a disk page holds three quantized records. In the HNSW graph, A's
neighbours are B and C, and D's neighbours are E and F.

Stock layout writes records in insertion order, which has nothing to do with the graph:

```
page 0: [ A  D  B ]     page 1: [ F  C  E ]
```

A search that reaches A next scores B and C, so it has to read **both** pages. The reordered layout puts graph
neighbours next to each other:

```
page 0: [ A  B  C ]     page 1: [ D  E  F ]
```

Now expanding A only needs page 0. At 100M vectors the same idea means a typical query touches 37% fewer pages
(§10.2.1).

### 5.2 Indexing flow

This runs at flush and merge, for each segment of a field that uses the reordered format.

```
 raw vectors
     │
     ▼
 1. quantize (SQ 1-bit) and build the HNSW graph            ── same as stock, built as a single layer
     │
     ▼
 2. compute the order from the graph's neighbour lists      ── greedy page-growing (§6)
     │      output: ordToPhysicalOrdMap[ordinal] = position on disk
     │              up to 32 hub nodes (highest degree)
     ▼
 3. write the files
     ├── .faiss   HNSW graph, unchanged, still uses the original ordinals
     ├── .veqlo   quantized records, written in the new order
     ├── .vemlo   the ordinal → position map, the hubs, quantization metadata
     └── .vec     full-precision vectors, unchanged, original order (used for rescore)
```

Step by step:

1. **Quantize and build the graph.** Exactly what the stock SQ format does. The graph is built as a single layer;
   that is also how the baseline is built in our tests (§10.7).
2. **Compute the order.** We read the graph's layer-0 neighbour lists in place (no copy) and grow pages one at a
   time. Each page starts from the highest-degree node not yet placed (a hub). We then keep adding whichever
   unplaced node has the most neighbours already on that page, until the page is full (32 KB worth of records).
   Pages are packed back to back with no padding. The result is a permutation: for every vector, the position its
   record will occupy on disk. As a by-product we keep the top 32 hubs as search entry points.
3. **Write.** The quantized records go to `.veqlo` in the new order. The record format is byte-for-byte the same as
   the stock `.veq`; only the order differs. The permutation and the hubs go to a small sidecar, `.vemlo`. The graph
   file is not rewritten: it keeps using the original ordinals, so nothing in the graph code changes. The
   full-precision `.vec` stays in its original order.

Cost: one extra pass over the graph per segment. Indexing time is unchanged within noise (§10.8), and the index is
the same size apart from the small `.vemlo` file (4 bytes per vector for the map, plus the hubs).

### 5.3 Search flow

```
 query
   │
   ▼
 1. pick the entry point: the hub closest to the query (of up to 32)
   │
   ▼
 2. walk the graph (.faiss) as usual, in original ordinals
   │     for each node the search expands:
   │       a. take its neighbour list
   │       b. translate each neighbour: position = ordToPhysicalOrdMap[ordinal]   (array lookup)
   │       c. prefetch the 32 KB blocks that hold those positions, one call per block
   │       d. read each record from .veqlo at position × record size, and score it
   ▼
 3. top-k results, still in original ordinals → doc ids
   │
   ▼
 4. optional rescore: read the top results' full-precision vectors from .vec (original order)
```

Step by step:

1. **Entry point.** Instead of the graph's default entry point, the search starts from whichever stored hub is
   closest to the query. Hubs are where pages were seeded, so the first reads land in a dense, well-packed region.
2. **Graph walk.** The HNSW search itself is unchanged. It works in original ordinals, the same ones stored in the
   graph. The only difference is how a vector is read: the reader looks up the vector's position in the
   ordinal→position map (a flat int array, so one array access) and reads the record at that position in `.veqlo`.
   Before scoring a batch of neighbours, it asks the storage layer to prefetch the blocks they live in, merging
   neighbours that fall in the same 32 KB block into one call. Because neighbours were packed together at indexing
   time, a batch usually needs fewer distinct blocks, and blocks loaded for one step are often reused by the next.
3. **Results.** Hits come back in original ordinals and map to doc ids the usual way, so recall and results are
   unchanged (§10.3).
4. **Rescore.** If rescoring is on, the full-precision vectors are read from `.vec`, which is not reordered. Those
   reads cost the same on both layouts (§8.4).

Search-time memory: the reader keeps the ordinal→position map in memory, 4 bytes per vector per segment.

### 5.4 What changes, what doesn't

| | Stock SQ format | Reordered format |
|---|---|---|
| Graph (`.faiss`) | original ordinals | same file, same ordinals |
| Quantized records | `.veq`, insertion order | `.veqlo`, graph-neighbour order |
| Extra files | — | `.vemlo` (ordinal→position map, hubs) |
| Full-precision vectors (`.vec`) | original order | unchanged |
| Search entry point | graph default | closest stored hub |
| Reading a vector | position = ordinal | position = map[ordinal] |
| Prefetch | groups records within 32 KB of the first one | groups records by absolute 32 KB block |
| Results / recall | — | identical |

Routing: the reordered format is selected per field (in the upstream codec, the
`KNN1040`/`Faiss1040SQHNSWReordered*` locality format; gated by an index/field setting such as
`index.knn.advanced.reordering_enabled`). Fields that don't opt in use the stock SQ format unchanged.

---

## 6. The Ordering Algorithm — Dense Greedy Page-Growing

Implemented natively in `buildOrderingOfVectorsUsingIndexStructure`
(`jni/src/org_opensearch_knn_jni_FaissService.cpp`), driven from
`MemOptimizedScalarQuantizedIndexBuildStrategy`.

### 6.1 Intuition
Grow "pages" one at a time. Seed a page from a graph hub, then repeatedly pull in the unplaced node
with the **strongest affinity** to what's already on the page (affinity = number of edges to
already-placed page members). Fill the page to `pageCapacity` nodes, then start the next page.
Because we always add the most-connected candidate, a page ends up holding a tight local
neighborhood of the graph — exactly the nodes a query traversing that region will read together.

### 6.2 Mechanics
- **`pageCapacity`** = `32768 / recordSize` (records that fit in a 32 KB page; ~292 @768-dim,
  ~227 @1024-dim).
- **Hub seeding:** pages are seeded from highest-degree nodes first (best-connected = best entry
  points; also the nodes searches enter through).
- **Affinity bucket queue:** an O(1) best-candidate structure. `affinity[v]` = # edges from `v` to
  placed page members; placing a node increments the affinity of its unplaced neighbors and moves
  them up a bucket. `popBest()` returns the highest-affinity candidate.
- **Dense packing (no padding):** ordinals are assigned densely `0..N-1`. When a page "closes
  early" (its 1-hop candidates are exhausted before it is full), the next page's nodes are placed
  immediately after — no gap. Physical 32 KB page boundaries therefore fall every `pageCapacity`
  ordinals and do **not** align with cluster boundaries; a physical page can hold the tail of one
  cluster and the head of the next. (This is deliberate — strict-page padding was evaluated and
  rejected; see `Rejected_Ideas.md`.)
- **Coverage / termination:** after seeding from all hubs, any still-unplaced node seeds a fresh
  page, guaranteeing every node is placed exactly once (`nextOrd == N` invariant).

```
for hub in hubs_by_degree:            # seed from best-connected nodes
    if not placed[hub]: grow_page(hub)
for v in 0..N-1:                       # cover the remainder
    if not placed[v]: grow_page(v)

grow_page(seed):
    place(seed)                        # assigns next dense ordinal; bumps neighbors' affinity
    while page.size < pageCapacity:
        best = popBest()               # highest-affinity unplaced candidate
        if best < 0: break             # 1-hop frontier exhausted -> close early (dense: no pad)
        place(best)
```

### 6.3 Output
A dense permutation `ordToPhysicalOrdMap[logicalOrd] = physicalOrd` and a hub list, persisted in
`.vemlo`. The quantized records are emitted to `.veqlo` in physical order.

---

## 7. On-Disk Format

- **`.veqlo`** — the quantized SQ records in **physical (reordered) order**. Byte-identical record
  layout to the stock `.veq`, only the order differs.
- **`.vemlo`** — sidecar metadata: the `ordToPhysicalOrdMap` permutation, the hub set, centroid /
  correction metadata.
- **`.vec` / full-precision** — written **once, in original order**, shared as both the rescore
  source and (in the single-pass design, §11) the quantization source. Not reordered.

Writer path: `Faiss1040SQHNSWReorderedWriter` → `LocalityOrderedQuantizedVectorsWriter`. Reader:
`Faiss1040SQHNSWReorderedReader` → `LocalityOrderedQuantizedVectorsReader`, which registers the
permutation for search-time translation.

*(AOSS V2 backport uses the analogous `Faiss1030SQHNSWReordered*` names on the older codec base;
the design is identical.)*

---

## 8. Search Path

### 8.1 Hub-seeded entry
Search enters the graph from a hub (persisted in `.vemlo`) rather than an arbitrary node, so the
first reads land in a dense, well-connected region.

### 8.2 Ordinal translation
The graph and candidate lists are in logical ordinals; the store is in physical ordinals. The
reader translates logical→physical via `ordToPhysicalOrdMap` on each read. A dense flat-array
lookup (not a hash map) keeps this O(1) and cache-friendly.

### 8.3 Prefetch
Before scoring a batch of candidates, the reader issues `madvise(WILLNEED)` prefetches for the
records it is about to read. Two strategies:
- **Grouped** (`PrefetchHelper.prefetch`, baseline layout): sort candidate offsets and coalesce
  those within a 32 KB window into one prefetch, anchored at the batch's first record.
- **Per-ordinal / block-aligned** (`prefetchBasedOnOrdinals`, reordered layout): coalesce
  candidates that fall in the same **absolute 32 KB block** (`offset / 32KB`, anchored at store
  start) into one prefetch. Because reordering co-locates a node's neighbors onto a few pages, a
  batch collapses to a handful of block-aligned prefetches that match exactly the pages the query
  will fault. This changes syscall count / kernel read-ahead behavior, not the set of pages read;
  it is a latency/CPU refinement, not a read-amplification lever.

### 8.4 Rescore interaction (important)
When full-precision rescore is enabled, each query also reads large `.vec` records (scattered,
layout-invariant). At high dimensionality (e.g. 1024-dim) these reads dominate bytes/pages/latency
per query and **mask** the graph-read win end-to-end. Isolating the graph path (rescore off, or
PageTouch's per-family accounting) reveals the benefit. This is a measurement caveat, not a codec
defect.

---

## 9. Design Decisions & Rejected Alternatives

All ordering alternatives below were evaluated in a dense-layout simulation
(`scripts/locality_simulation_dense.py`) that reproduces the production packing, measuring
distinct pages touched per query and the intra-page edge ratio.

_All rows: M=16, ef_construction=256, ef_search=256; intra-page @1M (@100K); pages/query reduction vs insertion @1M (@100K). Full data: `Simulation_Results_M16.md`._

| Alternative | Result | Verdict |
|---|---|---|
| **Dense greedy page-growing** (shipped) | intra-page 0.03% → 17.1% @1M (0.22% → 18.1% @100K); pages/query −47.6% @1M (−15.5% @100K) | **Chosen** (validated on AOSS) |
| BFS from entry point | intra-page → 0.9% @1M (1.4% @100K); pages/query −44.6% @1M (−7.2% @100K) | Rejected — ~16 pp behind greedy on intra-page |
| Reverse Cuthill–McKee (bandwidth min) | intra-page → 0.7% @1M (1.4% @100K); pages/query −40.8% @1M (−5.3% @100K) | Rejected — bandwidth ≠ page-packing |
| Gorder (windowed edge overlap, w=32) | intra-page → 12.6% @1M (12.9% @100K); **pages/query −57.5% @1M** (−17.5% @100K) | Not shipping (intra-page below greedy, ~6× slower build) — **but BEATS greedy on pages/query at M=16 (full 10k queries); A/B warranted, see §9.1** |
| Greedy + **`aff/deg` containment tiebreak** | intra-page 17.1% → 18.9% @1M (18.1% → 18.8% @100K); pages/query −55.0% @1M (−17.3% @100K) | **Adopt** — recommended refinement |
| **METIS min-cut partitioning** (ceiling) | intra-page → 22.7% @1M (21.4% @100K); pages/query n/m | Deferred — gap over greedy grows with scale (§9.1, MinCut doc) |

Other ideas we evaluated and dropped — degree as a selection weight, 2-hop refill, reordering the
full-precision `.vec`, strict-page padding, and graph remap in `.faiss` — are documented in
`Rejected_Ideas.md` (kept out of here so this table stays focused on the live alternatives). The
containment `aff/deg` tiebreak above is the *only* degree-based scoring that helps; using degree as a
positive selection weight hurts.

**Gorder, benchmarked properly (fast implementation).** Gorder (Wei et al. 2016; the method Coleman
et al. 2021 found best for CPU-cache locality) maximizes, over a sliding window of `w` consecutive
labels, the neighbor + shared-neighbor overlap.

*Intuition (contrast with greedy page-growing, §6.1).* Instead of filling discrete pages, Gorder
lays all nodes on a single 1-D line and slides a window over the last `w` placed nodes. For each
next slot it picks the unplaced node that best "fits" the window, where fit = (# direct edges to
window nodes, the *neighbor score*) + (# neighbors shared with window nodes, the *sibling score*).
So it grows a ribbon in which *any* run of `w` consecutive nodes is as internally-connected as
possible — like arranging people in a line so any `w` in a row either know each other or share many
mutual friends, making every `w`-consecutive load maximally useful. The sibling term matters because
two nodes sharing many neighbors get co-visited during search even without a direct edge. The key
difference from greedy: greedy fills a **fixed-capacity page then hard-resets** (its unit is an
*aligned* block), whereas Gorder **never resets** — its window *floats* continuously. That is exactly
why Gorder is ideal for CPU cache lines (any `w` consecutive addresses share a line) but loses for
disk pages, which are *fixed, aligned* 32 KB blocks a floating window straddles.

Our first Gorder was a pure-Python port that ran
~1 hr at 1M — too slow to trust as a comparison. We rebuilt it as a Numba-JIT core with an O(1)
bucket priority queue (`scripts/gorder_fast.py`): **1M now reorders in ~65 s** (100K in ~8 s), so
speed is no longer the objection. With a fair, tuned comparison it still trails greedy on the
disk-page objective:

- **Window sweep (100K, M=32 — mechanism):** intra-page edge ratio rises monotonically with `w` but
  plateaus below greedy — `w=1`→2.4%, `w=5`→7.2%, `w=16`→10.7%, `w=32`→12.6%, `w=64`→14.0%.
- **M=16 (production config), w=32:** intra-page Gorder 12.6% @1M vs **greedy+tiebreak 18.9%** (greedy
  keeps ~50% more edges on-page) — so **on intra-page, greedy clearly wins**, and Gorder builds ~6×
  slower (1M ~65 s vs ~11 s).

⚠ **But on the pages/query metric, the M=16 re-run flips this — confirmed on the FULL 10,000-query
test set at 1M:** Gorder **1113.3** vs greedy+tiebreak **1179.4** pages/query @1M (Gorder **~5.6%
fewer** pages, −57.5% vs −55.0% reduction; also lower at p50 1126 vs 1198 and p99 1775 vs 1838).
Tied at 100K (saturated — a query touches ~82% of all pages there, no headroom). This is **not
sampling noise**: the nq=500 signal (Gorder 1138.9 vs 1206.8) reproduces on all 10k queries.

**The lesson: intra-page edge ratio is the wrong proxy for read amplification.** It weighs *all*
graph edges equally, but pages/query depends only on the nodes a query *actually traverses*. Gorder's
sibling score co-locates *co-visited* nodes (shared neighbors), which aligns with real traversal
better than greedy's all-edge page-packing — so greedy wins the intra-page proxy (18.9% vs 12.6%) yet
*loses* the metric that maps to read amplification. Greedy still ships (validated on AOSS −36%
read-amp, single-pass, ~6× faster; Gorder's edge is sim-only and O(n·deg²)), but "greedy strictly
dominates Gorder" is false at M=16 on the production-relevant metric — a **cluster A/B of Gorder vs
greedy on actual read-amp (PageTouch) at ≥10M is warranted.** See `Simulation_Results_M16.md`.

### 9.1 Scoring refinements: degree, containment, and the min-cut ceiling

We investigated whether the greedy selection score (affinity = # edges to nodes already on the
current page) can be improved by also using **node degree**, and how much headroom exists at all.

_All §9/§9.1/§10.1 simulation numbers are **M=16, ef_construction=256, ef_search=256**_ (matching
AOSS/production; full data in `Simulation_Results_M16.md`). Intra-page edge ratio; higher = better.
The degree-as-bonus sweep below is from the earlier M=32 run — kept as a mechanism illustration
(the direction is identical at M=16).

**Degree as a bonus — hurts.** Preferring high-degree candidates (score = affinity + α·degree)
degrades locality monotonically: 17.7% (α=0) → 16.3% (α=0.05) → 10.1% (α=0.2) → 5.3% (α=1.0). A
high-degree node's edges mostly point *outside* any single page, so pulling it in early adds
cross-page edges and displaces a node more tightly bound to this page. Degree as a pure tiebreak is
also slightly negative (17.7% → 17.0%). (Note this is the *opposite* of RCM's rule, which orders by
*increasing* degree — and RCM is a bandwidth heuristic that doesn't help pages anyway.)

**Degree as containment — a small, free win.** The correct use of degree is as a *penalty on
leakage*, not a bonus: among candidates with equal affinity, prefer the one whose neighborhood is
most *contained* on the page — i.e. tiebreak by `affinity / degree` (fraction of edges internal, the
conductance/modularity flavor). This beats plain affinity and is coefficient-insensitive (it acts as
a tiebreak since affinity is integer-dominant):

| Rule | Intra-page (100K) | Intra-page (1M) |
|---|---|---|
| affinity only | 18.1% | 17.1% |
| **affinity, `aff/deg` tiebreak** | **18.8%** | **18.9%** |

Verified at both scales (+0.7 pp @100K, **+1.9 pp @1M**), same complexity, no tuning. It also
improves pages/query: 1M reduction vs insertion goes +47.6% → **+54.9%** (100K +15.5% → +17.8%).
Subtracting degree (`aff − β(deg−aff)`) instead *hurts* (drifts toward low-degree leaves).

**The min-cut ceiling.** "Maximize intra-page edges under ≤ pageCapacity per page" is exactly
balanced k-way **min-edge-cut partitioning** (k = N/pageCapacity). METIS (`pymetis`) gives the
near-optimal ceiling. Measured at both scales:

| Ordering | Intra-page (100K) | Intra-page (1M) |
|---|---|---|
| greedy (dense) | 18.1% | 17.1% |
| greedy + `aff/deg` tiebreak | 18.8% (~88% of ceiling) | 18.9% (~83% of ceiling) |
| METIS min-cut (idealized ceiling) | **21.4%** | **22.7%** |

METIS runtimes are small (~4 s @100K, ~63 s @1M; part sizes tracked the cap: 219–233 @100K,
283–300 @1M). Even the optimal partition leaves **~78–80% of edges crossing pages** — this corpus is
structurally *diffuse* (weak clustering; see the cluster-tendency report, Hopkins 0.69 / silhouette
~0.02), so there is a hard, low ceiling on extractable page-locality regardless of algorithm.

**The gap to the ceiling widens with scale.** At 100K the tiebreak is ~88% of the ceiling; at 1M the
ceiling rises to 22.7% while the tiebreak sits at 18.9% = only **~83%**, leaving **~3.8 pp of
headroom** that a min-cut partitioner captures. So the "greedy ≈ ceiling, nothing to chase"
conclusion holds loosely at 100K but **weakens at production scale**.

**Net:**
- **Adopt the `aff/deg` containment tiebreak** — free, verified at both scales (+0.9–1.5 pp), no
  tuning. This is the concrete recommendation.
- **Do not use degree-as-weight** — it degrades locality (above).
- **Reconsider a partitioning-based reorder (METIS/Louvain) at large segment sizes** rather than
  ruling it out: the headroom above greedy grows with scale (~3.8 pp at 1M and climbing). Caveats
  before committing: the ceiling numbers are *optimistic* (label-based; parts slightly exceed the
  page cap and spill, so a real page-aligned partition reorder lands below the ceiling), and it adds
  a C dependency + symmetrized-CSR build. Worth a page-aligned prototype + read-amp measurement at
  ≥10M before deciding.
- The ceiling is also dataset-specific — a more clustered corpus raises it (and the payoff of both
  the tiebreak and partitioning). Repro: degree/containment sweep + METIS ceiling via
  `locality_simulation_dense.greedy_page_growing_permutation` + `pymetis`.

**Why greedy wins on intra-page (today):** it optimizes directly for graph neighbors sharing a
physical page (intra-page edge ratio 18–19% vs ~1% for bandwidth/traversal methods, ~12.6% for a
tuned Gorder). It reaches ~88% of the min-cut ceiling at 100K (~83% at 1M), is cheap and single-pass,
its benefit **scales with segment size** (§10.1), and it is the layout validated end-to-end on AOSS
(−36% read-amp). ⚠ **Caveat surfaced by the M=16 re-run:** on the *pages/query* metric (which maps
more directly to read amplification) Gorder is competitive-to-better than greedy at M=16/ef=256,
**confirmed on the full 10,000-query set at 1M** (Gorder 1113.3 vs greedy+tiebreak 1179.4 pages/query,
−5.6%; also lower p50/p99; tied at 100K), despite Gorder's lower intra-page ratio — its global 1-D
contiguity lets the beam-search working set span fewer pages. This is a genuine open question the
earlier M=32/intra-page analysis missed; greedy ships (validated, single-pass, ~6× faster to build),
but a cluster A/B of Gorder vs greedy on pages/query / read-amp at ≥10M is warranted. See
`Simulation_Results_M16.md`.

---

## 10. Benchmark Evidence

### 10.1 Simulation (offline, page-touch model)
`scripts/locality_simulation_dense.py` on Cohere-style data, measuring distinct pages touched per
simulated query (baseline = insertion order, reordered = dense greedy + `aff/deg` tiebreak).
**M=16, ef_construction=256, ef_search=256** (matches AOSS; full data in `Simulation_Results_M16.md`):

| Segment size | Baseline pages/q | Reordered | Reduction | Intra-page edges (base→reord) |
|---|---|---|---|---|
| 100K @1024 | 440.8 | 362.2 | **+17.8%** | 0.22% → 18.8% |
| 1M @768 (shuffled) | 2673.1 | 1206.8 | **+54.9%** | 0.03% → 18.9% |

_This table is the shipped greedy + `aff/deg` tiebreak alone. The **greedy + sibling** refinement
(borrowing Gorder's co-visitation score, §9/§10.1.1) improves on it further — at 1M, full 10k queries,
it takes the 32 KB reduction to **+61.3%** (best of greedy/Gorder/METIS). See §10.1.1 for the full
four-way comparison at 32 KB and 8 KB._

**Key finding:** the benefit grows strongly with segment size — +17.8% at 100K vs **+54.9% at 1M**.
The 100K reduction looks modest because at ef_search=256 a 100K index is near-saturated: a query
visits ~4,400 nodes spread across ~82% of the 441 pages regardless of layout, so there is little
headroom. At 1M there are far more pages, so co-locating a query's working set pays off much more.
Benefit also depends on ingest order: pre-sorted data (near-neighbors already adjacent) has a smaller
head-room than realistic random arrival — but reordering converges to the same page floor either way.
(Earlier drafts quoted M=32/ef=64 numbers — e.g. "−50% @1M"; the table above is the corrected M=16
production config.)

Visualizations (`scripts/reordering_benefit_viz.py`) make the mechanism visible: per-query
page-touch grids (scattered → clustered), the distinct-pages CDF (whole distribution shifts left),
and the edge-gap distribution (reordered spikes at gap≈0; ~18.6% of edges within one page vs 0.5%
baseline at 100K).

![HNSW reordering benefit — 1M shuffled dataset (baseline vs reordered): page-touch grids,
distinct-pages CDF, edge-gap distribution, and visit-step trace. Avg pages/query 1745 → 872
(−50%).](images/reorder_benefit_1m_shuffled.png)

*Figure: 1M-shuffled benefit visualization (`reorder_benefit_1m.png`). TODO: add the PNG at
`design-docs/hnsw-locality-based-design/images/reorder_benefit_1m_shuffled.png`.*

**Derived read amplification (100K × 1024-dim, page-touch model).** From the same simulation
(recordSize = ⌈1024/8⌉ + 16 = 144 B; ~1,900 distinct vectors visited/query), read amplification =
(distinctPages × pageSize) / (distinctVectors × recordSize). Because the denominator is
layout-invariant (same query visits the same vectors), the read-amp reduction equals the
pages-touched reduction:

| Page size | Avg pages/query (base → reord) | Read amp (base → reord) | Reduction |
|---|---|---|---|
| 32 KB | 432 → 293 | ~52× → ~35× | **−32%** |
| 8 KB | 1198 → 697 | ~36× → ~21× | **−42%** |

Smaller pages give **lower absolute amplification** (less wasted per fault) and a **larger relative
reduction** (reordering's co-location matters more when a page holds fewer records) — the same
32 KB-vs-8 KB pattern the AOSS PageTouch logs show at 10M/113M (§10.4). These absolute magnitudes
are page-touch-model *derived* (the reduction % is exact; the ~52×/35× depends on the vectors-visited
estimate); 100K is a small-segment regime (baseline touches ~98% of all pages), so treat as
illustrative, not production-representative.

#### 10.1.1 Four-way comparison at 1M — Greedy, Greedy+sibling, Gorder, METIS (full 10,000 queries)

`scripts/locality_compare.py` (one cached index build, all methods), on `documents-1m.hdf5`
(1M × 768-dim, shuffled ingest), **M=16, efC=256, efS=256, all 10,000 test queries**
(avg 5,070 nodes visited/query, recall@100=0.9116, layout-invariant). pages-touched-per-query
(read-amp = pages·pageSize / (visited·recordSize), recordSize=112 B); `reorder-s` = wall-clock to
compute the permutation. **Greedy+sibling** = greedy's `aff/deg` tiebreak plus a borrowed Gorder
sibling (shared-neighbor / co-visitation) term with weight b=0.5, Numba-JIT'd (`greedy_sibling_perm`).

**32 KB pages (page_capacity=292):**

| Method | intra-page | ptpq mean | p50 | p90 | p99 | p100 | reduction | read-amp | reorder-s |
|---|---|---|---|---|---|---|---|---|---|
| Insertion (baseline) | 0.04% | 2586.9 | 2625 | 2823 | 2911 | 3006 | — | 151.7× | 0.0 |
| Gorder (w=32) | 12.56% | 1129.1 | 1147 | 1526 | 1788 | 2171 | +56.4% | 64.5× | 29.7 |
| Greedy + aff/deg tiebreak (shipped) | **18.86%** | 1192.3 | 1211 | 1590 | 1843 | 2160 | +53.9% | 68.1× | **11.7** |
| METIS (min-cut, packed) | 15.84% | 1013.4 | 1018 | 1413 | 1699 | 2027 | +60.8% | 57.7× | 62.1 |
| **Greedy + sibling(b=0.5) + tiebreak** | 17.29% | **1001.9** | **1009** | **1364** | **1616** | **1958** | **+61.3%** | **57.2×** | 14.6 |

METIS label-based intra-page ceiling: 22.59% (k=3425 parts, sizes 282–300).

**8 KB pages (page_capacity=73):**

| Method | intra-page | ptpq mean | p50 | p90 | p99 | p100 | reduction | read-amp | reorder-s |
|---|---|---|---|---|---|---|---|---|---|
| Insertion (baseline) | 0.01% | 4198.7 | 4261 | 4903 | 5252 | 5617 | — | 60.9× | 0.0 |
| Gorder (w=32) | 9.48% | 1846.3 | 1871 | 2535 | 3016 | 3678 | +56.0% | 26.2× | 31.0 |
| Greedy + aff/deg tiebreak (shipped) | **13.70%** | 2188.4 | 2229 | 2967 | 3500 | 3948 | +47.9% | 31.1× | **10.6** |
| METIS (min-cut, packed) | 12.15% | 1813.4 | **1826** | 2539 | 3091 | 3659 | +56.8% | 25.7× | 96.6 |
| **Greedy + sibling(b=0.5) + tiebreak** | 13.11% | **1827.1** | 1853 | **2492** | **2970** | **3537** | **+56.5%** | **26.0×** | 13.3 |

METIS label-based intra-page ceiling: 17.58% (k=13699 parts, sizes 69–75).

**What this shows (all on the production-relevant pages/query metric):**

1. **The borrowed sibling term makes greedy the best realizable layout.** Adding Gorder's
   shared-neighbor (co-visitation) score to greedy's ranking (b=0.5) takes greedy from the **worst** of
   the reorderings to the **best**: **1001.9 pages/q @32 KB (+61.3%)**, beating METIS (1013.4), Gorder
   (1129.1) and plain greedy (1192.3), and lowest across p50/p90/p99/p100. At 8 KB it ties the leaders
   (1827.1, +56.5%, essentially level with METIS 1813.4 / Gorder 1846.3) and still far ahead of plain
   greedy (2188.4). This directly validates §9's mechanism: pages/query is a co-visitation metric, and
   importing the co-visitation signal is what closes the gap.
2. **…and it stays cheap.** Numba-JIT'd, the sibling variant costs **13–15 s** — ~1.2–1.3× plain greedy
   and **2–7× faster than Gorder/METIS** — so it captures the partition-quality win without their build
   cost or the external `pymetis`/symmetrized-graph dependency. (Pure-Python was ~66 s @100K / minutes
   @1M; JIT is ~44× faster.)
3. **Intra-page edge ratio again disagrees with pages/query.** Plain greedy has the **highest**
   intra-page ratio (18.86% / 13.70%) yet the **worst** pages/query of the reorderings; greedy+sibling
   has *lower* intra-page (17.29% / 13.11%) but the *best* pages/query — reconfirming intra-page is the
   wrong proxy (§9).
4. **Ordering by pages/query @32 KB:** greedy+sibling (1001.9) < METIS (1013.4) < Gorder (1129.1) <
   greedy (1192.3) < insertion (2586.9). Greedy still ships today (AOSS-validated, single-pass); the
   sibling term is a **drop-in refinement to the existing C++ greedy** (add a 2-hop shared-neighbor
   bump to the affinity score) and is now the leading candidate for a **cluster A/B at ≥10M** (§9.1) —
   ahead of Gorder and METIS on both quality and cost.

### 10.2 Cluster benchmark setup (AOSS V2, gamma)
Reordered codec (candidate) vs stock SQ (baseline), field-name-gated so the two collections differ
only by codec. Cohere embeddings, SQ 1-bit **32× compression**, `ef_search=256`, `ef_construction=256`,
`innerproduct`. Scales: **1M and 10M @768-dim, 113M @1024-dim.** Codec use was confirmed from
indexing-worker logs (candidate writes `.veqlo`/`.vemlo` + `LocalityOrdering`; baseline writes
stock `.veq`/`.vec`). Full run metadata (accounts, collections, instances) is retained in the
internal benchmark records.

> **Note on earlier runs.** The 1M/10M/113M results in §10.4–§10.6 were collected before a deployment-tooling
> issue was found: the candidate fleet was running an older build with vector prefetch behaving differently from
> the baseline. Their direction agrees with the clean results below, but their magnitudes should be read as
> indicative. §10.2.1 is the reference result.

### 10.2.1 Cluster results

This section answers four questions about the reordered layout. Most runs use one large index (100M vectors);
the single-query check under question 2 also covers 1M, 10M and 113M:

1. Does it change recall?
2. How much less data does a query need to read?
3. Does anything change once everything is cached?
4. What happens when the cache is too small and pages keep getting evicted?

#### What we compared

We loaded the same dataset twice: 102.7M Cohere embeddings, 1024 dimensions, SQ 1-bit (32× compression), HNSW with
M=16 and ef_construction = ef_search = 256. One copy uses the stock on-disk layout (the **baseline**); the other
uses the reordered layout (the **candidate**). Everything else is the same: both run the same software build with
vector prefetch on, each on its own search fleet of the same size, and we send both the same queries at the same
time. Queries ask for k=100 with rescoring turned off, so what we measure is the graph search itself.

For each run we recorded:

- **Latency and throughput**, measured by the client.
- **Recall**, against exact nearest neighbours computed by brute force.
- **Pages touched per query**: the number of distinct 32 KB pages holding the vectors a query actually scores. This
  comes from instrumentation in the search path and doesn't depend on what is cached.
- **Bytes read from storage and cache evictions**, from the search nodes' own counters.

One caveat applies to every result below: the two indexes were built separately, so their HNSW graphs and segment
structure are not identical. As a result the candidate scores about 6% *more* vectors per query than the baseline.
That works against the candidate, not for it. (The simulations in §10.1 compare the same graph laid out two ways.)

#### 1. Recall does not change

We ran all 10,000 test queries against both copies.

| | Baseline | Candidate |
|---|---|---|
| recall@1 | 0.9771 | 0.9781 |
| recall@10 | 0.7367 | 0.7380 |
| recall@100 | 0.6644 | 0.6647 |

This is expected: reordering only changes where vectors sit on disk, not which vectors the search finds. (Recall@100
is low here because rescoring is off. An earlier run with rescoring on gave 0.9449 vs 0.9458.)

#### 2. A typical query touches 37% fewer pages

We ran all 10,000 queries once and counted, for each query, how many distinct 32 KB pages it touched on each
copy. Then we compared the two copies query by query.

| Pages touched per query | Baseline | Candidate | Change |
|---|---|---|---|
| Median | 164,382 | 101,976 | −38% |
| Mean | 161,200 | 101,794 | −37% |
| 90th percentile | 185,674 | 131,079 | −29% |

Looking at each query on its own, the median saving is 37%; one in ten queries saves less than 27%, and one in ten
saves more than 48%. **Every one of the 10,000 queries touched fewer pages on the candidate.** Read amplification
(bytes paged in per byte actually used) drops from 141× to 82×.

We repeated this run on a fresh set of worker hosts and got the same numbers to within a few pages per query
(median saving 37.2% both times). The page counts depend only on the index and the queries, not on the hardware or
the cache, which is why they reproduce so exactly.

The saving depends on segment size. Large segments are where most of the reads happen, and that is where
reordering helps most:

| Segment size (vectors) | Pages per segment search: baseline → candidate | Change |
|---|---|---|
| under 10K | 10.8 → 12.1 | about one page more |
| 10K – 100K | 143 → 123 | −14% |
| 100K – 500K | 578 → 364 | −37% |
| 500K and above | 1,114–1,488 → 652–872 | −41% |

Very small segments only span a few dozen pages in total, so there is nothing to save.

To check that fewer pages really means fewer bytes read from storage, we also ran single queries on a cold cache,
at four index sizes (see "One query on a cold cache" below).

**What the search nodes' own counters show for the same run.** The node counters measure what was actually read
from storage and how the cache behaved. They were identical across the two runs, down to the byte:

| Node counters, one pass of 10,000 queries | Baseline | Candidate | Change |
|---|---|---|---|
| Bytes read from storage | 61.75 GB | 61.46 GB | −0.5% |
| Storage reads | 1.88M | 1.87M | −0.4% |
| Cache evictions | 0 | 0 | — |
| Cache hit rate | 22.3% | 24.1% | +1.8 points |
| Cache hits | 0.90–0.92M | 1.02M | +11% to +13% |
| Prefetched blocks that were already in the cache | 70–76K | 198–212K | 2.6× to 3.0× as many |
| Blocks loaded by prefetch | 745–760K | 726–739K | −3% |

So the total read from storage is the same on both layouts, while the per-query page count drops 37%. That isn't a
contradiction. The cache was large enough that nothing was evicted, so each block was read from storage once and
then served from memory. Over 10,000 different queries, almost every block of the index gets used by some query,
on both layouts. The total therefore measures the size of the index, not the work per query.

The per-query saving still shows up in the counters, as more reuse: the candidate's cache hits go up 11–13%, and
its prefetches find the block already in memory about three times as often. Neighbouring vectors now share
blocks, so a block loaded for one step of the search is often what the next step needs too.

The saving reduces total storage reads only when blocks have to be read more than once: after a cold start (the
single cold queries below read 14–22% fewer bytes), or when the cache is too small and blocks keep being evicted
(question 4).

#### One query on a cold cache, at four index sizes

The 10,000-query run counts pages, which is a model of what a query reads. To see what is actually read from storage,
we took a single query and ran it on a cold cache, at four index sizes. For each index we reset both fleets so they
start with an empty cache (32 search OCU, two workers each), then sent the same query 10 times in a row to both
copies at the same time. Same software build on both sides, prefetch on, rescoring off.

Page counts are identical on every repetition, so they come from one run. Bytes read from storage and the cache
figures cover all 10 repetitions together; only the first repetition starts from an empty cache.

**What the query reads** (baseline → candidate):

| Index | Segments searched | Vectors scored | 32 KB pages touched | 8 KB pages touched | Bytes read from storage |
|---|---|---|---|---|---|
| 1M × 768 | 31 / 32 | 38,855 → 39,578 (+2%) | 3,398 → 3,253 (**−4%**) | 10,685 → 9,617 (−10%) | 332 MB → 326 MB (**−2%**) |
| 10M × 768 | 94 / 95 | 104,485 → 103,641 (−1%) | 29,840 → 22,215 (**−26%**) | 61,086 → 41,983 (−31%) | 2.61 GB → 2.13 GB (**−18%**) |
| 100M × 1024 | 329 / 331 | 281,639 → 295,153 (+5%) | 164,541 → 110,185 (**−33%**) | 214,635 → 157,363 (−27%) | 16.20 GB → 12.63 GB (**−22%**) |
| 113M × 1024 | 258 / 253 | 246,578 → 247,675 (0%) | 147,792 → 113,719 (**−23%**) | 183,505 → 153,801 (−16%) | 16.48 GB → 14.10 GB (**−14%**) |

**Cache and latency** (baseline → candidate):

| Index | Working set in cache (peak) | Blocks loaded by prefetch | First query (empty cache) | Later queries, median |
|---|---|---|---|---|
| 1M × 768 | 130 → 140 MB (+8%) | 3.9K → 5.2K (+33%) | 5,426 → 5,259 ms (−3%) | 46 → 54 ms (+16%) |
| 10M × 768 | 1,060 → 814 MB (**−23%**) | 46.7K → 37.9K (−19%) | 7,958 → 6,922 ms (**−13%**) | 57 → 61 ms (+8%) |
| 100M × 1024 | 5,880 → 4,340 MB (**−26%**) | 278K → 190K (−32%) | 20,668 → 15,934 ms (**−23%**) | 69 → 43 ms (−38%) |
| 113M × 1024 | 6,110 → 5,060 MB (**−17%**) | 269K → 203K (−25%) | 23,755 → 23,806 ms (0%) | 71 → 57 ms (−20%) |

How to read these:

- **The saving grows with index size.** At 1M the index is about 31 small segments, mostly under 100K vectors, and
  there is little to co-locate: pages drop 4% and bytes 2%. From 10M up, a single query reads 14–22% fewer bytes from
  storage and keeps a 17–26% smaller working set in cache. This matches the segment-size breakdown above.
- **Both copies do about the same amount of graph work.** Vectors scored differ by 0–5%, so the saving comes from
  where the vectors sit, not from scoring fewer of them.
- **Bytes from storage fall less than pages.** At 100M, pages drop 33% and bytes 22%. Storage reads also include data
  the layout doesn't touch, such as the graph file and other segment files, so part of the total can't shrink.
- **The 1M result is reproducible.** We ran 1M twice with a fresh reset each time. Page counts and storage bytes were
  identical between the runs, down to the byte.
- **Cold latency improves at 10M and 100M, and is flat at 113M.** The first query is 13% faster at 10M and 23% faster
  at 100M. At 113M it reads 14% less but takes the same time.
- **Don't read much into the "later queries" column.** It is based on 4–7 repetitions of one query. The numbers vary
  with which shard copy a request lands on (each shard has two copies, and a query is slow until every copy it hits
  is warm) and with the host. Across runs we've seen this kind of variance be as large as the differences in the
  table.

Earlier single-query runs at 100M agree: −20% bytes from storage with prefetch off, −24% with prefetch on. The query
in those runs is one of the less favourable ones for this layout (28% fewer pages, against a median of 37% across
10,000 queries), so a typical query should save more than these single-query numbers suggest.

#### 3. With everything cached, there is no slowdown

We warmed both fleets, then looped over all 10,000 queries for 20 minutes. During the measured window neither fleet
read anything from storage and nothing was evicted.

| Steady state, all data in memory | Baseline | Candidate | Change |
|---|---|---|---|
| p50 latency | 49.0 ms | 40.2 ms | −18% |
| p90 latency | 56.9 ms | 47.5 ms | −17% |
| p99 latency | 67.6 ms | 57.5 ms | −15% |
| Queries per second | 20.1 | 24.4 | +22% |

This looked like a speed-up, and a single-pass run on the same hosts showed a similar gap (p50 −17%). But when we
repeated the single-pass run on a fresh set of worker hosts, the gap almost disappeared:

| Single pass, warm part of the run (queries 2,000–9,999), p50 | Baseline | Candidate | Change |
|---|---|---|---|
| First run | 33.2 ms | 27.6 ms | −17% |
| Repeat on different hosts | 36.4 ms | 35.2 ms | −3% |

Between the two runs the candidate's p50 moved by 28% and the baseline's by 10%, just from landing on different
hosts. Host-to-host variance is as large as the difference we were seeing, so we don't count the in-memory speed-up
as a result. The claim we make is the safe one: **when everything is cached, the reordered layout is not slower.**
This makes sense. With nothing being read from storage, both layouts score the same kind of vectors with the same
code, and the layout only changes which memory pages they live on.

#### 4. When the cache is too small, the candidate completes 50% more queries

We halved search capacity, so the working set for 10,000 queries (about 24–26 GB) no longer fits in the cache. Then
we looped over all 10,000 queries for 30 minutes. Both fleets evicted millions of pages and both were overloaded:
some requests were rejected even after three retries.

| Heavy eviction, 30 minutes | Baseline | Candidate | Change |
|---|---|---|---|
| Successful queries | 1,780 | 2,674 | +50% |
| Requests rejected | 28.5% | 17.3% | 11 points fewer |
| p50 latency | 567 ms | 496 ms | −13% |
| p90 latency | 1,356 ms | 1,022 ms | −25% |
| p99 latency | 3,959 ms | 2,480 ms | −37% |
| Bytes read from storage per successful query | 331 MB | 257 MB | −22% |
| Evictions per successful query | 1,460 | 1,030 | −29% |

Here every page a query touches may already have been evicted and has to be read again. Both fleets were limited by
the same storage throughput. Because each candidate query needs fewer pages, the candidate completed more queries
with that throughput, and its slowest queries got faster. An earlier run at the same capacity with prefetch off
showed the same pattern (31–43% more successful queries, p99 −38% to −39%).

Because both clusters were overloaded, treat these as stress-test numbers rather than a production load profile.

#### Summary

| Situation | Effect of reordering |
|---|---|
| Recall | unchanged |
| Pages touched per query (10,000 queries) | −37% median; every query improved |
| One query on a cold cache | 14–22% fewer bytes read and 17–26% smaller working set from 10M up (2% at 1M); first query up to 23% faster |
| One pass of 10,000 queries, cache big enough | same total bytes read; 11–13% more cache hits; latency within host variance |
| Everything cached (steady state) | no slowdown (differences within host variance) |
| Cache too small, heavy eviction | +50% completed queries, p99 −37%, −22% bytes per query |

#### Not covered yet

- **Rescoring on.** All of the runs above had rescoring off. Production queries at 32× compression normally rescore
  the top results with the full-precision vectors, and those reads don't depend on the layout. We still need to
  measure how much of the latency gain remains with rescoring on.
- **Whether the layout is also faster in memory** (question 3). One run suggested it, a repeat on other hosts did
  not. Answering it needs several repeats on the same hosts.
- **An eviction test on a cluster that isn't overloaded** (question 4 was measured under overload).

### 10.3 Correctness — recall identical at every scale

Recall is **identical between baseline (stock) and candidate (reordered) at every scale**, by
design: reordering changes only the physical on-disk byte layout, not the graph topology or which
nodes the search visits, so the returned results are the same.

| Dataset | k | Recall@100 (baseline / candidate) | Recall@1 | Source / method |
|---|---|---|---|---|
| 1M @768 | 100 | 0.92 / 0.93 | 0.99 / 0.99 | OSB `no-train-test-search-only-aoss-v2`, full test procedure |
| 10M @768 | 100 | 0.94 / 0.94 | 0.98 / 0.98 | OSB full 10k distinct-query run |
| 113M @1024 | 100 | 0.95 / 0.97 | — | OSB 1,677-query run; iso-recall in every cell, 0% error |
| 100M @1024 (wikipedia-cohere-v3) | 100 | 0.944934 / 0.945793 | 0.9825 / 0.9819 | Full 10k-query run vs exact brute-force GT (`faiss.IndexFlatIP`); parity |

Notes:
- The 113M **"0.95 / 0.97" is the cold (`prod-queries`) vs warm (`prod-queries-cached`)
  test-procedure cells reporting slightly different values — NOT a candidate-vs-baseline gain.**
  Within each cell, baseline and candidate recall were identical.
- The **100M** run generated its own held-out query set (10,000 queries) with exact ground truth
  (k=100, cosine, `faiss.IndexFlatIP`) since OSB cannot drive the parquet corpus directly — recall is
  at parity (0.9449 / 0.9458), confirming the layout-only invariant at 100M scale too.
- Config for all: FAISS SQ 1-bit, 32× compression, HNSW, innerproduct, ef_search=256,
  ef_construction=256; recall reported by OSB against the dataset's ground-truth neighbors.

Layout-only change ⇒ correctness preserved; this is what lets the read-amplification / page-fault /
tail-latency wins come for free.

### 10.4 Read amplification (PageTouch model, graph 112/144-B family)

| Scale / index | Page | Baseline readAmp (p50) | Candidate | Δ |
|---|---|---|---|---|
| 10M natural | 32 KB | 44.5× | 30.7× | **−31%** |
| 10M natural | 8 KB | 33.6× | 21.2× | **−37%** |
| 10M shuffled | 8 KB | ~38× | 22.1× | **−42%** |
| 100M @1024 | 32 KB | 99.4× | 66.1× | **−34%** (avg −36.4%: 97.6×→62.1×; distinctPages −37.2%) |
| 113M | 32 KB | 90.4× | 65.0× | **−28%** |
| 113M | 8 KB | 31.8× | 25.3× | **−20%** |
| any | rescore (3072/4096 B) | ~10.7× (floor) / 7.78× @100M | ~10.7× / 7.78× | **flat** |

Reordering claws back graph-read amplification; rescore (full-precision) reads are flat —
the expected locality signature. Smaller pages amplify less and benefit relatively more.

### 10.5 Per-query memory (BufferPool, real node counters)

Controlled cold-cache, same query, per-query = burst-bucket ÷ N (`JunoSearch/Worker` NodeStats).

**10M @768 (`target_index_10m`, natural order):**

| Metric (per query) | Baseline | Candidate | Δ |
|---|---|---|---|
| Allocation.BytesDelta | 253.0 MB | 194.4 MB | **−23.2%** |
| Cache.Misses | 12,042 | 9,347 | −22.4% |
| Cache.Loads | 7,722 | 5,932 | −23.2% |
| WssRecentMB (max) | 989 MB | 779 MB | −21.2% |
| Prefetch.BlocksLoaded | 4,579 | 3,506 | −23.4% |
| Cache.HitRate | 28.84% | 29.75% | +0.9 pp |

The **10M shuffled** index (`target_index_10_shuffled`) is within noise of the above:
Allocation.BytesDelta 271.6 → 209.0 MB (−23.1%), misses −21.0%, loads −23.1%, WSS −25.1% — i.e.
the ~23% holds whether or not the data was pre-ordered (⇒ a codec-layout property, not data order).

**113M @1024 (single-query, rescore OFF; magnitude scales with cache warmth):**

| Metric (per query) | Baseline | Candidate | Δ |
|---|---|---|---|
| Allocation.BytesDelta — warmer regime (run 4) | 74.55 MB | 26.01 MB | **−65.1%** |
| Allocation.BytesDelta — reproducible cold fault (runs 5/6) | 605.9 MB | 476–492 MB | **−19 to −21%** |
| Cache.Misses (warmer) | 3,479 | 1,566 | −55.0% |
| Cache.Loads (warmer) | 2,277 | 953 | −58.1% |
| Prefetch.BlocksLoaded (warmer) | 1,318 | 387 | −70.7% |
| WssRecentMB max (warmer) | 4,479 MB | 2,763 MB | −38.3% |

The **reproducible cold-fault delta is ~−20% bytes** (runs 5/6 agree — baseline byte-identical
across them). The **−65%** is a warmer regime where graph-read locality dominates the differentiator
— real but state-dependent, not the reproducible baseline. Across all runs the **direction is
consistent on every counter** (bytes, misses, loads, prefetch, WSS); magnitude ranges from ~−20%
(fresh cold fault, both sides fault large working sets) up to ~−65% (warmer cache).

**100M @1024 (wikipedia-cohere-v3, rescore ON, per-min Maximum = peak per-worker load):**

| Metric (peak per worker) | Baseline | Candidate | Δ |
|---|---|---|---|
| Allocation.BytesDelta | 4.26 GB | 3.54 GB | **−17.0%** |
| Cache.Misses | 216,222 | 153,549 | −29.0% |
| Cache.Loads | 130,132 | 100,539 | −22.7% |
| WssRecentMB | 36,251 MB | 29,730 MB | −18.0% |
| Prefetch.BlocksLoaded | 103,166 | 82,410 | −20.1% |
| Cache.Evictions | 83,844 | 0 | **−100%** |

Even with rescore ON, candidate's **peak** per-worker read pressure is ~17–29% lower (evictions
83.8k → 0). The Sum-over-window aggregate looks flat (−0.5%) because layout-invariant rescore bytes
(4096 B) dominate the total at 1024-dim — the peak view and PageTouch (§10.4) are where the graph-path
win shows.

### 10.6 Latency

| Run | Metric | Baseline | Candidate | Δ |
|---|---|---|---|---|
| 1M, 100×10 loop | p99 (server took) | 248 ms | 149 ms | **−40%** |
| 100M full 10k, rescore ON | E2E p90 | 82.6 ms | 65.6 ms | **−20.6%** |
| 100M full 10k, rescore ON | E2E p100 | 15159 ms | 3202 ms | **−78.9%** |
| 113M OSB, warm, iso-recall | p99 | 290.8 ms | 185.0 ms | **−36%** |
| 113M OSB, warm | throughput | 14.4 ops/s | 16.4 ops/s | **+14%** |
| 113M single-query, rescore OFF | warm p50 | ~100 ms | ~84 ms | **−19%** |
| 1M/10M @32-OCU, fully cached | p50/p90 | — | — | within noise |

Latency benefit is real at the **tail** and at **scale/cold**, and is **muted when the working set
is cache-resident** (small scale, high OCU) — consistent with the design: the win is fewer page
faults, which only cost latency when you actually fault.

### 10.7 Confound check — layout vs single-layer
The candidate codec also builds a single-layer graph, so the A/B changes two variables. From exact
per-query PageTouch totals: candidate does **same-or-more** search work (ordsScored +0.6% @10M,
+6.8% shuffled; graph-read batches +1–9%) yet reads **~32–33% fewer pages**. "Single-layer does
less work" predicts the opposite → the win is driven by **layout**, not by scanning fewer vectors.
(Not fully resolved: a single-layer-no-reorder control arm is still needed to exclude subtler
layer-count effects on *which* vectors are visited.)

### 10.8 Indexing overhead
Negligible. 1M: baseline 824 s vs candidate 816 s. 113M: baseline 57,063 s vs candidate 54,851 s
(candidate slightly faster). Store size identical (reordering is a permutation, not extra data).
≤1% throughput difference.

---

## 11. Open Questions & Caveats

- **Rescore masking (measurement):** at high dim with rescore on, layout-invariant full-precision
  reads dominate and hide the graph-read win end-to-end. Report graph-family metrics or rescore-off
  to see the benefit; state the end-to-end number with the rescore caveat.
- **Layer-count confound:** candidate bundles single-layer + reorder. §10.7 rules out "less work"
  as the mechanism but a clean single-layer-no-reorder arm would fully isolate layout.
- **Double-write at index time (write amplification):** today the quantized data is written twice
  transiently (`.veq` then `.veqlo`, `.veq` deleted). The **single-pass `.veqlo` write** design
  (write `.veqlo` directly from the native Faiss SQ storage address via `MemorySegment`, quantizing
  from `.vec`) removes the second write; documented in `SinglePass_Veqlo_From_Native_Storage.md`,
  not yet implemented.
- **Small segments gain little:** the benefit needs large segments; pair with force-merge guidance.
- **Prefetch strategy** (grouped vs block-aligned) affects latency/syscalls, not read-amp; A/B on
  latency to confirm the block-aligned variant is a net win for the reordered layout.

---

## 12. Future Work

1. **Single-pass `.veqlo` write** — eliminate the transient double-write (design doc exists).
2. **Generalize beyond SQ 1-bit** — the ordering is quantization-agnostic; wire other encoders.
3. **Tune `pageCapacity` / page size** — 8 KB shows larger relative amplification wins; expose as a
   setting if storage page size differs.
4. **Single-layer-no-reorder control** — close the confound (§10.7).
5. **Force-merge / segment-size guidance** — the benefit scales with segment size (§10.1).
6. **Containment tiebreak in the C++ ordering** — add the `aff/deg` tiebreak to the bucket-pop in
   `buildOrderingOfVectorsUsingIndexStructure` (§9.1): free +0.9–1.5 pp intra-page, one-line change.
7. **Min-cut partitioning reorder at large segments** — the greedy→ceiling gap grows with scale
   (~2.8 pp above the tiebreak at 1M, §9.1). Prototype a page-aligned METIS/Louvain reorder and
   measure read-amp at ≥10M before deciding; weigh against the added dependency and build cost.

---

## 13. Appendix

### 13.1 Metric definitions
- **readAmplification** = dataReadBytes / dataUsedBytes = (distinctPages × pageSize) / (distinctRecords × recordSize).
- **PageTouch** = per-read model mapping record offsets → pages at a chosen page size (per-family:
  graph 112/144 B vs rescore 3072/4096 B). Modeled, uncached.
- **BufferPool.*** = real node-level counters (`JunoSearch/Worker`): `Allocation.BytesDelta` (bytes
  S3→pool), `Cache.Misses`/`Loads`, `WssRecentMB` (resident working set), `Prefetch.BlocksLoaded`.
- PageTouch and BufferPool corroborate directionally; magnitudes differ (PageTouch = graph-only,
  modeled 32 KB; BufferPool = all reads, real block size + read-ahead, cold burst ÷ N).

### 13.2 Reproduction scripts (in `scripts/`)
- `locality_simulation_dense.py` — dense-layout page-touch simulation; compares Insertion / BFS /
  RCM / greedy / greedy-2hop.
- `gorder_fast.py` — Numba-JIT Gorder (O(1) bucket priority queue); timing + intra-page-edge
  comparison vs greedy (`--window` for the sweep).
- `reordering_benefit_viz.py` — page-touch grids, distinct-pages CDF, edge-gap distribution, visit
  trace.
- `recall_reordering.py` — cluster A/B: ingest reordered vs baseline index, recall@k, PageTouch
  read-amp summary; flags `--ef-search`, `--compression`, `--size`, `--only`, `--search-only`,
  `--shuffle-ingest`.
- `dataset_clustering_report.py` — cluster-tendency analysis of a dataset.
- `shuffle_dataset.py` / `rotate_dataset.py` — decorrelate ingest order / random-rotate (GT-preserving).

### 13.3 Related internal records
Full AOSS run metadata and raw metrics: `hnsw-locality-benchmark-results.md` (1M/10M),
`hnsw-locality-113m-benchmark-results.md`, `steady_state_113M_run.md`,
`recall_across_datasets.md` (recall summary across scales), `airbnb-100M-run-benchmarks.md`
(100M / 102.7M run: recall parity, latency tail, PageTouch −36%, BufferPool peak view).
