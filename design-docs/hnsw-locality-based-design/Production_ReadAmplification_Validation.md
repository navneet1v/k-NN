# Production Read-Amplification Validation: Locality Reordering (Faiss SQ 1-bit, on_disk 32x)

**Date:** 2026-09-01
**Branch:** `hnsw_locality`
**Scope:** First end-to-end validation on a live OpenSearch node (not the Python simulation) of the
graph-locality vector reordering, measured with **read amplification** — the bytes the device faults
in versus the bytes a query actually uses.

---

## Executive Summary

- **Greedy dense reordering works in production.** On realistically-ordered (random-arrival) data it
  cuts read amplification **~40% at a 32 KB page and ~46% at an 8 KB page**, with **no recall change**.
- **Page size is the larger single lever.** Going 32 KB → 8 KB cuts amplification ~2.3–2.5× on its own.
  It compounds with reordering: **baseline @32 KB (137×) → greedy @8 KB (32×) = 4.2× less I/O.**
- **Absolute amplification stays high (32×+)** at `ef_search=256` because ANN search sweeps broadly
  across the graph; no layout can collapse a broad visited set onto a few pages. Lower ef narrows this.
- **The `--shuffle-ingest` benchmark is the realistic case, not a hack.** `documents-1m.hdf5` is
  atypically pre-sorted by similarity; shuffle removes that artifact to model normal random doc arrival.

---

## What "read amplification" means

The store is a flat array of 112-byte quantized records. The OS/device faults in a whole page
(32 KB or 8 KB) for every page a query touches, but only the *visited* vectors' records on it are
useful:

```
dataReadBytes (X) = distinctPages * pageSize          # bytes moved by the device
dataUsedBytes (Y) = distinctVectors * recordSize      # bytes the query actually needed (each once)
readAmplification = X / Y                              # bytes moved per useful byte; 1.0x = perfect
```

Equivalently, `readAmp = (pageSize / recordSize) / vecsPerPage`. Floors (fully scattered, one useful
record per page): 32 KB → 32768/112 ≈ **293×**, 8 KB → 8192/112 ≈ **73×**. Reordering raises
`vecsPerPage` (packs visited vectors together), pushing amplification below the floor.

---

## Setup

- **Dataset:** `documents-1m.hdf5` (Cohere Wikipedia, 768-dim, inner product), 1,000,000 vectors.
- **Index:** Faiss SQ 1-bit, `on_disk`, `compression_level: 32x`; single **force-merged** segment
  (so per-query == per-segment over one 1M-node graph). Record = 96 B code + 16 B corrections = 112 B.
- **Ingest:** `--shuffle-ingest` — doc-id decorrelated from vector content (fixed seed), modeling
  realistic random arrival (see "Dataset is pre-sorted").
- **Query:** `ef_search=256`, `k=100`, `rescore=false`, 20 queries.
- **Ordering:** dense greedy (`buildOrderingOfVectorsUsingIndexStructure`).
- **Instrument:** `PageTouchTracker`, reporting read amplification at both 32 KB and 8 KB.

---

## Results

### Read amplification (both stores 112 B/record → clean, no normalization)

| readAmplification | 32 KB page | 8 KB page |
|---|---|---|
| **baseline** (reordering disabled) | 137.2× | 59.7× |
| **greedy** (reordering enabled)    | **82.5×** | **32.4×** |

### Supporting per-query metrics (avg)

| | pages/query | vecs/page | page utilization |
|---|---|---|---|
| baseline 32 KB | 2797 | 2.13 | 0.73% |
| greedy 32 KB   | 1666 | 3.55 | 1.21% |
| baseline 8 KB  | 4867 | 1.23 | 1.68% |
| greedy 8 KB    | 2616 | 2.26 | 3.09% |

- Data used ≈ 12.7 MB total / 20 queries ≈ 635 KB/query (~5,960 distinct vectors/query).
- **Recall@100 flat:** baseline 0.6365 vs reordered 0.6335 (−0.003, noise). Reordering changes layout
  only, not vector values, so recall is unaffected by design.

---

## Analysis

### Two independent, compounding levers
- **Reordering (greedy vs baseline):** −40% at 32 KB, −46% at 8 KB. `vecsPerPage` +67–84%.
- **Page size (32 KB → 8 KB):** ~2.3× lower (baseline), ~2.5× lower (greedy).
- **Combined:** 137.2× → 32.4× = **4.2× less I/O**.
- Page size is the bigger single lever (~2.4×) than reordering (~1.4×) at this ef; reordering helps
  slightly more at 8 KB because finer pages convert clustering into savings with less rounding waste.

### Why absolute amplification is high
At `ef_search=256` the visited set is ~5,900 vectors, almost all distinct (little neighbor-list reuse).
Perfect clustering would need only 5964/292 ≈ **20** pages at 32 KB, but best-first search is a broad
sweep across many clusters, not one tight neighborhood — so ~1,666 pages is the realistic floor. Only
visiting fewer nodes (lower ef) reduces this further; the simulation's "greedy advantage grows at lower
efSearch" implies the reordering win widens as ef drops.

### Sanity: the shuffled baseline is genuinely random
For 5,964 vectors scattered uniformly over 3,425 32 KB pages, expected distinct pages ≈ 2,822; baseline
measured **2,797**. The shuffled baseline is effectively random placement (as intended); greedy
concentrates it to 1,666.

---

## Dataset is pre-sorted — why `--shuffle-ingest` is the realistic model

`documents-1m.hdf5` is atypically pre-sorted by similarity, so its *natural* ingest order is a
near-best-case (already-reordered) baseline. Data-only proof (no reordering code involved):

- **True nearest-neighbor id gap** `|id − nn_id|`: median **13** (p25=2, p75=67).
- **81%** of vectors have their true NN within one 32 KB page of ids (≤292).
- Realistic random arrival would show ≈292/N = **0.3%** within a page, median gap ~N/2 → this dataset
  is pre-sorted ~270× more than random.

In real OpenSearch, post-merge vectors are laid out in doc-id (ingest) order, and for most workloads
arrival order is uncorrelated with embedding space → physically scrambled. `--shuffle-ingest`
reproduces that; it removes the dataset's unrealistic pre-sort rather than breaking a realistic
baseline. On pre-sorted ingest the baseline is already near-optimal and reordering adds little — but
greedy does not hurt (unlike BFS, which scrambles an already-good order).

---

## Implementation changes on this branch

- **Ordering: BFS → dense greedy.** `buildOrderingOfVectorsUsingIndexStructure` packs affine clusters
  back-to-back into a dense `0..N-1` permutation (no strict-page padding — strict wasted ~90% and is
  deferred). Build strategy calls it with `pageCapacity = 32768 / (quantizedVecBytes + 16)` = 292.
  Confirmed applied in production via the reader diagnostic: `displaced=100%`, and the 2nd cluster
  seed lands at physical slot 292 (= pageCapacity), the dense-greedy signature.
- **Baseline prefetch-stride bug fixed.** `PrefetchableVectorValuesHelper.doPrefetch` used
  `getVectorByteLength()` (96 B, code only) as the prefetch stride, but SQ records are 112 B. It now
  derives the stride from `slice.length()/size()` → 112. This fixed both a real baseline prefetch bug
  (misaligned `ord*96` ranges) and the page-touch metric (baseline previously under-counted at 96).
- **Instrumentation: `PageTouchTracker`** (`org.opensearch.knn.index.codec.scorer`). Reports read
  amplification at 32 KB and 8 KB per query. Enable at runtime, no restart:
  ```
  PUT _cluster/settings
  {"persistent":{"logger.org.opensearch.knn.index.codec.scorer.PageTouchTracker":"DEBUG"}}
  ```
  or via `scripts/recall_reordering.py --page-touch --log-file <node log>` (toggles the logger and
  prints a per-index read-amplification summary).

### How to reproduce
```bash
# (rebuild plugin + restart node to pick up the branch)
python scripts/recall_reordering.py --dataset scripts/documents-1m.hdf5 \
  --num-docs 1000000 --num-queries 20 --k 100 --ef-search 256 \
  --only both --search-only --shuffle-ingest --page-touch \
  --log-file build/testclusters/integTest-0/logs/integTest.log
```

---

## Open items / next steps

- **Clean BFS-vs-greedy A/B in production.** BFS was measured earlier at a different ef and before the
  prefetch fix, so a same-ef head-to-head is still outstanding. The sim (identical conditions) puts
  BFS ~41% vs dense greedy ~52–56% page reduction at 1M/ef=256; production greedy came in at ~40%
  page reduction — closer to the sim's BFS than its greedy. A build-time BFS/greedy toggle is needed
  for the clean A/B (deliberately deferred). If greedy does not clearly beat BFS in production, the
  likely cause is our C++ greedy using 1-hop affinity vs the sim's two-hop expansion.
- **Sweep ef down (100, 64).** Expect both amplifications to drop and greedy's relative advantage to
  widen — the regime where reordering matters most.
- **Hub-based entry points.** Hub metadata is written to `.vemlo` but not yet wired into search as a
  `KnnSearchStrategy.Seeded`; search currently uses the graph's single stored entry point.

---

# Realistic-Order Dataset: Benefit Without Shuffle (2026-09-01)

The 1M results above used `documents-1m.hdf5`, which is pathologically pre-sorted, so `--shuffle-ingest`
was required to model realistic arrival. A second dataset, `dataset.hdf5` (1024-dim, 100k train,
10k queries), has a **naturally random** order — so the reordering benefit shows with **no shuffle**,
directly answering the "manufactured-win" concern.

## Dataset order report

A reusable check, `scripts/dataset_order_report.py`, classifies a dataset's natural order (auto-derives
recordSize/pageCapacity from the dimension):

| dataset | true-NN within one 32KB page | GT id-span (median) | verdict |
|---|---|---|---|
| `documents-1m.hdf5` (768-d, 1M) | **82%** (median id-gap 12) | 963k / 1M | PRE-SORTED → needs `--shuffle-ingest` |
| `dataset.hdf5` (1024-d, 100k)   | **3.8%** (median id-gap 26,725) | 98k / 100k | REALISTIC → benchmark as-is |

```
python scripts/dataset_order_report.py --dataset scripts/<file>.hdf5
```
Verdict tiers on "fraction of vectors whose true NN is within one 32KB page": ≥25% PRE-SORTED,
≤6% REALISTIC, between = mildly pre-sorted.

## Read amplification on `dataset.hdf5` (100k, 1 segment, ef=100, k=100, NO shuffle)

recordSize = 1024/8 + 16 = **144 B** → 100k vectors = **441 pages** (32 KB) / **1,786 pages** (8 KB).

| readAmplification | 32 KB | 8 KB |
|---|---|---|
| **baseline** | 44.1× | 31.7× |
| **greedy**   | **36.0×** | **19.0×** |
| reduction | **18%** | **40%** |

vecs/page 5.16→6.33 (32 KB) and 1.79→3.00 (8 KB, +68%). Recall flat (0.7135 vs 0.7140).
**Combined greedy+8 KB (19.0×) vs baseline+32 KB (44.1×) = 2.3× less I/O.** No shuffle used — this is
the honest realistic case.

## Key insight: small-file saturation understates the 32 KB benefit

At 100k the 32 KB file is only 441 pages, and the query visits ~2,224 vectors:

- **32 KB baseline touches 431 of 441 pages (98%)** — its 44.1× is essentially the "read the whole
  file" cap (`441×32768 / (2224×144) ≈ 45×`). Almost no headroom → greedy can only shave **18%**.
- **8 KB de-saturates** (baseline 1,240 of 1,786 pages = 69%, not saturated) → the clustering has room
  to matter → **40%** reduction.

So at small scale the 32 KB number is capped by the file being tiny; the 8 KB number is the honest
signal. On a **1M+ realistic dataset** the 32 KB benefit would open up too (more pages → more headroom),
matching the ~40% seen at 1M on the shuffled `documents-1m`. Finer pages both lower absolute
amplification and expose more of the locality benefit — a real design lever.

**Bottom line:** on realistically-ordered data, greedy reordering reduces read amplification with no
benchmark trickery; the magnitude grows with (a) larger single segments and (b) finer page sizes.
