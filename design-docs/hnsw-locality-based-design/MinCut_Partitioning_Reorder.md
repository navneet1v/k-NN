# Min-Cut Partitioning as a Locality Reorderer — Results & Implementation Plan

**Status:** Parked / investigate-later. Companion to `HNSW_Locality_Reordering_Design.md` (§9.1).
**Question:** can a balanced min-edge-cut graph partition (METIS/Louvain) beat the shipped greedy
page-growing ordering for the 32 KB disk-page objective, and is it worth implementing in the native
(C++/JNI) build?

---

## 1. Why this is the right formulation

Our objective — "put as many graph-edge endpoints as possible on the same fixed-size page" — is
*exactly* **balanced k-way minimum-edge-cut partitioning**:

- k = ⌈N / pageCapacity⌉ parts, each part ≤ pageCapacity nodes (a page).
- minimize the edge cut = maximize intra-part (= intra-page) edges.

Greedy page-growing is a cheap one-pass *approximation* of this; a multilevel partitioner (METIS)
is the near-optimal solver, so its intra-page ratio is the practical **ceiling** for any ordering.

---

## 2. Results (measured)

Intra-page edge ratio = fraction of layer-0 edges whose endpoints land in the same page. Higher is
better. Cohere-style data, SQ 1-bit, HNSW M=32; page_capacity = 32768 / recordSize.

### 100K × 1024-dim (page_capacity 227)
| Ordering | Intra-page | Notes |
|---|---|---|
| greedy (dense, affinity) | 17.5% | shipped |
| greedy + `aff/deg` containment tiebreak | 18.3% | ~94% of ceiling |
| **METIS min-cut (idealized ceiling)** | **19.4%** | 3.4 s; parts 218–233 (cap 227) |

### 1M × 768-dim, shuffled (page_capacity 292)
| Ordering | Intra-page | Time |
|---|---|---|
| greedy (dense, affinity) | 17.4% | 11 s |
| greedy + `aff/deg` containment tiebreak | 18.8% | 18 s |
| **METIS min-cut (idealized ceiling)** | **21.6%** | 71 s |

METIS 1M: k=3425, part sizes 282–300 (mean 292), edgecut 13.7M, HNSW build 389 s.

### Key observations
- **The greedy→ceiling gap widens with scale.** 100K: greedy ~90% / tiebreak ~94% of ceiling
  (≤2 pp to gain). 1M: ceiling rises to 21.6% while greedy stays ~17.4%, so the tiebreak (18.8%)
  is only **~87%** of ceiling — **~2.8 pp of headroom** a partitioner captures.
- **The ceiling is still low in absolute terms** — even optimal partitioning leaves ~78–80% of
  edges crossing pages. This corpus is structurally *diffuse* (weak clustering: Hopkins 0.69,
  silhouette ~0.02 per the cluster-tendency report), which caps extractable page-locality for
  *any* method. A more clustered corpus would raise the ceiling and the potential gain.
- **These METIS numbers are OPTIMISTIC (label-based).** They score `membership[u]==membership[v]`,
  i.e. "if each part were its own page." They do NOT account for (a) parts that exceed pageCapacity
  spilling across a page boundary, or (b) how parts are sequenced into the dense ordinal line. A
  real page-aligned reorder will land *below* the ceiling — how far below is the open question (§5).

---

## 3. How to implement in C++ (native / JNI)

METIS is a C library → links directly into `jni/` next to Faiss. Sketch:

1. **Build CSR from the layer-0 graph** (already materialized during the native build):
   `xadj` (offsets), `adjncy` (neighbors). Must be **undirected** → symmetrize (union of directed
   edges); dedup.
2. **Partition:** `METIS_PartGraphKway(&nvtxs, &ncon=1, xadj, adjncy, NULL, NULL, NULL, &nparts,
   NULL, ubvec, options, &objval, part)` with `nparts = ⌈N/pageCapacity⌉`. Pin `METIS_OPTION_SEED`
   for reproducibility.
3. **Enforce the page cap:** METIS balances but does not hard-cap; tune `ubvec`/`METIS_OPTION_UFACTOR`
   toward tight balance, then post-split any part > pageCapacity (a part of 300 in a 292 cap must be
   split or it spills). Tighter balance trades away some cut quality.
4. **Sequence the parts (critical — this realizes the benefit):** METIS gives *which* part, not an
   order. Build the **quotient graph** (parts as super-nodes, weighted by inter-part edge counts)
   and order the parts so edge-heavy neighbors are adjacent (BFS/greedy over the quotient graph, or
   even recurse the same greedy page-growing on super-nodes). Then assign each part a contiguous
   ordinal block in that sequence. Poor part sequencing turns inter-part edges into long cross-page
   jumps and erodes the gain.
5. **Emit `ordToPhysicalOrdMap`** in that order — same downstream path as greedy (write `.veqlo`
   in physical order, permutation to `.vemlo`). Keep hub seeding for the entry point.

Reference: the offline validation used `pymetis.part_graph(k, xadj=xadj, adjncy=adjncy)` +
`locality_simulation_dense.greedy_page_growing_permutation` for the baseline.

---

## 4. Challenges & costs

- **Label→page-aligned conversion (steps 3–4)** is the crux and the source of ceiling erosion.
- **Memory at scale:** METIS is fully in-memory. Symmetric CSR at 113M × ~64 degree ≈ tens of GB
  for `adjncy` alone, plus multilevel coarsening overhead — a large step up from greedy's light,
  single-pass footprint. Gate on the indexing host's memory.
- **Build time:** 71 s @1M for the partition alone (vs greedy 7–18 s); grows with N. Offline, so
  tolerable, but not free.
- **Determinism:** pin the METIS seed so segment builds are reproducible.
- **Native dependency:** adds a C library to the JNI CMake/packaging across platforms.
- **Licensing (gating):** METIS must be license-compatible with OpenSearch (Apache-2.0). Modern
  METIS (KarypisLab) is Apache-2.0 — **must be verified** for the exact vendored version before use;
  historically METIS licensing was restrictive.

### Alternatives to linking METIS
- **Home-grown multilevel partitioner** in C++ (coarsen → initial partition → Fiduccia–Mattheyses /
  Kernighan–Lin refinement) — more work, no external license/dependency.
- **Page-aware refinement on top of greedy** — local KL/FM boundary swaps across adjacent pages to
  reduce cut, keeping greedy's contiguity. Smaller change; captures part of the gap without full
  partitioning. Likely the best effort/reward tradeoff to try first.

---

## 5. Decision gate (what to do when we come back)

Do **not** commit to METIS yet. Before implementing:

1. **Prototype the full page-aligned reorder** (partition → cap-enforce → part-sequence → dense
   permutation) in Python and measure its *page-aligned* intra-page ratio — i.e. discount the
   optimistic label-based ceiling to a real number.
2. **Measure actual read amplification at ≥10M** (PageTouch / BufferPool) for METIS-reorder vs
   greedy+tiebreak, not just intra-page ratio.
3. **Decide:**
   - If the page-aligned reorder retains a real multi-pp read-amp win over greedy+tiebreak → wire a
     partitioner (METIS or home-grown) into the native build, subject to the licensing/memory gates.
   - If page-alignment eats most of the ceiling gap → stick with greedy + the free `aff/deg`
     tiebreak, and treat the ceiling as "greedy is good enough."
4. Re-run the ceiling on a **more clustered** corpus — the diffuse Cohere data understates the
   potential; a clustered dataset would show whether partitioning's advantage is materially larger.

---

## 6. Immediate, unconditional takeaway
Independent of the METIS decision: **adopt the `aff/deg` containment tiebreak in greedy** (free,
one-line, +0.9–1.5 pp verified at 100K/1M). See `HNSW_Locality_Reordering_Design.md` §9.1.
