# Simulation Results — M=16 (production config)

**Config:** M=16, ef_construction=256, ef_search=256. Single-layer HNSW, shuffled ingest, dense
packing. Metric page = 32 KB; `page_capacity = 32768 / recordSize` (227 @1024-dim, 292 @768-dim).
This supersedes earlier simulation numbers that were mistakenly run at **M=32** — AOSS and all
production indices use **M=16** (layer-0 max degree = 2·M = 32).

Metrics: **intra-page edge ratio** (fraction of layer-0 edges with `perm[u]//pc == perm[v]//pc`;
for METIS, membership equality) and **pages/query** (distinct 32 KB pages touched per query, from
multilayer beam search at ef_search=256), with p99. The per-scale tables below use `nq=500`; the
Gorder-vs-greedy question is additionally validated on the **full 10,000-query** test set (see the
"Full test-set validation" section).

---

## 100K × 1024-dim (`dataset.hdf5`, pc=227, layer0 avg deg=28.1/max 32, avg visited=4400)

| Method | intra-page | pages/query | reduction vs insertion | p99 |
|---|---|---|---|---|
| Insertion (baseline) | 0.22% | 440.8 | — | 441 |
| BFS | 1.42% | 409.3 | +7.2% | 439 |
| RCM | 1.36% | 417.5 | +5.3% | 440 |
| Gorder (w=32) | 12.86% | 362.2 | +17.8% | 422 |
| Greedy (affinity) | 18.13% | 372.6 | +15.5% | 426 |
| **Greedy + aff/deg tiebreak** | **18.83%** | **362.2** | **+17.8%** | 421 |
| METIS min-cut (ceiling) | 21.41% | — | — | (parts 219–233) |

## 1M × 768-dim (`documents-1m-shuffled.hdf5`, pc=292, layer0 avg deg=22.9/max 32, avg visited=5263)

| Method | intra-page | pages/query | reduction vs insertion | p99 |
|---|---|---|---|---|
| Insertion (baseline) | 0.03% | 2673.1 | — | 2938 |
| BFS | 0.91% | 1480.1 | +44.6% | 2117 |
| RCM | 0.72% | 1583.6 | +40.8% | 2162 |
| Gorder (w=32) | 12.56% | 1138.9 | +57.4% | 1781 |
| Greedy (affinity) | 17.06% | 1399.9 | +47.6% | 1994 |
| **Greedy + aff/deg tiebreak** | **18.94%** | **1206.8** | **+54.9%** | 1822 |
| METIS min-cut (ceiling) | 22.73% | — | — | (parts 283–300) |

---

## Full test-set validation — ALL 10,000 queries (1M, M=16, ef=256)

Re-ran the Gorder-vs-greedy question on the **entire 10,000-query test set** (not nq=500) to rule out
sampling noise. Fresh single-layer build; avg nodes visited/query = 5054; 10k searches ≈ 144 s.

| Method | intra-page | pages/query | reduction vs insertion | p99 | p50 |
|---|---|---|---|---|---|
| Insertion | 0.03% | 2620.1 | — | 2936 | 2662 |
| **Greedy + aff/deg tiebreak** | 19.00% | 1179.4 | +55.0% | 1838 | 1198 |
| **Gorder (w=32)** | 12.55% | **1113.3** | **+57.5%** | **1775** | **1126** |

**Confirmed: Gorder beats greedy+tiebreak on pages/query on the full query set** — 1113.3 vs 1179.4
(−5.6%), and also at p50 (1126 vs 1198) and p99 (1775 vs 1838) — despite the much lower intra-page
ratio (12.55% vs 19.00%). The nq=500 signal (1138.9 vs 1206.8) reproduces at full scale, so it is not
sampling noise. Intra-page edge ratio and pages/query genuinely disagree, and pages/query is the one
that maps to read amplification.

---

## Key observations (M=16)

1. **Containment `aff/deg` tiebreak reproduces the win at M=16:** intra-page +0.7 pp @100K
   (18.13→18.83) and **+1.9 pp @1M** (17.06→18.94); pages/query improves too (100K 372.6→362.2;
   1M 1399.9→1206.8, i.e. +47.6%→+54.9% reduction). Confirmed: adopt it.
2. **Greedy still wins on intra-page** (18.1–18.9% vs BFS/RCM ~1%, Gorder ~12.6%), and reaches
   ~85–88% of the METIS min-cut ceiling (21.4% @100K, 22.7% @1M).
3. **⚠ NOTABLE — on pages/query, Gorder BEATS greedy at M=16/ef=256 (confirmed on the full 10k-query
   set):** 100K tied (saturated); 1M Gorder **1113.3 < greedy+tiebreak 1179.4** (−5.6%; also lower
   p50/p99), despite Gorder's much lower intra-page ratio (12.55% vs 19.00%). The nq=500 signal
   (1138.9 vs 1206.8) reproduces on all 10,000 queries — not sampling noise. The earlier "greedy
   dominates / Gorder rejected" conclusion was based on **intra-page at M=32** — the wrong proxy:
   intra-page weighs all edges equally, but pages/query depends only on the *traversed* subgraph, and
   Gorder's sibling score (co-visited nodes) aligns with traversal better than greedy's all-edge
   page-packing. Greedy still ships (validated on AOSS −36% read-amp, single-pass, ~6× faster; Gorder
   is sim-only and O(n·deg²)), but "greedy strictly dominates" is false on the production-relevant
   metric — **a cluster A/B of Gorder vs greedy on actual read-amp (PageTouch) at ≥10M is warranted.**
4. **Numbers differ from the old M=32 sims:** M=16 is sparser (avg deg ~23–28 vs ~55), so intra-page
   is similar-to-higher and pages/query reductions at 1M are large (+55%). The 100K reductions look
   smaller (+18%) than earlier because ef_search=256 makes a 100K index a near-saturated regime (a
   query visits ~4400 nodes across ~82% of the 441 pages regardless of layout) — small segment +
   high ef = little headroom.

---

## How to reproduce

Scripts (committed in `scripts/`): `m16_locality_compare.py` (full comparison) and
`m16_locality_allqueries.py` (all-queries Gorder-vs-greedy). Both use `locality_simulation_dense.py`
+ `gorder_fast.py` + `pymetis`.

```bash
cd /Users/navneev/workplace/k-NN-main/scripts
# full comparison (identity/BFS/RCM/Gorder w=32/greedy/greedy+tiebreak/METIS), nq=500:
python m16_locality_compare.py dataset.hdf5 500                 # 100K (fast)
python m16_locality_compare.py documents-1m-shuffled.hdf5 500   # 1M (index build ~6 min)

# full 10,000-query validation (insertion / greedy+tiebreak / Gorder w=32):
python m16_locality_allqueries.py documents-1m-shuffled.hdf5
```

The script builds `build_hnsw_index(train, 16, 256, shuffle=True)`, extracts layer-0 neighbors,
computes all 7 orderings, simulates `nq` multilayer searches at ef_search=256, and prints intra-page
+ pages/query + p99 per method. Gorder uses `gorder_fast_permutation(nb, n, 32)` (Numba JIT). METIS
uses `pymetis.part_graph(ceil(n/pc), xadj, adjncy)` on the symmetrized graph.
