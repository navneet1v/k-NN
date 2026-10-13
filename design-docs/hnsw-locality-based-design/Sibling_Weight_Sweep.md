# Sibling-Weight (b) Sweep — Greedy + sibling + aff/deg

Sweeping the sibling (shared-neighbor / Gorder co-visitation) weight **b** in the
`Greedy + sibling(b) + aff/deg tiebreak` ordering to find the best value. Parked here for later use;
not yet folded into the main design doc.

**Setup:** `scripts/locality_compare.py`, `documents-1m.hdf5` (1M × 768-dim, shuffled ingest),
**M=16, efC=256, efS=256, all 10,000 test queries** (avg 5,070 nodes visited/query,
recall@100=0.9116, layout-invariant). One cached index build; the 10k-query search pass is
layout-invariant and reused across all b. `score(v) = affinity(v) + b·sibling(v) + affinity(v)/degree(v)`.
Numba-JIT `greedy_sibling_perm`; reorder cost is flat (~14–16 s) across all b.

Reproduce:
```bash
cd scripts
python locality_compare.py --dataset documents-1m.hdf5 --nq 10000 --skip-gorder --skip-metis \
    --sibling-sweep "0,0.25,0.5,1,2,4,8"                       # 32 KB (default page)
python locality_compare.py --dataset documents-1m.hdf5 --nq 10000 --page-size 8192 \
    --skip-gorder --skip-metis --sibling-sweep "0,0.25,0.5,1,2,4,8"   # 8 KB
```

Metric = **pages-touched-per-query** (read-amp = pages·pageSize / (visited·recordSize), recordSize=112 B).
Lower ptpq / higher reduction = better. Baselines (Insertion, plain Greedy+aff/deg) included for reference.

---

## 32 KB pages (page_capacity=292)

| Method | intra-page | ptpq mean | p50 | p90 | p99 | p100 | reduction | read-amp | reorder-s |
|---|---|---|---|---|---|---|---|---|---|
| Insertion (baseline) | 0.04% | 2586.9 | 2625 | 2823 | 2911 | 3006 | — | 151.67× | 0.0 |
| Greedy + aff/deg tiebreak | 18.86% | 1192.3 | 1211 | 1590 | 1843 | 2160 | +53.9% | 68.14× | 11.3 |
| Greedy + sibling(b=0.0) | 19.01% | 1169.7 | 1187 | 1568 | 1837 | 2115 | +54.8% | 66.78× | 16.0 |
| Greedy + sibling(b=0.25) | 18.24% | 1014.3 | 1023 | 1381 | 1632 | 1993 | +60.8% | 57.92× | 14.0 |
| Greedy + sibling(b=0.5) | 17.29% | 1001.9 | 1009 | 1364 | 1616 | 1958 | +61.3% | 57.21× | 13.7 |
| Greedy + sibling(b=1.0) | 16.34% | 986.7 | 994 | 1342 | 1592 | 1941 | +61.9% | 56.34× | 14.8 |
| **Greedy + sibling(b=2.0)** | 15.28% | 984.7 | 993 | 1339 | 1588 | **1928** | +61.9% | 56.27× | 15.1 |
| **Greedy + sibling(b=4.0)** | 14.77% | **983.2** | **990** | **1335** | **1581** | 1941 | **+62.0%** | **56.18×** | 14.6 |
| Greedy + sibling(b=8.0) | 14.64% | 986.4 | 994 | 1340 | 1587 | 1936 | +61.9% | 56.35× | 14.0 |

## 8 KB pages (page_capacity=73)

| Method | intra-page | ptpq mean | p50 | p90 | p99 | p100 | reduction | read-amp | reorder-s |
|---|---|---|---|---|---|---|---|---|---|
| Insertion (baseline) | 0.01% | 4198.7 | 4261 | 4903 | 5252 | 5617 | — | 60.86× | 0.0 |
| Greedy + aff/deg tiebreak | 13.70% | 2188.4 | 2229 | 2967 | 3500 | 3948 | +47.9% | 31.05× | 11.8 |
| Greedy + sibling(b=0.0) | 13.79% | 2136.1 | 2174 | 2912 | 3445 | 3952 | +49.1% | 30.30× | 14.8 |
| Greedy + sibling(b=0.25) | 14.06% | 1837.1 | 1862 | 2508 | 2992 | 3558 | +56.2% | 26.09× | 16.0 |
| Greedy + sibling(b=0.5) | 13.11% | 1827.1 | 1853 | 2492 | 2970 | 3537 | +56.5% | 25.95× | 15.5 |
| Greedy + sibling(b=1.0) | 11.80% | 1810.7 | 1835 | 2473 | 2947 | 3588 | +56.9% | 25.72× | 15.6 |
| **Greedy + sibling(b=2.0)** | 10.06% | **1807.4** | **1830** | **2466** | **2935** | 3581 | **+57.0%** | **25.69×** | 14.2 |
| Greedy + sibling(b=4.0) | 9.28% | 1810.1 | 1837 | 2468 | 2943 | 3558 | +56.9% | 25.73× | 15.4 |
| Greedy + sibling(b=8.0) | 9.06% | 1808.8 | 1833 | 2468 | 2934 | 3567 | +56.9% | 25.70× | 18.8 |

---

## Observations

1. **Same curve shape at both page sizes:** a large jump from b=0 → b=0.25, diminishing returns
   through b=0.5, then a **plateau from b≈1 onward** (mean flat within ~4 pages). Marginal minimum
   around **b=2** (best at both 32 KB and 8 KB on mean/p99); b=4 is a hair better on mean @32 KB but
   within noise.
2. **b=0.5 already captures ~99% of the win.** Going 0.5 → 2.0 gains only ~0.5–1% more pages/query.
   No reason to over-tune to a simulation.
3. **b=0 is NOT plain greedy** (32 KB: 1169.7 vs 1192.3; 8 KB: 2136.1 vs 2188.4). Even with the
   sibling term zeroed, pure-sibling nodes are still *enrolled* as candidates, so edgeless co-visited
   nodes backfill page tails — a small standalone win from candidate enrollment alone.
4. **intra-page drops monotonically as b rises** (32 KB 19.0% → 14.6%; 8 KB 13.8% → 9.1%) while
   pages/query keeps improving — reconfirming intra-page edge ratio is anti-correlated with the
   production-relevant metric (§9 of the design doc).
5. **Reorder cost is flat (~14–16 s) regardless of b** — no build-time penalty for a higher weight.

## Recommendation

**b = 1.0** — a round number on the plateau, robustly near-optimal at both 32 KB and 8 KB, with the
best tail among the "safe" values. b=2 is marginally better but past the point of diminishing returns.
Current script/C++ default is 0.5 (already ~99% of the way there); a move to 1.0 is a low-risk
refinement to validate in a cluster A/B before changing the shipped default.
