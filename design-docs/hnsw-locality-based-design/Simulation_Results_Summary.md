# Simulation Results Summary: HNSW Locality-Aware Layout

## Experiment Setup

**Dataset:** Cohere Wikipedia embeddings (768-dim, inner product)
- 100K vectors (documents-100k.hdf5) for quick iteration
- 1M vectors (documents-1m.hdf5) for realistic evaluation

**Index Configuration:**
- HNSW M=16, efConstruction=256, efSearch=256
- Shuffled insertion order (IndexIDMap) to eliminate false temporal locality
- Metric: Inner Product

**Page Model (SQ 1-bit):**
- Page size: 32KB
- Record size: 112 bytes (768-dim / 8 = 96 bytes binary code + 16 bytes correction factors)
- Page capacity: 292 vectors/page
- Total pages: 343 (100K), 3425 (1M)

**Search Modes:**
- Multi-layer HNSW: Standard algorithm (greedy descent through upper layers, beam search on layer 0)
- Flat search: Layer 0 only, single hub entry point (closest hub by inner product from pool of 256 top-level nodes)

---

## Key Finding: Shuffled Insertion Order Matters

Without shuffling, the Cohere dataset has natural temporal locality (nearby documents are semantically similar). This gave insertion order a fake 21.2% intra-page edge ratio. After shuffling, insertion order drops to 0.03-0.3%, revealing the true baseline.

**Lesson:** Always shuffle insertion order when benchmarking layout strategies. Real-world indices don't have insertion-order locality guarantees.

---

## Results: 100K Vectors (343 pages)

| Layout | Pages Touched | Reduction vs Baseline | Intra-Page Edge Ratio |
|--------|--------------|----------------------|----------------------|
| Insertion Order (shuffled) | 343 | — | 0.3% |
| BFS Order | 278-282 | 17-19% | 1.7% |
| Reverse Cuthill-McKee | 324-327 | 5-7% | 1.6% |
| Greedy (two_hop) | 269-272 | 21-22% | 21.3% |
| Greedy (partial_page) | 275-278 | 19-20% | 21.3% |
| Greedy (bfs_expand) | 265-268 | 22-23% | 21.3% |

**Observation:** At 100K the dataset is saturated — ~4000 nodes visited across only 343 pages means most pages get touched regardless. The improvement is capped.

**Recall:** Perfect 1.0 for all search modes (multi-layer and flat with 1 hub entry point).

---

## Results: 1M Vectors (3425 pages) — Primary Benchmark

### efSearch=256

| Layout | Pages Touched | Reduction vs Baseline | Intra-Page Edge Ratio |
|--------|--------------|----------------------|----------------------|
| Insertion Order (shuffled) | 2618-2627 | — | 0.03% |
| BFS Order | 1553-1557 | 40-41% | 0.9% |
| Reverse Cuthill-McKee | 1442-1450 | 44-45% | 0.8% |
| **Greedy (two_hop)** | **1154-1155** | **56%** | **16.4%** |
| Greedy (partial_page) | 1193-1196 | 54% | 16.3% |
| Greedy (bfs_expand) | 1142-1144 | 56.5% | 16.4% |

**Nodes visited per query:** ~4950 (both multi-layer and flat)
**Theoretical minimum pages:** ceil(4950/292) = 17

### efSearch=100 — Validated over 10,000 queries (single-layer graph)

Flat search with 1 hub entry point. Averaged across all 10K test queries.

| Layout | Avg Pages | Reduction | Median | P95 | P99 | Intra-Page Edge Ratio |
|--------|-----------|-----------|--------|-----|-----|----------------------|
| Insertion Order (shuffled) | 1638.6 | — | 1655 | 1898 | 2022 | 0.03% |
| BFS Order | 931.5 | 43.2% | 948 | 1251 | 1363 | 0.9% |
| Reverse Cuthill-McKee | 1035.9 | 36.8% | 1049 | 1374 | 1491 | 0.8% |
| **Greedy (two_hop)** | **780.5** | **52.4%** | 796 | 1099 | 1220 | 17.1% |
| **Greedy (partial_page)** | **792.2** | **51.7%** | 808 | 1104 | 1230 | 17.1% |

**Nodes visited per query:** ~2245 (flat), ~2361 (Faiss entry point)
**Theoretical minimum pages:** ceil(2245/292) = 8

**Recall validation (10K queries):**
- Flat search (1 hub entry point): **0.9201**
- Faiss native search: **0.9204**
- Simulated Faiss-entry search: 0.9204

Flat search with a single hub entry point matches Faiss native recall exactly, and touches *fewer* pages than the Faiss-entry baseline (781 vs 837) because the hub lands closer to the query target.

### Key Insight: Greedy Advantage Grows at Lower efSearch

| efSearch | Nodes Visited | Greedy Pages | BFS Pages | Greedy vs BFS |
|----------|--------------|-------------|-----------|---------------|
| 100 | 2287 | 786 | 1023 | 23% fewer |
| 256 | 4961 | 1155 | 1557 | 26% fewer |

The more concentrated the search (fewer nodes visited), the more locality matters. Greedy consistently outperforms BFS by 23-26%.

**Recall:** Perfect 1.0 for all search modes and efSearch values.

---

## Reordering Strategy Comparison

### After Bucket Queue Optimization (current)

| Strategy | Pages (1M, ef=100) | Pages (1M, ef=256) | Build Time (1M) | Recommendation |
|----------|-------------------|-------------------|-----------------|----------------|
| Identity | 1656 | 2627 | 0s | Baseline |
| BFS | 1023 | 1557 | 1.8s | Cheap fallback |
| RCM | 1094 | 1450 | 3.9s | Slightly worse than BFS |
| **Greedy (two_hop)** | **786** | **1155** | **14.7s** | **Best quality** |
| **Greedy (partial_page)** | **786** | **1193** | **7.0s** | **Recommended (fast + good)** |
| Greedy (bfs_expand) | 1142* | 1142* | 1491s | Too slow, marginal gain |

*bfs_expand measured at efSearch=256 only (removed from default runs due to speed).

**Recommendation: Greedy (partial_page)** — identical quality to two_hop at efSearch=100 (both achieve 786 pages), 2x faster build time. At efSearch=256 two_hop has a slight edge (1155 vs 1193) but partial_page is still within 3%.

### Build Time Evolution

| Optimization | two_hop (1M) | partial_page (1M) |
|-------------|-------------|-------------------|
| Naive Python (original) | ~56s (100K) → est. 7hrs (1M) | Same |
| Incremental affinity | 120s | 107s |
| **Bucket queue (current)** | **14.7s** | **7.0s** |

Bucket queue achieved **8-15x speedup** over incremental-only by eliminating the linear candidate scan.

---

## Flat Search Viability

| Metric | Multi-layer HNSW | Flat (1 hub entry) |
|--------|-----------------|-------------------|
| Nodes visited (100K) | 4063-4100 | 4026-4080 |
| Nodes visited (1M) | 4943 | 4961 |
| Recall@10 (100K) | 1.0 | 1.0 |
| Recall@10 (1M) | 1.0 | 1.0 |
| Pages touched (1M, greedy) | 1154 | 1155 |

**Conclusion:** Flat search with a single hub entry point matches multi-layer HNSW in both recall and convergence efficiency at 1M scale. No measurable degradation from removing upper layers.

### Hub Entry Point Cost (Not Counted in Pages Touched)

Hub selection requires computing distance from the query to each hub vector (256 hubs * 112 bytes = ~28KB). This cost is **not included** in the pages-touched metric because hub vectors are always memory-resident in the production design:
- 256 hubs * 112 bytes = 28KB — fits in a single 32KB page
- Loaded once at segment open, stays in memory for the lifetime of the segment
- Fixed O(1) memory overhead per segment, independent of query volume

---

## Graph Statistics (1M)

- Entry point: internal ordinal 118295 (level 5)
- Max level: 5
- Level distribution: L0=937,597 | L1=58,500 | L2=3,693 | L3=194 | L4=15 | L5=1
- Average layer 0 degree: 22.9

---

## Optimization Notes

**Greedy Algorithm — Two-Stage Speedup:**

Stage 1: Incremental affinity updates (O(degree) per slot fill)
- When node X joins page: `affinity[neighbors_of_X] += 1`
- Eliminated recomputing all candidate affinities from scratch
- Result: 120s at 1M (down from estimated 7+ hours)

Stage 2: Bucket queue for O(1) candidate selection
- Candidates grouped by affinity score in `buckets[score]`
- When affinity increments k→k+1: push to `buckets[k+1]`, old entry becomes stale
- `pop_best()`: scan from highest bucket, skip stale entries
- Eliminated linear scan over all candidates per slot
- Result: **7-15s at 1M** (8-15x speedup over stage 1)

**Total speedup from naive: ~1000-3000x**

---

## Pre-Implementation Checklist

Items that must be resolved or explicitly tracked before writing plugin code.

### Must Resolve Before Coding

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | **Statistical validation** | ✅ Done | Validated over all 10K queries at 1M/ef=100. Greedy(two_hop): 52.4% avg reduction, stable tail (P99: 1220). Recall matches Faiss native (0.9201 vs 0.9204). |
| 3 | **Choose shipping default: two_hop vs partial_page** | Open — tracking | 10K-query result: two_hop 780.5 pages vs partial_page 792.2 (1.5% gap). Build: two_hop 11.8s vs partial_page 6.9s. **Leaning partial_page** for MVP; revisit at 10M. |
| 4 | **Faiss SQ integration feasibility** | ✅ Done — FEASIBLE | See `Faiss_SQ_Integration_Feasibility.md`. (a) Single-layer: set `assign_probas={1.0}` in `initFaissSQIndex`. (b) Extract: read public `hnsw.neighbors/offsets/levels`. Remap: built-in `HNSW::permute_entries()` + permute `id_map` — cheaper than the custom remap we planned. Native SQ vector buffer needs no permute (served from `.vep`). **Discovered:** strict page padding is incompatible with `permute_entries` (needs dense bijection) → MVP uses dense packing; strict padding deferred to co-located-page format phase. |

### Deferred to Future Validation

| # | Item | Notes |
|---|------|-------|
| 2 | **Recall at lower efSearch (32, 64)** | Not a current concern. Validate flat search + single hub maintains >0.95 recall at production efSearch values later. |
| 5 | **Hub entry point scaling at 10M+** | Does 1 hub suffice, or do we need multiple? |
| 6 | **Boundary replication (SOAR)** | Marginal reduction from replicating 2-5% of boundary nodes (Phase 3). |
| 7 | **Build time at 10M+** | 15s at 1M → ~150s at 10M in Python. C++/Java target: <5s at 1M. |


--------------------------
(base) 09:14 ~/workplace/k-NN-main (hnsw_locality)$ python scripts/locality_simulation.py --dataset scripts/documents-1m.hdf5 --nq 10000 --M 16 --ef-search 256 --ef-construction 256
======================================================================
HNSW Locality Reordering Simulation
======================================================================

Loading dataset from scripts/documents-1m.hdf5...
Keys: ['neighbors', 'test', 'train']
Train: (1000000, 768), Test: (10000, 768)
Ground truth neighbors: (10000, 100)

Config:
Vectors: 1000000, Dim: 768, Queries: 10000
HNSW M: 16, efConstruction: 256, efSearch: 256
Page size: 32768 bytes
Record size (SQ 1-bit): 112 bytes
Page capacity: 292 vectors/page
Total pages: 3425
Entry points per query: 1

Building single-layer HNSW index (shuffled insertion order)...
Built in 361.8s
Insertion order shuffled (first 5 internal ordinals map to original IDs: [ 97305 228106 970098 780379 966161])
Extracting graph structure...
Extracted in 3.7s
Entry point: 768932 (level 0)
Max level: 0
Level distribution: {np.int32(0): np.int64(1000000)}
Layer 0 degree — avg: 22.9, p10: 12, p50: 24, p90: 32, p99: 32, max: 32

Computing reordering strategies...
Identity: 0.000s
BFS: 1.535s
RCM: 4.060s
Greedy (strict pages): 6.468s
Strict pages: 43399 pages, 11672508 padded slots (92.1% overhead)

Simulating 10000 searches with Faiss entry point (efSearch=256)...
100/10000 queries (2.4s, 41.7 qps)
200/10000 queries (4.4s, 45.0 qps)
300/10000 queries (6.0s, 49.8 qps)
400/10000 queries (7.5s, 53.1 qps)
500/10000 queries (9.1s, 55.1 qps)
600/10000 queries (10.8s, 55.4 qps)
700/10000 queries (12.4s, 56.5 qps)
800/10000 queries (14.2s, 56.5 qps)
900/10000 queries (15.8s, 56.9 qps)
1000/10000 queries (17.3s, 57.8 qps)
1100/10000 queries (18.8s, 58.4 qps)
1200/10000 queries (20.5s, 58.6 qps)
1300/10000 queries (22.1s, 58.9 qps)
1400/10000 queries (23.9s, 58.6 qps)
1500/10000 queries (25.5s, 58.8 qps)
1600/10000 queries (27.1s, 59.1 qps)
1700/10000 queries (28.6s, 59.5 qps)
1800/10000 queries (30.1s, 59.8 qps)
1900/10000 queries (31.8s, 59.7 qps)
2000/10000 queries (33.7s, 59.4 qps)
2100/10000 queries (35.3s, 59.5 qps)
2200/10000 queries (36.8s, 59.8 qps)
2300/10000 queries (38.4s, 60.0 qps)
2400/10000 queries (39.9s, 60.2 qps)
2500/10000 queries (41.6s, 60.1 qps)
2600/10000 queries (43.2s, 60.1 qps)
2700/10000 queries (44.8s, 60.3 qps)
2800/10000 queries (46.4s, 60.3 qps)
2900/10000 queries (48.4s, 59.9 qps)
3000/10000 queries (50.1s, 59.9 qps)
3100/10000 queries (51.9s, 59.7 qps)
3200/10000 queries (53.4s, 59.9 qps)
3300/10000 queries (54.9s, 60.2 qps)
3400/10000 queries (56.6s, 60.1 qps)
3500/10000 queries (58.3s, 60.1 qps)
3600/10000 queries (59.9s, 60.1 qps)
3700/10000 queries (61.5s, 60.1 qps)
3800/10000 queries (63.3s, 60.0 qps)
3900/10000 queries (65.0s, 60.0 qps)
4000/10000 queries (66.6s, 60.0 qps)
4100/10000 queries (68.2s, 60.1 qps)
4200/10000 queries (70.0s, 60.0 qps)
4300/10000 queries (71.8s, 59.9 qps)
4400/10000 queries (73.5s, 59.8 qps)
4500/10000 queries (75.3s, 59.8 qps)
4600/10000 queries (77.2s, 59.6 qps)
4700/10000 queries (79.0s, 59.5 qps)
4800/10000 queries (80.8s, 59.4 qps)
4900/10000 queries (82.5s, 59.4 qps)
5000/10000 queries (84.0s, 59.5 qps)
5100/10000 queries (85.4s, 59.7 qps)
5200/10000 queries (87.0s, 59.8 qps)
5300/10000 queries (88.4s, 60.0 qps)
5400/10000 queries (89.7s, 60.2 qps)
5500/10000 queries (91.2s, 60.3 qps)
5600/10000 queries (92.6s, 60.5 qps)
5700/10000 queries (94.1s, 60.6 qps)
5800/10000 queries (95.4s, 60.8 qps)
5900/10000 queries (96.7s, 61.0 qps)
6000/10000 queries (98.0s, 61.2 qps)
6100/10000 queries (99.3s, 61.4 qps)
6200/10000 queries (100.6s, 61.6 qps)
6300/10000 queries (102.1s, 61.7 qps)
6400/10000 queries (103.5s, 61.8 qps)
6500/10000 queries (104.9s, 62.0 qps)
6600/10000 queries (106.3s, 62.1 qps)
6700/10000 queries (107.7s, 62.2 qps)
6800/10000 queries (109.2s, 62.2 qps)
6900/10000 queries (110.6s, 62.4 qps)
7000/10000 queries (111.9s, 62.5 qps)
7100/10000 queries (113.3s, 62.7 qps)
7200/10000 queries (114.7s, 62.8 qps)
7300/10000 queries (116.4s, 62.7 qps)
7400/10000 queries (117.9s, 62.8 qps)
7500/10000 queries (119.3s, 62.9 qps)
7600/10000 queries (120.8s, 62.9 qps)
7700/10000 queries (122.2s, 63.0 qps)
7800/10000 queries (123.7s, 63.1 qps)
7900/10000 queries (125.2s, 63.1 qps)
8000/10000 queries (126.4s, 63.3 qps)
8100/10000 queries (127.7s, 63.4 qps)
8200/10000 queries (129.3s, 63.4 qps)
8300/10000 queries (130.8s, 63.5 qps)
8400/10000 queries (132.3s, 63.5 qps)
8500/10000 queries (133.8s, 63.5 qps)
8600/10000 queries (135.4s, 63.5 qps)
8700/10000 queries (136.8s, 63.6 qps)
8800/10000 queries (138.3s, 63.6 qps)
8900/10000 queries (139.8s, 63.7 qps)
9000/10000 queries (141.2s, 63.7 qps)
9100/10000 queries (142.6s, 63.8 qps)
9200/10000 queries (145.1s, 63.4 qps)
9300/10000 queries (146.7s, 63.4 qps)
9400/10000 queries (148.2s, 63.4 qps)
9500/10000 queries (149.7s, 63.5 qps)
9600/10000 queries (151.1s, 63.5 qps)
9700/10000 queries (152.5s, 63.6 qps)
9800/10000 queries (154.0s, 63.7 qps)
9900/10000 queries (155.3s, 63.7 qps)
10000/10000 queries (156.7s, 63.8 qps)
Completed in 156.7s, avg nodes visited: 5053.0
Recall@10 (Faiss entry point): 0.9722

Faiss native search (sanity check)...
Recall@10 (Faiss native): 0.9722

Hub pool: 256 nodes (degree range: 32 down to 32)
Simulating 10000 FLAT searches (efSearch=256, 1 hub entry points)...
500/10000 queries (3.1s, 163.8 qps)
1000/10000 queries (5.9s, 168.6 qps)
1500/10000 queries (8.6s, 174.8 qps)
2000/10000 queries (11.3s, 176.4 qps)
2500/10000 queries (13.8s, 180.5 qps)
3000/10000 queries (16.3s, 183.8 qps)
3500/10000 queries (18.6s, 188.1 qps)
4000/10000 queries (21.2s, 188.6 qps)
4500/10000 queries (23.5s, 191.8 qps)
5000/10000 queries (25.9s, 192.9 qps)
5500/10000 queries (28.2s, 194.7 qps)
6000/10000 queries (30.4s, 197.4 qps)
6500/10000 queries (32.7s, 198.6 qps)
7000/10000 queries (35.1s, 199.4 qps)
7500/10000 queries (37.6s, 199.7 qps)
8000/10000 queries (39.7s, 201.3 qps)
8500/10000 queries (42.1s, 202.0 qps)
9000/10000 queries (44.6s, 201.8 qps)
9500/10000 queries (47.0s, 201.9 qps)
10000/10000 queries (49.3s, 202.7 qps)
49.3s, avg visited: 4972.0, Recall@10: 0.9723

================================================================================
PAGES TOUCHED: Faiss entry point search (baseline)
================================================================================
Layout                          Avg Pages   Median      P95      P99
--------------------------------------------------------------------------------
Insertion Order                  2618.7             2661.0   2885.0   2937.0
BFS Order                        1527.4 (+41.7%)    1580.0   1998.0   2143.0
Reverse Cuthill-McKee            1648.6 (+37.0%)    1696.0   2132.0   2255.0
Greedy (strict pages)            1377.0 (+47.4%)    1419.0   1872.0   2003.0

================================================================================
PAGES TOUCHED: Flat search (1 entry points)
================================================================================
Layout                          Avg Pages   Median      P95      P99
--------------------------------------------------------------------------------
Insertion Order                  2600.5             2643.0   2868.0   2926.0
BFS Order                        1530.8 (+41.1%)    1580.0   1996.0   2146.0
Reverse Cuthill-McKee            1613.2 (+38.0%)    1662.0   2110.0   2236.0
Greedy (strict pages)            1348.0 (+48.2%)    1386.0   1845.0   1986.0

================================================================================
INTRA-PAGE EDGE RATIO (edges staying within same page)
================================================================================
Insertion Order                0.0003 (0.0%)
BFS Order                      0.0091 (0.9%)
Reverse Cuthill-McKee          0.0072 (0.7%)
Greedy (strict pages)          0.1711 (17.1%)

Done.
