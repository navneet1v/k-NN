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

--------------------------- 100K -----------------------------------------------
(base) 15:58 ~/workplace/k-NN-main (hnsw_locality)$ python scripts/locality_simulation.py --dataset scripts/documents-100k.hdf5 --nq 10000 --M 16 --ef-search 256 --ef-construction 256                                                  
======================================================================                                                                                                                                                                   
HNSW Locality Reordering Simulation                                                                                                                                                                                                    
======================================================================

Loading dataset from scripts/documents-100k.hdf5...                                                                                                                                                                                      
Keys: ['neighbors', 'test', 'train']                                                                                                                                                                                                   
Train: (100000, 768), Test: (10000, 768)                                                                                                                                                                                               
Ground truth neighbors: (10000, 100)

Config:                                                                                                                                                                                                                                  
Vectors: 100000, Dim: 768, Queries: 10000                                                                                                                                                                                              
HNSW M: 16, efConstruction: 256, efSearch: 256                                                                                                                                                                                         
Page size: 32768 bytes                                                                                                                                                                                                                 
Record size (SQ 1-bit): 112 bytes                                                                                                                                                                                                      
Page capacity: 292 vectors/page                                                                                                                                                                                                        
Total pages: 343                                                                                                                                                                                                                       
Entry points per query: 1

Building single-layer HNSW index (shuffled insertion order)...                                                                                                                                                                           
Built in 23.4s                                                                                                                                                                                                                         
Insertion order shuffled (first 5 internal ordinals map to original IDs: [45262 49850 30394 90177 95569])                                                                                                                              
Extracting graph structure...                                                                                                                                                                                                            
Extracted in 0.2s                                                                                                                                                                                                                      
Entry point: 46535 (level 0)                                                                                                                                                                                                           
Max level: 0                                                                                                                                                                                                                           
Level distribution: {np.int32(0): np.int64(100000)}                                                                                                                                                                                    
Layer 0 degree — avg: 22.6, p10: 13, p50: 23, p90: 31, p99: 32, max: 32

Computing reordering strategies...                                                                                                                                                                                                       
Identity: 0.000s                                                                                                                                                                                                                       
BFS: 0.118s                                                                                                                                                                                                                            
RCM: 0.342s                                                                                                                                                                                                                            
Greedy (strict pages): 0.519s                                                                                                                                                                                                          
Strict pages: 3722 pages, 986824 padded slots (90.8% overhead)

Simulating 10000 searches with Faiss entry point (efSearch=256)...                                                                                                                                                                       
100/10000 queries (0.6s, 161.1 qps)                                                                                                                                                                                                    
200/10000 queries (1.2s, 161.3 qps)                                                                                                                                                                                                    
300/10000 queries (1.9s, 157.8 qps)                                                                                                                                                                                                    
400/10000 queries (2.5s, 157.1 qps)                                                                                                                                                                                                    
500/10000 queries (3.2s, 158.5 qps)                                                                                                                                                                                                    
600/10000 queries (3.7s, 162.5 qps)                                                                                                                                                                                                    
700/10000 queries (4.2s, 167.6 qps)                                                                                                                                                                                                    
800/10000 queries (4.7s, 170.9 qps)                                                                                                                                                                                                    
900/10000 queries (5.2s, 174.0 qps)                                                                                                                                                                                                    
1000/10000 queries (5.7s, 174.8 qps)                                                                                                                                                                                                   
1100/10000 queries (6.3s, 175.7 qps)                                                                                                                                                                                                   
1200/10000 queries (6.8s, 176.6 qps)                                                                                                                                                                                                   
1300/10000 queries (7.3s, 178.7 qps)                                                                                                                                                                                                   
1400/10000 queries (7.8s, 179.5 qps)                                                                                                                                                                                                   
1500/10000 queries (8.4s, 179.6 qps)                                                                                                                                                                                                   
1600/10000 queries (8.9s, 180.2 qps)                                                                                                                                                                                                   
1700/10000 queries (9.4s, 180.4 qps)                                                                                                                                                                                                   
1800/10000 queries (9.9s, 180.9 qps)                                                                                                                                                                                                   
1900/10000 queries (10.5s, 181.0 qps)                                                                                                                                                                                                  
2000/10000 queries (11.0s, 181.6 qps)                                                                                                                                                                                                  
2100/10000 queries (11.5s, 182.4 qps)                                                                                                                                                                                                  
2200/10000 queries (12.1s, 182.4 qps)                                                                                                                                                                                                  
2300/10000 queries (12.6s, 182.8 qps)                                                                                                                                                                                                  
2400/10000 queries (13.1s, 183.4 qps)                                                                                                                                                                                                  
2500/10000 queries (13.6s, 183.5 qps)                                                                                                                                                                                                  
2600/10000 queries (14.1s, 184.1 qps)                                                                                                                                                                                                  
2700/10000 queries (14.6s, 184.4 qps)                                                                                                                                                                                                  
2800/10000 queries (15.1s, 185.0 qps)                                                                                                                                                                                                  
2900/10000 queries (15.7s, 185.2 qps)                                                                                                                                                                                                  
3000/10000 queries (16.1s, 186.0 qps)                                                                                                                                                                                                  
3100/10000 queries (16.7s, 185.7 qps)                                                                                                                                                                                                  
3200/10000 queries (17.2s, 185.7 qps)                                                                                                                                                                                                  
3300/10000 queries (17.7s, 186.0 qps)                                                                                                                                                                                                  
3400/10000 queries (18.3s, 186.2 qps)                                                                                                                                                                                                  
3500/10000 queries (18.7s, 186.7 qps)                                                                                                                                                                                                  
3600/10000 queries (19.2s, 187.1 qps)                                                                                                                                                                                                  
3700/10000 queries (19.8s, 186.8 qps)                                                                                                                                                                                                  
3800/10000 queries (20.4s, 186.5 qps)                                                                                                                                                                                                  
3900/10000 queries (20.9s, 186.7 qps)                                                                                                                                                                                                  
4000/10000 queries (21.4s, 186.8 qps)                                                                                                                                                                                                  
4100/10000 queries (21.9s, 186.9 qps)                                                                                                                                                                                                  
4200/10000 queries (22.5s, 187.1 qps)                                                                                                                                                                                                  
4300/10000 queries (22.9s, 187.7 qps)                                                                                                                                                                                                  
4400/10000 queries (23.4s, 188.1 qps)                                                                                                                                                                                                  
4500/10000 queries (23.9s, 188.2 qps)                                                                                                                                                                                                  
4600/10000 queries (24.4s, 188.2 qps)                                                                                                                                                                                                  
4700/10000 queries (25.0s, 188.4 qps)                                                                                                                                                                                                  
4800/10000 queries (25.5s, 188.3 qps)                                                                                                                                                                                                  
4900/10000 queries (26.0s, 188.4 qps)                                                                                                                                                                                                  
5000/10000 queries (26.5s, 188.5 qps)                                                                                                                                                                                                  
5100/10000 queries (27.1s, 188.3 qps)                                                                                                                                                                                                  
5200/10000 queries (27.6s, 188.2 qps)                                                                                                                                                                                                  
5300/10000 queries (28.2s, 188.1 qps)                                                                                                                                                                                                  
5400/10000 queries (28.7s, 188.2 qps)                                                                                                                                                                                                  
5500/10000 queries (29.3s, 187.6 qps)                                                                                                                                                                                                  
5600/10000 queries (29.9s, 187.6 qps)                                                                                                                                                                                                  
5700/10000 queries (30.3s, 188.0 qps)                                                                                                                                                                                                  
5800/10000 queries (30.9s, 187.9 qps)                                                                                                                                                                                                  
5900/10000 queries (31.4s, 188.1 qps)                                                                                                                                                                                                  
6000/10000 queries (31.9s, 188.1 qps)                                                                                                                                                                                                  
6100/10000 queries (32.4s, 188.2 qps)                                                                                                                                                                                                  
6200/10000 queries (32.9s, 188.3 qps)                                                                                                                                                                                                  
6300/10000 queries (33.4s, 188.6 qps)                                                                                                                                                                                                  
6400/10000 queries (33.9s, 188.7 qps)                                                                                                                                                                                                  
6500/10000 queries (34.4s, 188.7 qps)                                                                                                                                                                                                  
6600/10000 queries (35.0s, 188.7 qps)                                                                                                                                                                                                  
6700/10000 queries (35.5s, 188.7 qps)                                                                                                                                                                                                  
6800/10000 queries (36.0s, 188.8 qps)                                                                                                                                                                                                  
6900/10000 queries (36.5s, 189.1 qps)                                                                                                                                                                                                  
7000/10000 queries (37.0s, 189.4 qps)                                                                                                                                                                                                  
7100/10000 queries (37.5s, 189.2 qps)                                                                                                                                                                                                  
7200/10000 queries (38.0s, 189.2 qps)                                                                                                                                                                                                  
7300/10000 queries (38.6s, 189.3 qps)                                                                                                                                                                                                  
7400/10000 queries (39.1s, 189.1 qps)                                                                                                                                                                                                  
7500/10000 queries (39.7s, 189.0 qps)                                                                                                                                                                                                  
7600/10000 queries (40.2s, 189.0 qps)                                                                                                                                                                                                  
7700/10000 queries (40.7s, 189.0 qps)                                                                                                                                                                                                  
7800/10000 queries (41.3s, 189.0 qps)                                                                                                                                                                                                  
7900/10000 queries (41.8s, 189.0 qps)                                                                                                                                                                                                  
8000/10000 queries (42.2s, 189.7 qps)                                                                                                                                                                                                  
8100/10000 queries (42.7s, 189.9 qps)                                                                                                                                                                                                  
8200/10000 queries (43.2s, 189.9 qps)                                                                                                                                                                                                  
8300/10000 queries (43.7s, 189.9 qps)                                                                                                                                                                                                  
8400/10000 queries (44.2s, 190.0 qps)                                                                                                                                                                                                  
8500/10000 queries (44.7s, 190.0 qps)                                                                                                                                                                                                  
8600/10000 queries (45.2s, 190.1 qps)                                                                                                                                                                                                  
8700/10000 queries (45.7s, 190.2 qps)                                                                                                                                                                                                  
8800/10000 queries (46.3s, 190.1 qps)                                                                                                                                                                                                  
8900/10000 queries (46.8s, 190.1 qps)                                                                                                                                                                                                  
9000/10000 queries (47.3s, 190.1 qps)                                                                                                                                                                                                  
9100/10000 queries (47.9s, 190.0 qps)                                                                                                                                                                                                  
9200/10000 queries (48.4s, 189.9 qps)                                                                                                                                                                                                  
9300/10000 queries (49.0s, 190.0 qps)                                                                                                                                                                                                  
9400/10000 queries (49.5s, 189.9 qps)                                                                                                                                                                                                  
9500/10000 queries (50.0s, 189.9 qps)                                                                                                                                                                                                  
9600/10000 queries (50.5s, 190.1 qps)                                                                                                                                                                                                  
9700/10000 queries (51.1s, 189.8 qps)                                                                                                                                                                                                  
9800/10000 queries (51.8s, 189.2 qps)                                                                                                                                                                                                  
9900/10000 queries (52.4s, 188.9 qps)                                                                                                                                                                                                  
10000/10000 queries (53.0s, 188.8 qps)                                                                                                                                                                                                 
Completed in 53.0s, avg nodes visited: 4243.9                                                                                                                                                                                          
Recall@10 (Faiss entry point): 0.9857

Faiss native search (sanity check)...                                                                                                                                                                                                    
Recall@10 (Faiss native): 0.9857

Hub pool: 256 nodes (degree range: 32 down to 32)                                                                                                                                                                                        
Simulating 10000 FLAT searches (efSearch=256, 1 hub entry points)...                                                                                                                                                                     
500/10000 queries (1.8s, 272.7 qps)                                                                                                                                                                                                    
1000/10000 queries (3.6s, 279.3 qps)                                                                                                                                                                                                   
1500/10000 queries (5.5s, 271.6 qps)                                                                                                                                                                                                   
2000/10000 queries (7.4s, 271.7 qps)                                                                                                                                                                                                   
2500/10000 queries (9.2s, 272.3 qps)                                                                                                                                                                                                   
3000/10000 queries (10.8s, 277.0 qps)                                                                                                                                                                                                  
3500/10000 queries (12.7s, 276.0 qps)                                                                                                                                                                                                  
4000/10000 queries (14.5s, 275.0 qps)                                                                                                                                                                                                  
4500/10000 queries (16.2s, 277.3 qps)                                                                                                                                                                                                  
5000/10000 queries (18.0s, 277.6 qps)                                                                                                                                                                                                  
5500/10000 queries (19.8s, 277.2 qps)                                                                                                                                                                                                  
6000/10000 queries (21.6s, 277.9 qps)                                                                                                                                                                                                  
6500/10000 queries (23.3s, 278.7 qps)                                                                                                                                                                                                  
7000/10000 queries (25.0s, 280.0 qps)                                                                                                                                                                                                  
7500/10000 queries (27.0s, 277.4 qps)                                                                                                                                                                                                  
8000/10000 queries (28.7s, 278.3 qps)                                                                                                                                                                                                  
8500/10000 queries (30.5s, 278.2 qps)                                                                                                                                                                                                  
9000/10000 queries (32.3s, 278.5 qps)                                                                                                                                                                                                  
9500/10000 queries (34.2s, 277.6 qps)                                                                                                                                                                                                  
10000/10000 queries (36.0s, 277.8 qps)                                                                                                                                                                                                 
36.0s, avg visited: 4183.6, Recall@10: 0.9859
                                                                                                                                                                                                                                         
================================================================================                                                                                                                                                         
PAGES TOUCHED: Faiss entry point search (baseline)
================================================================================                                                                                                                                                         
Layout                          Avg Pages   Median      P95      P99
--------------------------------------------------------------------------------                                                                                                                                                         
Insertion Order                   343.0              343.0    343.0    343.0                                                                                                                                                             
BFS Order                         329.0 (+4.1%)      335.0    342.0    343.0                                                                                                                                                             
Reverse Cuthill-McKee             332.6 (+3.0%)      336.0    343.0    343.0                                                                                                                                                             
Greedy (strict pages)             413.9 (-20.7%)     417.0    456.0    468.0
                                                                                                                                                                                                                                         
================================================================================                                                                                                                                                         
PAGES TOUCHED: Flat search (1 entry points)
================================================================================                                                                                                                                                         
Layout                          Avg Pages   Median      P95      P99
--------------------------------------------------------------------------------                                                                                                                                                         
Insertion Order                   343.0              343.0    343.0    343.0                                                                                                                                                             
BFS Order                         329.0 (+4.1%)      335.0    342.0    343.0                                                                                                                                                             
Reverse Cuthill-McKee             331.7 (+3.3%)      336.0    343.0    343.0                                                                                                                                                             
Greedy (strict pages)             409.5 (-19.4%)     414.0    453.0    465.0
                                                                                                                                                                                                                                         
================================================================================                                                                                                                                                         
INTRA-PAGE EDGE RATIO (edges staying within same page)
================================================================================                                                                                                                                                         
Insertion Order                0.0030 (0.3%)                                                                                                                                                                                           
BFS Order                      0.0171 (1.7%)                                                                                                                                                                                           
Reverse Cuthill-McKee          0.0173 (1.7%)                                                                                                                                                                                           
Greedy (strict pages)          0.2163 (21.6%)

Done.                                          

---

# Production Validation on a Live OpenSearch Cluster (2026-09-01)

Everything above is the Python simulation. This section records the first end-to-end
validation on a real OpenSearch node (k-NN plugin, Faiss SQ 1-bit `on_disk` 32x), measured
with the `PageTouchTracker` read-amplification instrumentation.

**Setup:** 1M vectors (`documents-1m.hdf5`), single force-merged segment (so per-query ==
per-segment over one 1M-node graph), `--shuffle-ingest` (decorrelates doc-id from vector
space — see "Dataset is pre-sorted" below), `ef_search=256`, `k=100`, 20 queries, `rescore=false`.
Ordering = **dense greedy** (`buildOrderingOfVectorsUsingIndexStructure`, now dense-packed).

## Read Amplification (bytes faulted in / bytes actually used)

`readAmp = (distinctPages * pageSize) / (distinctVectors * recordSize)`, recordSize = 112 B.
Both baseline and reordered stores are 112 B/record (after the prefetch-stride fix below), so
this is a clean apples-to-apples comparison — no normalization.

| readAmplification | 32 KB page | 8 KB page |
|---|---|---|
| **baseline** (reordering disabled) | 137.2× | 59.7× |
| **greedy** (reordering enabled)    | **82.5×** | **32.4×** |

Supporting metrics (per query, avg):

| | pages/query | vecs/page | page util |
|---|---|---|---|
| baseline 32 KB | 2797 | 2.13 | 0.73% |
| greedy 32 KB   | 1666 | 3.55 | 1.21% |
| baseline 8 KB  | 4867 | 1.23 | 1.68% |
| greedy 8 KB    | 2616 | 2.26 | 3.09% |

Recall@100 flat (baseline 0.6365 vs reordered 0.6335; reordering does not change values).

### Two independent, compounding levers
- **Reordering (greedy vs baseline):** −40% read-amp at 32 KB, −46% at 8 KB. vecs/page +67–84%.
- **Page size (32 KB → 8 KB):** ~2.3× lower for baseline, ~2.5× for greedy (less waste per fault).
- **Combined (baseline+32 KB → greedy+8 KB): 137.2× → 32.4× = 4.2× less I/O.**
- Page size is the *bigger* single lever here (~2.4×) than reordering (~1.4×); reordering helps
  slightly more at 8 KB (finer pages convert clustering to savings with less rounding waste).

### Why absolute amplification stays high (32×+)
At `ef_search=256` the visited set is ~5,900 vectors, almost all distinct (little neighbor-list
reuse). Perfect clustering would need only 5964/292 ≈ **20** pages at 32 KB, but best-first search
is a *broad sweep across many clusters*, not one tight neighborhood — so ~1,666 pages is the
realistic floor, not 20. No layout fixes this; only visiting fewer nodes (lower ef) does. The sim's
"greedy advantage grows at lower efSearch" implies the reordering win widens as ef drops.

### Sanity check: shuffled baseline ≈ random placement
For 5,964 vectors uniformly scattered over 3,425 32 KB pages, expected distinct pages ≈ 2,822;
baseline measured **2,797**. So the shuffled baseline is effectively random placement (as intended),
and greedy concentrates it to 1,666.

## Dataset is pre-sorted — why `--shuffle-ingest` is the *realistic* model, not a hack

`documents-1m.hdf5` is atypically pre-sorted by similarity, which makes its *natural* ingest order a
near-best-case (already-reordered) baseline. Data-only proof (no reordering code involved):

- **True nearest-neighbor id gap** `|id − nn_id|`: **median 13** (p25=2, p75=67).
- **81%** of vectors have their true NN within one 32 KB page's worth of ids (≤292).
- A realistic random-arrival ingest would show ≈292/N = **0.3%** within a page, median gap ~N/2.
  → this dataset is pre-sorted ~270× more than random.

In real OpenSearch, post-merge vectors are laid out in doc-id (ingest) order, and for most workloads
arrival order is uncorrelated with embedding space → physically scrambled. `--shuffle-ingest`
reproduces that; it removes the dataset's unrealistic pre-sort rather than breaking a realistic
baseline. Conclusion: on realistically-ordered data, greedy reordering delivers the numbers above;
on pre-sorted ingest the baseline is already near-optimal and reordering adds little (but greedy does
not hurt, unlike BFS — see below).

## Implementation notes (this branch)

- **Ordering switched BFS → dense greedy.** `buildOrderingOfVectorsUsingIndexStructure` now packs
  affine clusters back-to-back into a dense `0..N-1` map (no strict-page padding — the strict variant
  wasted ~90% and is deferred). The build strategy calls it with
  `pageCapacity = 32768 / (quantizedVecBytes + 16)` = 292. Confirmed applied in production via the
  reader diagnostic (`displaced=100%`, and physical slot of the 2nd cluster seed = 292 = pageCapacity).
- **Baseline prefetch-stride bug fixed.** `PrefetchableVectorValuesHelper.doPrefetch` used
  `getVectorByteLength()` (96 B, code only) as the prefetch stride, but SQ records are 112 B
  (code + corrections). It now derives the stride from `slice.length()/size()` → 112. This fixed both
  a real baseline prefetch bug (was fetching misaligned `ord*96` ranges) and the page-touch metric
  (baseline was previously measured at 96 and under-counted).
- **Instrumentation:** `PageTouchTracker` (in `codec.scorer`) reports read amplification at 32 KB and
  8 KB per query. Enable at runtime with no restart:
  `PUT _cluster/settings {"persistent":{"logger.org.opensearch.knn.index.codec.scorer.PageTouchTracker":"DEBUG"}}`
  (or `scripts/recall_reordering.py --page-touch --log-file <node log>`, which toggles it and prints
  a per-index summary).

## BFS vs greedy (production) — open

BFS was measured earlier at a *different* ef and *before* the prefetch fix, so a clean production
head-to-head at identical ef=256 is still outstanding. The sim (identical conditions) puts BFS ~41%
vs dense greedy ~52–56% page reduction at 1M/ef=256; production greedy came in at ~40% page reduction
(82.5× vs 137.2× read-amp) — closer to the sim's BFS than its greedy, so it is not yet confirmed that
greedy beats BFS in production at this ef. A build-time BFS/greedy toggle would be needed for the
clean A/B (not implemented — deliberately deferred).
