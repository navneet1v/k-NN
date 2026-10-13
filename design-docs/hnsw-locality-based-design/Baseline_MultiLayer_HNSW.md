# Baseline — multi-layer HNSW recall (1M × 768)

Reference recall for the locality-reordering simulations: a standard Faiss HNSW with all its layers and the
normal top-down entry point. Script: `scripts/baseline_hnsw.py`. Run date: 2026-10-05.

## Setup

| Item | Value |
|---|---|
| Dataset | `documents-1m.hdf5` — 1,000,000 × 768-d, 10,000 test queries with exact ground truth |
| Index | Faiss `IndexHNSWFlat`, inner product, full-precision vectors, wrapped in `IndexIDMap` (returns dataset ids) |
| HNSW | M = 16 (up to 32 neighbours at layer 0), ef_construction = 256, default level assignment (multi-layer) |
| Insertion order | shuffled with seed 42 — the same order `locality_compare.py` uses for its single-layer graph |
| Search | all 10,000 queries, ef_search = 256 and 100, k = 100 |
| Threads | 4 (build and search) |
| Saved index | `scripts/documents-1m.hdf5.baseline.n1000000.M16.efc256.shuf.faiss` (3.2 GB); reloaded on later runs, `--rebuild` forces a new build |

Command:
```bash
cd scripts
python baseline_hnsw.py --dataset documents-1m.hdf5            # ef_search 256, k 100 (defaults)
python baseline_hnsw.py --dataset documents-1m.hdf5 --ef-search 100   # reuses the saved index
```

## Results

Both rows use the same saved index; only ef_search changes.

| Metric | ef_search = 256 | ef_search = 100 |
|---|---|---|
| recall@1 | 0.9861 | 0.9550 |
| recall@10 | 0.9723 | 0.9207 |
| **recall@100** | **0.9116** | **0.7920** |
| Vectors scored per query (Faiss `hnsw_stats.ndis`) | ≈ 5,072 | ≈ 2,339 |
| Search time, 10,000 queries (4 threads, laptop) | 3.4 s (0.34 ms/query) | 1.5 s (0.15 ms/query) |

Index facts (same for both rows):

| | Value |
|---|---|
| Levels built | 6 (max_level = 5) |
| Nodes whose top level is L0 / L1 / L2 / L3 / L4 / L5 | 937,597 / 58,500 / 3,693 / 194 / 15 / 1 |
| Build time | 840 s (4 threads, laptop) |

The first ef_search = 256 search right after the build measured 211.9 s (21 ms/query) because the machine was
still busy; the clean rerun on the saved index is the 3.4 s above. Recall and vectors scored were identical in
both. Timings are from a laptop and are only meant for comparing the two ef_search values.

## Comparison with the single-layer graph used in the simulations

| | Multi-layer HNSW (this baseline) | Single-layer graph (`locality_compare.py`) |
|---|---|---|
| Graph | all levels, standard top-down entry | layer 0 only, entry = highest-degree node |
| recall@100, ef_search 256 (10,000 queries) | 0.9116 | 0.9116 |
| Vectors scored per query, ef_search 256 | ≈ 5,072 | 5,070 |
| recall@100, ef_search 100 (10,000 queries) | 0.7920 | 0.7915 |
| Vectors scored per query, ef_search 100 | ≈ 2,339 | 2,340 |

At 1M vectors the upper layers make no measurable difference at either ef_search: recall@100 and the work per query are the same as
the single-layer graph. This supports building the reordered index as a single layer, and it confirms the
simulation's numbers against Faiss's own multi-layer search.

An earlier one-off check (`scripts/multilayer_recall.py`, natural insertion order, not saved) gave recall@100
0.9110, recall@10 0.9722, recall@1 0.9854 — consistent with this run.
