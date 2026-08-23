# RFC: Locality-Aware Physical Layout for Disk-Resident HNSW Graphs

> **Status:** Draft v0.1  
> **Author:** Research Notes  
> **Audience:** Search Infrastructure / Lucene / ANN Engineers

---

# Executive Summary

This RFC proposes a locality-aware physical storage layout for disk-resident HNSW graphs to reduce SSD read amplification during approximate nearest neighbor search.

Rather than storing vectors in insertion order, vectors are organized into fixed-size disk pages that correspond to graph neighborhoods. The proposal combines:

- Structural analysis of the HNSW graph
- Greedy page-growing
- Limited boundary replication inspired by SOAR
- Optional hub-aware seeding

The objective is to minimize page faults without changing HNSW search semantics.

> **NOTE**
>
> This document captures the current state of the research discussions. Several algorithms are hypotheses that require empirical validation.

---

# Table of Contents

1. Background
2. Problem Statement
3. Goals
4. Non-Goals
5. Existing Systems
6. Design Overview
7. Structural Signals
8. Greedy Page Growing
9. Hub-based Seeding
10. Boundary Replication
11. Disk Serialization
12. Lucene Integration
13. Cost Model
14. Evaluation Plan
15. Open Questions
16. Research Roadmap

---

# 1. Background

## HNSW

HNSW provides logarithmic search by navigating a hierarchical proximity graph.

Upper layers perform routing.

Layer 0 contains nearly all vectors.

Current Lucene stores vectors sequentially in insertion order.

Graph locality and physical locality are unrelated.

---

# 2. Problem Statement

Assume:

- Page = 32 KB
- Vector = 128 bytes
- ~250 vectors/page

Traversal:

Node1 → Node81234 → Node9 → Node7000

Every hop may require another SSD read.

The result is high read amplification.

Optimization target:

Store vectors that are traversed together in the same physical page.

---

# 3. Goals

- Reduce page reads
- Preserve recall
- Preserve HNSW semantics
- Integrate naturally with Lucene
- Require no runtime profiling

# 4. Non-Goals

- New ANN algorithm
- Graph rewiring
- Compression improvements

---

# 5. Existing Systems

## DiskANN

Separates routing from disk-resident vectors.

## SOAR

Uses selective replication to reduce boundary crossings.

## Graph Partitioning

Produces balanced cuts but is expensive and not naturally incremental.

---

# 6. Proposed Architecture

HNSW Graph

↓

Structural Analysis

↓

Greedy Page Growing

↓

Boundary Replication

↓

Disk Serialization

↓

Search

---

# 7. Structural Signals

Potential signals include:

- HNSW level
- In-degree
- Neighbor diversity
- Local clustering coefficient
- Search frequency (future)

Example hub score:

hub_score =
α·level
+ β·indegree
+ γ·diversity

---

# 8. Greedy Page Growing

Algorithm

Repeat:

1. Choose seed
2. Create page
3. Add highest affinity neighbor
4. Continue until page full
5. Repeat

Affinity may combine:

- edge count
- shared neighbors
- estimated search probability
- boundary penalty

Pseudo-code

```
while unassigned:
    seed = choose_seed()
    page = {seed}

    while page.not_full():
        candidate = best_neighbor(page)
        page.add(candidate)
```

Complexity depends on candidate expansion strategy.

---

# 9. Hub-Based Seeding

Instead of random seeds:

Choose highest hub score.

Grow surrounding territory.

Potential issue:

Never create "hub-only" pages.

Instead:

Hub defines page.

Not page contents.

---

# 10. Boundary Replication

After assignment:

Find nodes with many external edges.

Replicate only small percentage.

Possible metrics:

external_edges / total_edges

or approximate betweenness.

Inspired by SOAR.

---

# 11. Disk Serialization

Each page contains:

- vectors
- graph edges
- replicated nodes
- metadata

Future work:

Compressed page headers.

---

# 12. Lucene Integration

Potential indexing pipeline

Build HNSW

↓

Compute structural metrics

↓

Assign pages

↓

Replicate boundaries

↓

Serialize

Future possibility:

Perform page assignment during construction.

---

# 13. Cost Model

Let

P(q)

be pages read for query q.

Objective:

minimize

Σ P(q)

subject to:

- fixed page size
- bounded replication
- unchanged graph

---

# 14. Evaluation

Datasets

- SIFT
- DEEP1B
- LAION subsets

Metrics

- Recall@10
- Recall@100
- SSD page reads
- Latency
- Build time
- Memory overhead

Baselines

- Lucene
- Graph partitioning
- Random layout
- BFS ordering

---

# 15. Open Questions

- Can Lucene beam search traces improve affinity?
- Is HNSW level sufficient for hub detection?
- Should replication happen online?
- What replication budget is optimal?
- Should upper layers remain memory resident?

---

# 16. Research Roadmap

Phase 1

- Cost model
- Simulator

Phase 2

- Greedy implementation

Phase 3

- Lucene prototype

Phase 4

- Evaluation

Phase 5

- Publication

---

# Appendix A

Ideas explored

- BFS ordering
- DFS ordering
- Spectral ordering
- Graph partitioning
- Greedy page growing
- Hub seeding
- SOAR-inspired replication

Current hypothesis:

Greedy page growing with limited boundary replication appears to offer the best tradeoff between implementation complexity and expected reduction in disk read amplification.

---

# Future Expansion

This RFC intentionally serves as Version 0.1.

Planned Version 1.0 additions include:

- Formal optimization problem
- Mathematical notation
- Detailed pseudocode
- Complexity proofs
- Search trace analysis
- Lucene implementation details
- Worked examples
- Failure cases
- Sensitivity analysis
- Experimental methodology
- Comparison against DiskANN, Vamana, and SOAR
- Publication-quality figures
