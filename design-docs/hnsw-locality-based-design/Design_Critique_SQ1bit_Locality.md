# Design Critique: Locality-Aware Physical Layout for Faiss SQ 1-bit HNSW

> Brutal assessment of failure modes, compounding risks, and open questions that must be answered before implementation.

---

## 1. Flat Search Without Upper Layers Destroys Convergence Guarantees

The most dangerous decision. HNSW upper layers provide O(log N) routing — they get you to the right neighborhood fast. CAGRA compensates for flat search by using a **much higher-quality graph** (built with GPU-accelerated IVF-PQ + graph optimization passes that prune and add long-range edges).

We're taking a standard CPU-built HNSW graph (designed for multi-layer routing) and ripping off the layers. The Layer 0 graph was **never designed to be navigated flat** — it has short-range edges only. Random entry points on a pure Layer 0 graph means:

- You need many more hops to converge (higher latency, more pages touched — defeating the whole purpose)
- You need more entry points to compensate (CAGRA uses 64-128 random entries)
- Recall will drop unless you increase `efSearch` significantly, which means visiting more nodes, which means reading more pages

**The locality optimization might be eaten alive by the increased hop count from flat search.** You could have perfect page locality but still read 3x more pages because convergence takes 3x longer.

---

## 2. The Affinity Function Is Graph-Oblivious to Traversal Direction

"Shared neighbor count" measures structural similarity but NOT traversal probability. Consider:

- Node A has 10 shared neighbors with the page — but is in a dead-end cluster that search rarely enters from this direction
- Node B has 3 shared neighbors with the page — but is on the primary traversal path from the entry point

You'll pick A over B. The page will have great internal connectivity but search will still cross page boundaries on the hot path.

The real signal is: "which nodes are visited together in the same query?" That requires search trace data — which was explicitly excluded as a non-goal. Without it, you're optimizing for graph topology, not traversal topology. They correlate but aren't the same.

---

## 3. Hub Seeding Creates Pathological Page Boundaries

You seed pages from high-level nodes (hubs). But hubs are by definition nodes that connect different regions. Their neighborhoods span multiple clusters. When you grow a page from a hub:

- The hub's neighbors pull in vectors from multiple distinct clusters
- The page becomes a mixed bag of vectors from different regions
- When search arrives at the hub, the neighbors it wants next are on THIS page — good
- But when search arrives at a non-hub node on this page, the node's other neighbors are on different pages — bad

The irony: **hub-seeded pages optimize for the routing phase but not for the final convergence phase** where most page reads happen (the last 50-80% of scored candidates are in the local neighborhood, not at hubs).

---

## 4. Segment Merge Invalidates All Locality Work

When Lucene merges segments:
- Multiple segments are combined into one
- The graph is rebuilt from scratch
- All ordinals change
- The `.vep` file must be recomputed

This means:
- Every merge pays the full page-assignment cost again
- The locality optimization is ephemeral — it only lives until the next merge
- For write-heavy workloads, you're constantly paying build overhead for transient benefit
- The threshold of 10K docs is meaningless — a 10K segment will be merged into a larger segment soon anyway, wasting the locality computation

**The only segments where locality matters are large, stable segments that won't be merged again.** But those are exactly the segments where the page-assignment algorithm is most expensive (O(N * M * pageCapacity) at minimum).

---

## 5. The `.vep` File Doubles Storage I/O During Build

The `.veb` file is written first (Lucene requirement), then you read it all back and write `.vep` in permuted order. For a 10M vector segment at 112 bytes/vector:

- `.veb` write: 1.12 GB
- `.veb` read-back: 1.12 GB (random access pattern — reading by `inversePermutation` order)
- `.vep` write: 1.12 GB

That's **3.36 GB of I/O** just for the vector file, compared to 1.12 GB today. The read-back is the worst part — you're reading `.veb` in permuted order, which means random I/O across the entire file. You're hitting the exact locality problem during build that you're trying to solve during search.

---

## 6. The `.veb` File Is Dead Weight

You write `.veb` (Lucene requires it) then write `.vep` (for locality). At search time you use `.vep`. The `.veb` file sits there consuming disk space forever, never read. For the example above, that's 1.12 GB of wasted disk per segment.

You can't skip writing `.veb` because Lucene's writer produces it. You can't delete it because Lucene's segment tracking expects it. You're doubling the vector storage footprint for every locality-enabled segment.

---

## 7. Page-Growing Is NP-Hard's Greedy Cousin — It Gets Stuck

The greedy algorithm makes locally optimal choices. Classic failure modes:

**Fragmentation:** After assigning 80% of nodes, the remaining 20% are scattered fragments with no strong affinity to each other. They get dumped into pages with poor locality. These "leftover" pages have high external edge ratios — exactly the nodes that needed boundary replication (Phase 3). But Phase 3 is deferred.

**Ordering sensitivity:** The order in which hubs are processed determines which nodes get claimed first. Hub A and Hub B share a neighborhood — whoever gets processed first claims the shared nodes. The other hub's page is then forced to include less-related nodes. This is the classic greedy coloring problem — you can get arbitrarily bad results depending on ordering.

**No backtracking:** If the algorithm assigns node X to page P, but later discovers that X would have been much better on page Q (because Q's subsequently-added nodes are all neighbors of X), there's no way to fix this. The algorithm is single-pass.

---

## 8. Memory Pressure During Build

The page-assignment algorithm needs:
- Full Layer 0 adjacency lists in Java heap: N * M * 4 bytes (10M vectors, M=32 → 1.28 GB)
- Levels array: N * 4 bytes (40 MB)
- Permutation array: N * 4 bytes (40 MB)
- Assigned boolean array: N bytes (10 MB)
- Candidate priority queue state: variable but significant

For a 10M vector segment, you're looking at ~1.4 GB of Java heap just for the page assignment — ON TOP of the native memory holding the Faiss index. This happens during indexing, where memory is already under pressure.

The JNI `extractSQHnswGraphData` call copies the entire graph from native to Java. That's a full duplication of the neighbor array in Java heap.

---

## 9. The CAGRA Code Path Assumption Is Fragile

**Status: Fixable.** We can introduce a dedicated `FaissLocalityAwareHNSW` subclass with its own type marker rather than reusing `FaissCagraHNSW`. This is straightforward.

However, the random entry point problem remains regardless of how we type the index:

- The `RandomEntryPointsKnnSearchStrategy` picks random ordinals — but in the permuted layout, adjacent ordinals are in the same neighborhood. Random entry points will cluster in the same region instead of spreading across the vector space.

That last point is critical: **random entry points on a locality-reordered graph are NOT spatially diverse.** Ordinals 0-292 are all on page 0 — they're all in the same graph neighborhood. Picking 5 random ordinals out of [0, N) will likely land in 5 different pages = 5 different neighborhoods, but it's by chance. The CAGRA strategy assumes ordinals are in insertion order (roughly random). After permutation, ordinals are clustered — random selection becomes biased.

---

## 10. Recall Will Drop — And You Can't Easily Debug Why

Multiple factors compound:
- Flat search (fewer routing shortcuts)
- Random entry points (poor initial positioning)
- Same efSearch (tuned for multi-layer)

When recall drops by 5%, you won't know if it's because:
- The flat search needs more entry points
- The entry points are spatially clustered (point #9)
- The page-growing algorithm fragmented some neighborhoods
- The graph remap introduced subtle ordering bugs

You'll need to A/B test each factor independently — which is hard when they're all baked into a single new file format.

---

## 11. The Graph Remap Has a Subtle Correctness Risk

`remapSQHnswNeighbors` walks the neighbor array and replaces `n` with `permutation[n]`. But:

- The HNSW neighbor array stores neighbors for ALL levels (level 0, 1, 2, ...) interleaved via the offsets array
- Upper-level neighbor IDs must ALSO be remapped (they reference the same node ordinals)
- If you only remap level 0 entries but leave upper-level entries with old ordinals, the graph file is internally inconsistent
- The "null flat storage" sentinel means the file won't be used for upper-level search — but if anyone ever reads the full graph (debugging, validation), they'll get corrupted upper-level data

---

## 12. Compounding Risk: Death by a Thousand Approximations

The design's biggest risk isn't any single flaw — it's the **compounding of approximations**:

1. Greedy page assignment (approximate)
2. Flat search without proper long-range edges (approximate)
3. Random entry points on clustered ordinals (approximate)
4. Simple affinity function ignoring traversal direction (approximate)

Each one loses 5-15% versus the optimal. Compounded, you could easily see 30-40% more page reads than theoretical minimum, while paying 2x build time and 2x storage. At that point, you might get comparable results from just doing a simple BFS reordering (which is trivially implementable and has none of these failure modes).

---

## The Question to Answer Before Building Any of This

Run a simulation. Take a real HNSW graph, record which nodes are visited per query, measure pages touched under:
- Insertion order (baseline)
- BFS ordering from entry point
- Reverse Cuthill-McKee ordering
- Greedy page assignment

If greedy is only 20% better than BFS ordering, the complexity isn't worth it. BFS reordering is a one-liner and has zero risk.
