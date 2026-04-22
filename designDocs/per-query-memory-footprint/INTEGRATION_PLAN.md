# Per-Query Memory Footprint — Integration Plan

## 1. Problem

The KNN search path has zero per-query resource observability. There is no way to answer: "how much vector data did this query read from disk/RAM to produce results?"

### What exists today

| Metric | Location | Limitation |
|---|---|---|
| `KnnCollector.visitedCount` | Incremented per vector scored | Trapped inside the collector, never surfaced to the caller after `topDocs()` |
| `GRAPH_QUERY_REQUESTS` | `KNNCounter` | Global counter, not per-query |
| `graph_memory_usage` | `NativeMemoryCacheManager` | Cluster-level aggregate of cached graphs, not per-query |

### What's missing

- **Vectors scored** — How many vectors had their distance computed against the query vector.
- **Bytes read** — How many bytes of vector data were read from disk/mmap to compute those scores.
- **Graph nodes visited** (HNSW) — How many nodes in the HNSW graph were entered during traversal.
- **Centroids compared** (Cluster ANN) — How many cluster centroids were evaluated to select candidate clusters.
- **Member lists scanned** (Cluster ANN) — How many vectors within selected clusters were scanned.
- **Vectors rescored** — In 2-phase search (quantized first pass + full precision second pass), how many vectors went through the second pass.

## 2. Scope

Focused on the **memory-optimized search path** (`MemoryOptimizedKNNWeight` → Lucene reader). This covers:
- HNSW via `FaissMemoryOptimizedSearcher` → `HnswGraphSearcher` → `FaissHnswGraph`
- Cluster ANN via `ClusterANN1040KnnVectorsReader`
- Future engine-less algorithms

The native JNI path (`DefaultKNNWeight` → FAISS C++) is out of scope — FAISS returns only `(docId, score)[]` through the JNI boundary with no internal metrics exposed.

## 3. Audit: What's Available in the Memory-Optimized Path

### 3.1 Vectors scored — `KnnCollector.visitedCount()`

**Already computed, just discarded.**

Lucene's `HnswGraphSearcher.searchLevel()` calls `knnCollector.incVisitedCount(1)` for every node it scores. After search, `knnCollector.visitedCount()` holds the total. But in `MemoryOptimizedKNNWeight.queryIndex()`, the collector goes out of scope after `topDocs()` is called — the count is lost.

**Location:** `MemoryOptimizedKNNWeight.queryIndex()` line ~207, between `search()` and `topDocs()`.

### 3.2 Edges traversed — `FaissHnswGraph.nextNeighbor()`

**We own `FaissHnswGraph`. Can instrument directly.**

`HnswGraphSearcher` calls these methods on `FaissHnswGraph` during traversal:

| Method | What it does | What it tells us |
|---|---|---|
| `seek(level, nodeId)` | Loads a node's neighbor list from disk via `indexInput.readInt()` | A node's edges were read. Count = **nodes whose neighbors were loaded**. |
| `nextNeighbor()` | Returns the next neighbor ID from the loaded list | An edge was traversed. Count (excluding `NO_MORE_DOCS`) = **total edges traversed**. |
| `neighborCount()` | Returns how many neighbors were loaded for current node | Edges loaded per seek. |

**Bytes read per seek:** `numNeighbors × 4` bytes (each neighbor ID is an `int`).

Instrumentation: add `long` counters in `FaissHnswGraph` — increment in `seek()` and `nextNeighbor()`. Zero overhead (primitive increment on hot path).

### 3.3 Vector bytes prefetched — `PrefetchHelper` + `PrefetchableFlatVectorScorer`

**We own the prefetch layer. Can instrument directly.**

During HNSW traversal, `HnswGraphSearcher` calls `bulkScore(nodes[], scores[], numNodes)` on `PrefetchableRandomVectorScorer`. Before scoring, it:

1. Takes the batch of candidate node ordinals
2. Sorts by offset, groups within 128KB windows
3. Calls `indexInput.prefetch(offset, length)` per group — issues `madvise(MADV_WILLNEED)` to asynchronously page in those bytes

**Overlap with `visitedCount`:** Every vector that goes through `bulkScore` is both prefetched and scored — they're the same vectors. But prefetch bytes ≠ scored vector bytes because:
- **128KB grouping** — If vectors A and C are requested but B sits between them within the window, B's bytes are prefetched too but never scored.
- **Single-node `score(node)` path** — Entry point scoring and individual callbacks are scored but NOT prefetched.

**Therefore:**
- `visitedCount` = algorithmic work (vectors whose distance was computed)
- Prefetch bytes = actual memory/IO pressure (includes grouping padding)
- Don't compute `vector_bytes_read` from `visitedCount` separately — prefetch bytes already cover that more accurately
- Don't sum them — that would double-count

**The right model:**
```
memory_pressure = prefetch_bytes + neighbor_bytes_read
useful_work     = vectors_scored (= visitedCount)
efficiency      = (vectors_scored × vectorByteSize) / prefetch_bytes
```

Instrumentation: accumulate `totalPrefetchedBytes` and `prefetchGroupCount` in `PrefetchHelper.prefetch()`. Zero overhead (two `long` increments per group).

### 3.4 Summary

| Metric | Source | Category | Description |
|---|---|---|---|
| `vectors_scored` | `knnCollector.visitedCount()` | Traversal | Number of vectors whose distance to the query was computed |
| `edges_traversed` | `FaissHnswGraph.nextNeighbor()` counter | Traversal | Individual neighbor IDs read from loaded lists. Sum of neighbor list sizes across all seeks |
| `neighbor_seeks` | `FaissHnswGraph.seek()` counter | Traversal | Times a node's full neighbor list was loaded from disk. Each seek loads up to M neighbors |
| `vector_bytes_prefetched` | `PrefetchHelper.prefetch()` accumulated lengths | Data Read | Bytes requested via madvise(MADV_WILLNEED). Includes 128KB grouping padding |
| `vector_bytes_read` | `visitedCount × dimension × bytesPerElement` | Data Read | Actual vector data bytes touched for distance computation |
| `neighbor_bytes_read` | `sum(numNeighbors × 4)` per seek | Data Read | Bytes read for HNSW neighbor lists |
| `total_bytes_read` | `vector_bytes_read + neighbor_bytes_read` | Data Read | Total data touched by the query |
| `prefetch_groups` | `PrefetchHelper.prefetch()` group counter | Data Read | Number of madvise I/O groups issued (one per 128KB window) |
| `early_terminated` | `knnCollector.earlyTerminated()` | Search behavior | Whether search hit the visit limit |
| `results_returned` | `topDocs.scoreDocs.length` | Output | Segment-level results before top-k merge |

**`neighbor_seeks` vs `edges_traversed`:** A seek loads one node's entire neighbor list. Edges are the individual links within those lists. With M=16 and 10 seeks, expect ~160 edges. The ratio `edges_traversed / neighbor_seeks` approximates the average neighbor list size.

### 3.5 Cluster ANN

For `ClusterANN1040KnnVectorsReader.search()`, we control the entire loop:
- Vectors scored = loop iteration count (already tracked via `incVisitedCount`)
- No graph structure (brute-force), so no edges/nodes metrics
- Future cluster index will have centroids compared + member lists scanned

## 4. Design Considerations

### 4.1 Two-layer metrics approach

Per-query metrics should be captured at two layers:

**Layer 1: Logical reads (what the algorithm asks for)**

Counters in our own code — always on, zero overhead (primitive `long` increments).

| Metric | Source | Overhead |
|---|---|---|
| Vectors scored | `knnCollector.visitedCount()` | Zero — already computed |
| Edges traversed | `FaissHnswGraph.nextNeighbor()` counter | Zero — primitive increment |
| Neighbor seeks | `FaissHnswGraph.seek()` counter | Zero — primitive increment |
| Neighbor bytes read | `sum(numNeighbors × 4)` per seek | Zero — computed from existing field |
| Vector bytes read | `visitedCount × dimension × bytesPerElement` | Zero — computed |

**Layer 2: Physical reads (what the kernel actually does)**

The memory-optimized path uses mmap with `MADV_RANDOM` (via `DataAccessHint.RANDOM` → `NativeAccess.madvise()`). This disables kernel read-ahead, so the OS only faults in the exact pages touched — no speculative prefetching. But the minimum read unit is still a **page** (4KB on Linux, 16KB on Apple Silicon), not the exact bytes requested.

To measure actual physical I/O:

| Mechanism | What it gives | Overhead | Platform |
|---|---|---|---|
| `getrusage()` before/after search | Major faults (disk reads) + minor faults (page cache hits) | ~2 syscalls per segment | Linux + macOS |
| `/proc/self/io` snapshot before/after | `read_bytes` (physical I/O) + `rchar` (VFS reads including cache) | ~2 file reads per segment | Linux only |
| `perf_event_open` for `PERF_COUNT_SW_PAGE_FAULTS` | Precise fault count via hardware counter | Near-zero | Linux only |
| `mincore()` on mmap region | Which pages are resident before/after | Expensive for large files | Linux + macOS |

**Recommended:** `getrusage()` — low overhead, cross-platform, gives both major faults (cold reads from disk) and minor faults (warm reads from page cache). Only enable when a "detailed metrics" flag is set.

**Example per-query output:**
```
logical_vectors_scored: 150
logical_vector_bytes: 76800          (150 × 128dim × 4bytes)
logical_edges_traversed: 2400
logical_neighbor_bytes: 9600         (2400 × 4bytes)
physical_major_faults: 3             (3 pages read from disk)
physical_minor_faults: 45            (45 pages served from cache)
physical_io_bytes: 49152             (3 × 16KB pages on Apple Silicon)
```

### 4.2 Capture mechanism for logical reads

The `FaissHnswGraph` instance is created per-search in `FaissMemoryOptimizedSearcher.doSearch()`. Counters on the graph instance are naturally per-search, per-segment. The `knnCollector` is also per-segment.

**Proposed:** Add counters directly on `FaissHnswGraph`. After `HnswGraphSearcher.search()` returns, read them alongside `knnCollector.visitedCount()` in `MemoryOptimizedKNNWeight.queryIndex()`.

### 4.3 Per-segment to per-query aggregation

`MemoryOptimizedKNNWeight.doANNSearch()` is called once per segment. `KNNWeight.searchLeaf()` collects per-segment results. Aggregation should happen at the `KNNWeight` level across all `searchLeaf()` calls.

### 4.4 How to pass graph metrics up

`FaissHnswGraph` is deep in the call chain: `MemoryOptimizedKNNWeight` → `reader.getVectorReader().search()` → `FaissMemoryOptimizedSearcher.doSearch()` → `HnswGraphSearcher.search(scorer, collector, graph, ...)`.

Options:
- **Thread-local:** Set before search, read after. No API changes. But fragile with concurrent segment searches.
- **On the collector:** Extend or wrap `KnnCollector` to carry a metrics object. The collector is already passed through the entire chain.
- **Return from reader:** Change `KnnVectorsReader.search()` to return metrics. But this is a Lucene interface we don't own.

### 4.5 MADV_RANDOM and page granularity

Files opened with `DataAccessHint.RANDOM` get `madvise(MADV_RANDOM)` applied to their mmap regions (via Lucene's `MemorySegmentIndexInputProvider.map()`). This means:
- No kernel read-ahead — only the touched page is faulted in
- Minimum read unit is still a page (4KB Linux / 16KB macOS ARM)
- A single `indexInput.readInt()` (4 bytes) faults in an entire page
- Contiguous neighbor IDs within the same page cost only 1 fault
- Neighbor lists from distant nodes cost 1 fault each

This is why Layer 2 (physical reads) matters — logical bytes read can be much smaller than physical bytes faulted.

## 5. Instrumentation Types

Two distinct instrumentation types, each answering a different question:

### Type A: Data Read Instrumentation

**Question:** How much data was read from disk/memory to serve this query?

| Metric | Source | What it measures |
|---|---|---|
| Vector bytes prefetched | `PrefetchHelper.prefetch()` accumulated lengths | Bytes the kernel was asked to page in for vector data (includes 128KB grouping padding) |
| Neighbor bytes read | `sum(numNeighbors × 4)` per `FaissHnswGraph.seek()` | Bytes read for HNSW neighbor lists |
| Prefetch group count | `PrefetchHelper.prefetch()` group counter | Number of I/O requests issued for vector data |
| Physical major faults | `getrusage()` delta | Pages actually read from disk (opt-in) |
| Physical minor faults | `getrusage()` delta | Pages served from page cache (opt-in) |

**Total memory pressure = vector_bytes_prefetched + neighbor_bytes_read**

### Type B: Search Traversal Instrumentation

**Question:** How much work did the HNSW algorithm do to find the k nearest neighbors?

| Metric | Source | What it measures |
|---|---|---|
| Vectors scored | `knnCollector.visitedCount()` | Nodes whose distance to query was computed |
| Neighbor seeks | `FaissHnswGraph.seek()` count | Nodes whose neighbor lists were loaded (graph expansion points) |
| Edges traversed | `FaissHnswGraph.nextNeighbor()` count | Individual neighbor links followed |
| Early terminated | `knnCollector.earlyTerminated()` | Whether the search stopped before exhausting the visit limit |
| Results returned | `topDocs.scoreDocs.length` | Final result count |

**Efficiency = vectors_scored / edges_traversed** (how many edges led to a useful scoring)

### Relationship between Type A and Type B

Type B (traversal) drives Type A (data read). Every vector scored requires reading its vector data. Every neighbor seek requires reading a neighbor list. But the mapping is not 1:1:
- Prefetch groups vectors within 128KB windows, so prefetch bytes > scored vector bytes
- A single seek loads all neighbors for a node, but not all neighbors get scored (only those not already visited)

Both types should be captured independently and reported together for a complete picture.

## 6. Tasks

### Type A: Data Read Instrumentation
- [ ] Add `totalPrefetchedBytes`, `prefetchGroupCount` counters to `PrefetchHelper`
- [ ] Add `neighborBytesRead` counter to `FaissHnswGraph` (accumulated from `numNeighbors × 4` per seek)
- [ ] Design `DataReadMetrics` data structure
- [ ] Implement per-segment capture in `MemoryOptimizedKNNWeight.doANNSearch()`
- [ ] Implement per-segment capture in `ClusterANN1040KnnVectorsReader.search()`
- [ ] Implement per-query aggregation across segments in `KNNWeight`
- [ ] (Opt-in) Implement `getrusage()` snapshot before/after search for physical fault counts

### Type B: Search Traversal Instrumentation
- [ ] Add `edgesTraversed`, `neighborSeeks` counters to `FaissHnswGraph`
- [ ] Read `knnCollector.visitedCount()` in `MemoryOptimizedKNNWeight.queryIndex()` before collector is discarded
- [ ] Design `TraversalMetrics` data structure
- [ ] Implement per-segment capture in `MemoryOptimizedKNNWeight.doANNSearch()`
- [ ] Implement per-segment capture in `ClusterANN1040KnnVectorsReader.search()`
- [ ] Implement per-query aggregation across segments in `KNNWeight`

### Common
- [ ] Determine how to pass `FaissHnswGraph` and `PrefetchHelper` counters back to `MemoryOptimizedKNNWeight` (thread-local vs collector wrapper)
- [ ] Expose metrics (stats API, slow log, or profile API — TBD)
- [ ] Add tests validating metric correctness for both types
