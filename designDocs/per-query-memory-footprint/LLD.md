# Per-Query Search Metrics — Low-Level Design

## 1. Overview

This document describes the design for capturing, aggregating, and exposing per-query search metrics across the memory-optimized KNN search path. Metrics are captured at query time, aggregated at shard, node, and cluster levels.

## 2. Data Model

### 2.1 Per-Query Metrics (captured per segment, aggregated per query)

```java
/**
 * Immutable snapshot of metrics for a single KNN search query.
 * Built by accumulating per-segment metrics during search.
 */
public class KNNSearchMetrics {

    // --- Type A: Data Read ---
    private final long vectorBytesPrefetched;    // bytes requested via PrefetchHelper
    private final long neighborBytesRead;        // bytes read for HNSW neighbor lists
    private final int prefetchGroupCount;        // number of prefetch I/O groups issued

    // --- Type B: Search Traversal ---
    private final long vectorsScored;            // vectors whose distance was computed
    private final long neighborSeeks;            // nodes whose neighbor lists were loaded
    private final long edgesTraversed;           // individual neighbor links followed
    private final boolean earlyTerminated;       // whether search hit visit limit
    private final int resultsReturned;           // final k results

    // --- Derived ---
    public long totalBytesRead() {
        return vectorBytesPrefetched + neighborBytesRead;
    }

    public float prefetchEfficiency(int dimension, int bytesPerElement) {
        if (vectorBytesPrefetched == 0) return 0;
        return (float)(vectorsScored * dimension * bytesPerElement) / vectorBytesPrefetched;
    }
}
```

### 2.2 Per-Segment Metrics (mutable, accumulated during search)

```java
/**
 * Mutable accumulator for a single segment's search.
 * One instance per doANNSearch() call. Not thread-safe — single segment = single thread.
 */
public class SegmentSearchMetrics {
    long vectorBytesPrefetched;
    long neighborBytesRead;
    int prefetchGroupCount;
    long vectorsScored;
    long neighborSeeks;
    long edgesTraversed;

    /** Merge another segment's metrics into this one. */
    void merge(SegmentSearchMetrics other) {
        this.vectorBytesPrefetched += other.vectorBytesPrefetched;
        this.neighborBytesRead += other.neighborBytesRead;
        this.prefetchGroupCount += other.prefetchGroupCount;
        this.vectorsScored += other.vectorsScored;
        this.neighborSeeks += other.neighborSeeks;
        this.edgesTraversed += other.edgesTraversed;
    }

    /** Freeze into immutable snapshot. */
    KNNSearchMetrics toSearchMetrics(boolean earlyTerminated, int resultsReturned) { ... }
}
```

## 3. Capture Points

### 3.1 Instrumentation in `FaissHnswGraph`

```java
public class FaissHnswGraph extends HnswGraph {
    // Existing fields...
    private long edgesTraversed;
    private long neighborSeeks;
    private long neighborBytesRead;

    @Override
    public void seek(int level, int internalVectorId) {
        // ... existing logic ...
        loadNeighborIdList(begin, end);
        neighborSeeks++;
        neighborBytesRead += (long) numNeighbors * Integer.BYTES;
    }

    @Override
    public int nextNeighbor() {
        if (nextNeighborIndex < numNeighbors) {
            edgesTraversed++;
            return neighborIdList[nextNeighborIndex++];
        }
        return NO_MORE_DOCS;
    }

    // Getters for reading after search
    public long getEdgesTraversed() { return edgesTraversed; }
    public long getNeighborSeeks() { return neighborSeeks; }
    public long getNeighborBytesRead() { return neighborBytesRead; }
}
```

`FaissHnswGraph` is created per-search in `FaissMemoryOptimizedSearcher.doSearch()`, so counters are naturally per-segment, per-search. No reset needed.

### 3.2 Instrumentation in `PrefetchHelper`

`PrefetchHelper` is a static utility. Per-query state needs a different approach.

**Option A: Thread-local accumulator**
```java
public class PrefetchHelper {
    private static final ThreadLocal<long[]> PREFETCH_METRICS = ThreadLocal.withInitial(() -> new long[2]);
    // [0] = totalBytes, [1] = groupCount

    public static void resetMetrics() {
        long[] m = PREFETCH_METRICS.get();
        m[0] = 0; m[1] = 0;
    }

    public static long getPrefetchedBytes() { return PREFETCH_METRICS.get()[0]; }
    public static int getPrefetchGroupCount() { return (int) PREFETCH_METRICS.get()[1]; }

    private static void prefetchExactVectorSize(...) {
        // ... existing logic ...
        // At each indexInput.prefetch(groupStartOffset, length):
        PREFETCH_METRICS.get()[0] += length;
        PREFETCH_METRICS.get()[1]++;
    }
}
```

**Option B: Pass accumulator through scorer** — `PrefetchableRandomVectorScorer` already wraps the delegate. It could hold a metrics reference. But this requires changing the scorer constructor chain.

**Recommendation:** Option A (thread-local). Search within a segment is single-threaded. Reset before `doANNSearch`, read after. Simple, no API changes.

### 3.3 Reading `visitedCount` from `KnnCollector`

In `MemoryOptimizedKNNWeight.queryIndex()`, read before the collector is discarded:

```java
// After search, before topDocs()
long vectorsScored = knnCollector.visitedCount();
boolean earlyTerminated = knnCollector.earlyTerminated();
TopDocs topDocs = knnCollector.topDocs();
```

### 3.4 Assembly in `MemoryOptimizedKNNWeight.doANNSearch()`

This is where all per-segment metrics come together. The challenge is getting `FaissHnswGraph` counters — the graph is deep in the call chain.

**Flow:** `doANNSearch()` → `queryIndex()` → `reader.getVectorReader().search()` → `FaissMemoryOptimizedSearcher.doSearch()` → `HnswGraphSearcher.search(scorer, collector, graph)`

The graph is created inside `FaissMemoryOptimizedSearcher.doSearch()` and not returned. Two options:

**Option A: Thread-local for graph metrics too**
```java
// In FaissMemoryOptimizedSearcher.doSearch():
FaissHnswGraph graph = new FaissHnswGraph(hnsw, indexInput.clone());
HnswGraphSearcher.search(scorer, collector, graph, acceptedOrds);
// Store on thread-local after search
SearchMetricsContext.setGraphMetrics(graph.getEdgesTraversed(), graph.getNeighborSeeks(), graph.getNeighborBytesRead());
```

**Option B: Store on the VectorSearcher instance**
`FaissMemoryOptimizedSearcher` is per-field, per-segment. After `doSearch()`, the graph metrics could be stored on the searcher and read by the caller. But `MemoryOptimizedKNNWeight` accesses the searcher indirectly through `reader.getVectorReader().search()`.

**Recommendation:** Thread-local context object that holds all per-segment metrics. Set/reset in `MemoryOptimizedKNNWeight.doANNSearch()`.

```java
/**
 * Thread-local context for accumulating search metrics within a single segment search.
 */
public class SearchMetricsContext {
    private static final ThreadLocal<SegmentSearchMetrics> CURRENT = ThreadLocal.withInitial(SegmentSearchMetrics::new);

    public static SegmentSearchMetrics current() { return CURRENT.get(); }
    public static void reset() { CURRENT.set(new SegmentSearchMetrics()); }
}
```

All instrumentation points (`FaissHnswGraph`, `PrefetchHelper`) write to `SearchMetricsContext.current()`. `MemoryOptimizedKNNWeight.doANNSearch()` calls `reset()` before search and reads after.

## 4. Output: OpenSearch Telemetry Framework (OTEL-native)

Instead of custom log files, integrate directly with OpenSearch's built-in `MetricsRegistry` (`org.opensearch.telemetry.metrics`). When the `telemetry-otel` plugin is installed, metrics are automatically exported via OpenTelemetry to any configured backend (Prometheus, CloudWatch, Datadog, etc.).

### 4.1 How OpenSearch Telemetry Works

OpenSearch provides `MetricsRegistry` as an abstraction. The `telemetry-otel` plugin implements it via `OTelMetricsTelemetry`, which wraps the OpenTelemetry SDK's `Meter`. Available metric types:

| Type | Method | Use case |
|---|---|---|
| Counter | `createCounter(name, desc, unit)` | Monotonically increasing totals (e.g., total bytes read) |
| UpDownCounter | `createUpDownCounter(name, desc, unit)` | Values that go up and down |
| Histogram | `createHistogram(name, desc, unit)` | Distribution of values (e.g., vectors scored per query) |
| Gauge | `createGauge(name, desc, unit, supplier, tags)` | Point-in-time values polled periodically |

Metrics are recorded with `Tags` (dimensions/labels) for slicing by index, shard, algorithm, etc.

### 4.2 Metric Definitions

```java
public class KNNSearchMetricsTelemetry {

    private final Counter vectorBytesPrefetchedCounter;
    private final Counter neighborBytesReadCounter;
    private final Counter edgesTraversedCounter;
    private final Counter neighborSeeksCounter;
    private final Histogram vectorsScoredHistogram;
    private final Histogram totalBytesReadHistogram;

    public KNNSearchMetricsTelemetry(MetricsRegistry metricsRegistry) {
        this.vectorBytesPrefetchedCounter = metricsRegistry.createCounter(
            "knn.search.vector_bytes_prefetched",
            "Total bytes prefetched for vector data during KNN search",
            "bytes"
        );
        this.neighborBytesReadCounter = metricsRegistry.createCounter(
            "knn.search.neighbor_bytes_read",
            "Total bytes read for HNSW neighbor lists during KNN search",
            "bytes"
        );
        this.edgesTraversedCounter = metricsRegistry.createCounter(
            "knn.search.edges_traversed",
            "Total HNSW edges traversed during KNN search",
            "1"
        );
        this.neighborSeeksCounter = metricsRegistry.createCounter(
            "knn.search.neighbor_seeks",
            "Total HNSW neighbor list loads during KNN search",
            "1"
        );
        this.vectorsScoredHistogram = metricsRegistry.createHistogram(
            "knn.search.vectors_scored",
            "Distribution of vectors scored per KNN query",
            "1"
        );
        this.totalBytesReadHistogram = metricsRegistry.createHistogram(
            "knn.search.total_bytes_read",
            "Distribution of total bytes read per KNN query",
            "bytes"
        );
    }

    public void record(KNNSearchMetrics metrics, Tags tags) {
        vectorBytesPrefetchedCounter.add(metrics.getVectorBytesPrefetched(), tags);
        neighborBytesReadCounter.add(metrics.getNeighborBytesRead(), tags);
        edgesTraversedCounter.add(metrics.getEdgesTraversed(), tags);
        neighborSeeksCounter.add(metrics.getNeighborSeeks(), tags);
        vectorsScoredHistogram.record(metrics.getVectorsScored(), tags);
        totalBytesReadHistogram.record(metrics.totalBytesRead(), tags);
    }
}
```

### 4.3 Tags (Dimensions)

Each metric is tagged for slicing in the observability backend:

```java
Tags tags = Tags.create()
    .addTag("index", indexName)
    .addTag("shard", String.valueOf(shardId))
    .addTag("algorithm", algorithm)   // "hnsw", "cluster", etc.
    .addTag("node_id", nodeId);
```

This gives you per-shard, per-node, per-cluster aggregation for free in Prometheus/Grafana/CloudWatch — no in-process aggregation needed.

### 4.4 Call Site

In `KNNWeight`, after all segments are searched:

```java
// After all searchLeaf() calls complete
KNNSearchMetrics metrics = queryMetrics.toSearchMetrics(earlyTerminated, totalResults);

Tags tags = Tags.create()
    .addTag("index", knnQuery.getIndexName())
    .addTag("shard", String.valueOf(knnQuery.getShardId()))
    .addTag("algorithm", algorithm);

knnSearchMetricsTelemetry.record(metrics, tags);
```

### 4.5 Accessing MetricsRegistry in the Plugin

The k-NN plugin gets `MetricsRegistry` from the `NodeEnvironment` or via `TelemetrySettings`:

```java
// In KNNPlugin.createComponents():
MetricsRegistry metricsRegistry = ... // obtained from OpenSearch's telemetry framework
KNNSearchMetricsTelemetry searchMetrics = new KNNSearchMetricsTelemetry(metricsRegistry);
```

When `telemetry-otel` plugin is not installed, `MetricsRegistry` is a no-op implementation — zero overhead.

### 4.6 Debug Logging

When debug logging is enabled on the k-NN logger, per-query metrics are also emitted via the standard logger. This allows developers to see metrics in the main OpenSearch log without needing an OTEL backend.

```java
private static final Logger log = LogManager.getLogger(KNNSearchMetricsTelemetry.class);

public void record(KNNSearchMetrics metrics, Tags tags) {
    // Always record to OTEL (no-op if plugin not installed)
    vectorBytesPrefetchedCounter.add(metrics.getVectorBytesPrefetched(), tags);
    neighborBytesReadCounter.add(metrics.getNeighborBytesRead(), tags);
    vectorsScoredHistogram.record(metrics.getVectorsScored(), tags);
    totalBytesReadHistogram.record(metrics.totalBytesRead(), tags);

    // Debug log for local observability without OTEL backend
    if (log.isDebugEnabled()) {
        log.debug(
            "KNN search metrics: vectors_scored={}, edges_traversed={}, "
            + "vector_bytes_prefetched={}, neighbor_bytes_read={}, total_bytes_read={}, "
            + "results={}, tags={}",
            metrics.getVectorsScored(),
            metrics.getEdgesTraversed(),
            metrics.getVectorBytesPrefetched(),
            metrics.getNeighborBytesRead(),
            metrics.totalBytesRead(),
            metrics.getResultsReturned(),
            tags
        );
    }
}
```

Enable via `opensearch.yml` or dynamically:
```yaml
logger.org.opensearch.knn.plugin.stats.KNNSearchMetricsTelemetry: debug
```

This adds zero overhead in production (`isDebugEnabled()` is a single branch-predicted check) and gives full per-query visibility during development or troubleshooting.

### 4.6 What the OTEL Backend Sees

With Prometheus exporter configured, metrics appear as:

```
# TYPE knn_search_vectors_scored histogram
knn_search_vectors_scored_bucket{index="my-vectors",shard="0",algorithm="hnsw",le="50"} 120
knn_search_vectors_scored_bucket{index="my-vectors",shard="0",algorithm="hnsw",le="100"} 450
knn_search_vectors_scored_bucket{index="my-vectors",shard="0",algorithm="hnsw",le="200"} 980

# TYPE knn_search_vector_bytes_prefetched_total counter
knn_search_vector_bytes_prefetched_total{index="my-vectors",shard="0",algorithm="hnsw"} 1073741824

# TYPE knn_search_total_bytes_read histogram
knn_search_total_bytes_read_bucket{index="my-vectors",shard="0",algorithm="hnsw",le="65536"} 200
knn_search_total_bytes_read_bucket{index="my-vectors",shard="0",algorithm="hnsw",le="131072"} 800
```

Aggregation at shard/node/cluster level is done by the observability backend using the tag dimensions.

### 4.7 Advantages Over Custom Log File

| Aspect | Custom log file | MetricsRegistry (OTEL) |
|---|---|---|
| Setup | Custom log4j appender + OTEL filelog receiver | Zero — works with existing telemetry-otel plugin |
| Aggregation | Must parse JSON, extract fields, aggregate | Built-in via Prometheus/OTEL labels |
| Overhead | File I/O per query (even with async appender) | In-memory metric recording, batched export |
| Cardinality | One log line per query (high volume) | Pre-aggregated counters/histograms (low volume) |
| Dashboarding | Custom parsing rules per backend | Standard OTEL metric format, auto-discovered |

## 5. Aggregation Hierarchy

```
Per-Segment (SegmentSearchMetrics via SearchMetricsContext)
    ↓ merge across segments in KNNWeight
Per-Query (KNNSearchMetrics)
    ↓ record into MetricsRegistry with tags
OpenSearch Telemetry Framework (MetricsRegistry)
    ↓ OTelMetricsTelemetry exports via OTEL SDK
OTEL Backend (Prometheus / CloudWatch / Datadog)
    → per-shard, per-node, per-cluster dashboards via tag dimensions
```

No in-process shard/node/cluster aggregation needed — the observability backend handles it using the `index`, `shard`, `algorithm`, and `node_id` tags.

## 6. Thread Safety

| Level | Concurrency | Safety mechanism |
|---|---|---|
| Per-segment | Single-threaded (one thread per segment search) | No synchronization needed. Thread-local `SearchMetricsContext`. |
| Per-query | Sequential segments within a query | `SegmentSearchMetrics.merge()` called sequentially in `searchLeaf()` loop. |
| Metric recording | One `record()` per query, on the search thread | `MetricsRegistry` implementations are thread-safe (OTEL SDK uses lock-free accumulators). |

## 7. Performance Impact

- **Per-segment instrumentation:** 3 `long` increments in `FaissHnswGraph` (hot path) + 2 `long` increments in `PrefetchHelper` (per group, not per vector) + 1 read of `visitedCount` (already computed). **Negligible.**
- **Per-query aggregation:** One `merge()` call per segment (a few additions). **Negligible.**
- **Metric recording:** One `counter.add()` + `histogram.record()` per query. OTEL SDK uses lock-free delta aggregation internally. **Negligible.**
- **When telemetry-otel not installed:** `MetricsRegistry` is a no-op. **Zero overhead.**
- **Thread-local access:** `ThreadLocal.get()` is ~1ns. **Negligible.**

### 7.1 Pre-building Base Tags

`Tags` creation involves array allocation and sorting. To avoid per-query allocation, pre-build base tags at shard initialization and only concat query-specific parts at query time:

```java
// At shard init (once, stored on the shard):
Tags baseTags = Tags.ofStringPairs("index", indexName, "shard", String.valueOf(shardId));

// At query time (cheap concat — one small array copy):
Tags queryTags = Tags.concat(baseTags, Tags.of("algorithm", algorithm));
```

This reduces per-query tag overhead from ~200ns (full creation) to ~50ns (concat of pre-sorted arrays).

## 8. Cluster ANN Considerations

For `ClusterANN1040KnnVectorsReader.search()`, the same `SearchMetricsContext` is used:
- `vectorsScored` = loop iteration count (already tracked via `incVisitedCount`)
- `vectorBytesPrefetched` = 0 (no prefetch in brute-force, all vectors read sequentially)
- `neighborBytesRead` = 0 (no graph structure)
- `algorithm` = `"cluster"`
- Future cluster index will add: `centroidsCompared`, `memberListsScanned`

The log schema is extensible — new fields can be added without breaking OTEL parsers.
