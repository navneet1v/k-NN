# Per-Query Memory Footprint — Design Documents

## Documents

1. **INTEGRATION_PLAN.md** — Problem statement, audit of available metrics in the memory-optimized path, two instrumentation types (Data Read + Search Traversal), and task breakdown.

2. **LLD.md** — Low-level design: data model (`SegmentSearchMetrics`, `KNNSearchMetrics`, `SearchMetricsContext`), capture points (`FaissHnswGraph`, `PrefetchHelper`, `ClusterANN1040KnnVectorsReader`), wiring through `MemoryOptimizedKNNWeight` and `KNNWeight`, output via OpenSearch `MetricsRegistry` (OTEL) + debug logging.

3. **OPENSEARCH_TELEMETRY_DEEP_DIVE.md** — Deep dive on OpenSearch's `OTelMetricsTelemetry` framework: architecture, class diagrams, data flow, metric types, Tags API, and how the k-NN plugin integrates.

## Implementation Status

All 4 phases implemented and tested:

- **Phase 1:** Data structures — `SegmentSearchMetrics`, `KNNSearchMetrics`, `SearchMetricsContext`
- **Phase 2:** Instrumentation — counters in `FaissHnswGraph`, `PrefetchHelper`, `ClusterANN1040KnnVectorsReader`
- **Phase 3:** Wiring — `FaissMemoryOptimizedSearcher` → `SearchMetricsContext` → `MemoryOptimizedKNNWeight` → `KNNWeight` aggregation
- **Phase 4:** Emission — `KNNSearchMetricsEmitter` with debug logging

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Per-Segment Search                            │
│                                                                     │
│  SearchMetricsContext.reset()                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  reader.getVectorReader().search()                       │       │
│  │       │                                                  │       │
│  │       ├── FaissHnswGraph                                 │       │
│  │       │     seek() → neighborSeeks++, neighborBytesRead  │       │
│  │       │     nextNeighbor() → edgesTraversed++            │       │
│  │       │                                                  │       │
│  │       ├── PrefetchHelper                                 │       │
│  │       │     prefetch() → vectorBytesPrefetched,          │       │
│  │       │                   prefetchGroupCount             │       │
│  │       │                                                  │       │
│  │       └── ClusterANN1040KnnVectorsReader                 │       │
│  │             search loop → vectorBytesRead                │       │
│  └─────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  FaissMemoryOptimizedSearcher writes graph metrics                  │
│  to SearchMetricsContext.current()                                  │
│       │                                                             │
│       ▼                                                             │
│  MemoryOptimizedKNNWeight captures                                  │
│  knnCollector.visitedCount() → SearchMetricsContext                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Per-Query Aggregation                         │
│                                                                     │
│  KNNWeight.searchLeaf()                                             │
│       │                                                             │
│       ├── querySearchMetrics.merge(SearchMetricsContext.current())  │
│       │                                                             │
│       └── KNNSearchMetricsEmitter.emit(metrics, index, shard, algo) │
│             │                                                       │
│             ├── Debug log (if enabled)                               │
│             └── MetricsRegistry counters/histograms (future OTEL)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Observability Backend                         │
│                                                                     │
│  Option A: Debug logging                                            │
│    logger.org.opensearch.knn.index.query.metrics                    │
│    .KNNSearchMetricsEmitter: debug                                  │
│                                                                     │
│  Option B: OTEL (future)                                            │
│    MetricsRegistry → OTelMetricsTelemetry → OTLP exporter          │
│    → Prometheus / CloudWatch / Datadog                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Running with OTEL Metrics Enabled

**Note:** OTEL metrics require the `telemetry-otel` plugin, which is bundled with the OpenSearch distribution but NOT with the k-NN plugin's `./gradlew run` test cluster. The OTEL integration is a follow-up task. For now, use debug logging.

### Option 1: Debug Logging (works immediately, no extra plugins)

```bash
./gradlew run -Dtests.opensearch.logger.org.opensearch.knn.index.query.metrics.KNNSearchMetricsEmitter=debug
```

This enables per-query metrics in the main OpenSearch log. Run a KNN search and check the log output.

### Option 2: OTEL Metrics (downloads and installs telemetry-otel plugin)

```bash
./gradlew run -Dtelemetry.enabled=true
```

This automatically:
- Downloads the `telemetry-otel` plugin from CI
- Sets the JVM feature flag for experimental telemetry
- Configures `telemetry.feature.metrics.enabled: true`
- Configures the `LoggingMetricExporter` (writes to `logs/<cluster_name>_otel_metrics.log`)

For gRPC export to an OTEL collector, edit `build.gradle` and change the exporter class to `io.opentelemetry.exporter.otlp.metrics.OtlpGrpcMetricExporter`.

To change the publish interval, add to the telemetry block in `build.gradle`:
```groovy
cluster.setting('telemetry.otel.metrics.publish.interval', '10s')
```

## Enabling Debug Logging for KNN Search Metrics

Without OTEL, you can see per-query metrics via debug logging. Add to `opensearch.yml` or set dynamically:

```yaml
logger.org.opensearch.knn.index.query.metrics.KNNSearchMetricsEmitter: debug
```

Or dynamically via API:

```bash
PUT /_cluster/settings
{
  "transient": {
    "logger.org.opensearch.knn.index.query.metrics.KNNSearchMetricsEmitter": "debug"
  }
}
```

Example output in the main OpenSearch log:
```
[DEBUG][o.o.k.i.q.m.KNNSearchMetricsEmitter] KNN search metrics: index=my-vectors, shard=0, algorithm=hnsw, vectors_scored=150, edges_traversed=2400, neighbor_seeks=150, vector_bytes_prefetched=131072, neighbor_bytes_read=9600, total_bytes_read=140672, prefetch_groups=8, early_terminated=false, results_returned=10
```
