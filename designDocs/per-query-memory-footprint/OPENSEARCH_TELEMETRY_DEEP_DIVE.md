# OpenSearch Telemetry Framework — Deep Dive

## 1. High-Level Architecture

OpenSearch has a built-in telemetry framework (`libs/telemetry`) that provides an abstraction layer for metrics. The `telemetry-otel` plugin implements this abstraction using the OpenTelemetry Java SDK, enabling metrics export to any OTEL-compatible backend.

```mermaid
graph TB
    subgraph "OpenSearch Node"
        subgraph "Plugin Code (k-NN)"
            KW[KNNWeight] -->|"counter.add(value, tags)"| MR
            KW -->|"histogram.record(value, tags)"| MR
        end

        subgraph "libs/telemetry (Abstraction Layer)"
            MR[MetricsRegistry Interface]
            C[Counter Interface]
            H[Histogram Interface]
            G[Gauge Interface]
            T[Tags - Immutable Dimensions]
            MR --> C
            MR --> H
            MR --> G
        end

        subgraph "telemetry-otel Plugin (Implementation)"
            OTM[OTelMetricsTelemetry]
            OTC[OTelCounter]
            OTH[OTelHistogram]
            SDK[OpenTelemetry SDK]
            OTM --> OTC
            OTM --> OTH
            OTM --> SDK
        end

        MR -.->|"delegates to"| OTM
    end

    subgraph "Export Pipeline"
        SDK -->|"OTLP gRPC/HTTP"| COLLECTOR[OTEL Collector]
        COLLECTOR --> PROM[Prometheus]
        COLLECTOR --> CW[CloudWatch]
        COLLECTOR --> DD[Datadog]
    end
```

## 2. Component Breakdown

### 2.1 Abstraction Layer (`libs/telemetry`)

Lives in `libs/telemetry/src/main/java/org/opensearch/telemetry/metrics/`. This is what plugins code against.

```mermaid
classDiagram
    class MetricsRegistry {
        <<interface>>
        +createCounter(name, description, unit) Counter
        +createUpDownCounter(name, description, unit) Counter
        +createHistogram(name, description, unit) Histogram
        +createGauge(name, description, unit, supplier, tags) Closeable
        +close()
    }

    class MetricsTelemetry {
        <<interface>>
        extends MetricsRegistry
    }

    class Counter {
        <<interface>>
        +add(double value)
        +add(double value, Tags tags)
    }

    class Histogram {
        <<interface>>
        +record(double value)
        +record(double value, Tags tags)
    }

    class Tags {
        -String[] keys
        -Object[] values
        +of(key, value) Tags
        +addTag(key, value) Tags
        +concat(Tags a, Tags b) Tags
        +getTagsMap() Map
    }

    class DefaultMetricsRegistry {
        -MetricsTelemetry metricsTelemetry
        +createCounter() Counter
        +createHistogram() Histogram
    }

    MetricsRegistry <|-- MetricsTelemetry
    MetricsRegistry <|.. DefaultMetricsRegistry
    DefaultMetricsRegistry --> MetricsTelemetry : delegates
    MetricsRegistry ..> Counter : creates
    MetricsRegistry ..> Histogram : creates
    Counter ..> Tags : uses
    Histogram ..> Tags : uses
```

### 2.2 OTEL Implementation (`plugins/telemetry-otel`)

```mermaid
classDiagram
    class OTelMetricsTelemetry {
        -RefCountedReleasable~OpenTelemetrySdk~ refCountedOpenTelemetry
        -Meter otelMeter
        -T meterProvider
        +createCounter(name, desc, unit) Counter
        +createHistogram(name, desc, unit) Histogram
        +createGauge(name, desc, unit, supplier, tags) Closeable
    }

    class OTelCounter {
        -DoubleCounter doubleCounter
        +add(double value)
        +add(double value, Tags tags)
    }

    class OTelHistogram {
        -DoubleHistogram doubleHistogram
        +record(double value)
        +record(double value, Tags tags)
    }

    class OTelAttributesConverter {
        +convert(Tags tags) Attributes$
    }

    MetricsTelemetry <|.. OTelMetricsTelemetry
    OTelMetricsTelemetry ..> OTelCounter : creates
    OTelMetricsTelemetry ..> OTelHistogram : creates
    OTelCounter --> OTelAttributesConverter : uses
    OTelHistogram --> OTelAttributesConverter : uses
```

### 2.3 No-Op Implementation (`noop/`)

When `telemetry-otel` plugin is NOT installed, OpenSearch uses no-op implementations. All metric operations become empty method calls — zero overhead.

```mermaid
classDiagram
    class NoopMetricsRegistry {
        +createCounter() NoopCounter
        +createHistogram() NoopHistogram
    }

    class NoopCounter {
        +add(value) void -- no-op
        +add(value, tags) void -- no-op
    }

    class NoopHistogram {
        +record(value) void -- no-op
        +record(value, tags) void -- no-op
    }

    MetricsRegistry <|.. NoopMetricsRegistry
    NoopMetricsRegistry ..> NoopCounter
    NoopMetricsRegistry ..> NoopHistogram
```

## 3. Data Flow

```mermaid
sequenceDiagram
    participant Plugin as k-NN Plugin
    participant Registry as MetricsRegistry
    participant OTel as OTelMetricsTelemetry
    participant SDK as OpenTelemetry SDK
    participant Exporter as OTLP Exporter
    participant Backend as Prometheus/CW

    Note over Plugin: At plugin startup
    Plugin->>Registry: createCounter("knn.search.bytes_read", ...)
    Registry->>OTel: createCounter(...)
    OTel->>SDK: meter.counterBuilder(...).build()
    SDK-->>OTel: DoubleCounter
    OTel-->>Registry: OTelCounter
    Registry-->>Plugin: Counter

    Note over Plugin: At query time (per query)
    Plugin->>Registry: counter.add(140672, Tags.of("index","my-vec"))
    Registry->>OTel: OTelCounter.add(140672, tags)
    OTel->>SDK: doubleCounter.add(140672, attributes)
    Note over SDK: Accumulated in-memory (lock-free)

    Note over SDK: Periodic export (default 60s)
    SDK->>Exporter: batch of metric points
    Exporter->>Backend: OTLP gRPC/HTTP push
```

## 4. Key Design Decisions

### 4.1 Metric Types and When to Use

| Type | Semantics | Use for |
|---|---|---|
| **Counter** | Monotonically increasing sum | Cumulative totals: total bytes read, total edges traversed |
| **Histogram** | Distribution of values | Per-query distributions: vectors scored per query, bytes per query |
| **Gauge** | Point-in-time snapshot | Current state: cache size, active searches |
| **UpDownCounter** | Sum that can decrease | In-flight counts: concurrent searches |

### 4.2 Tags (Dimensions)

`Tags` is immutable, sorted by key, supports String/long/double/boolean values. Used for slicing metrics in the backend.

```java
// Creation patterns
Tags.of("index", "my-vectors")                          // single tag
Tags.of("index", "my-vectors").addTag("shard", 0L)      // chained
Tags.ofStringPairs("index", "my-vectors", "algo", "hnsw") // bulk
Tags.concat(baseTags, queryTags)                         // merge
```

**Important:** High-cardinality tags (e.g., query ID, user ID) should be avoided — they explode metric series in the backend.

### 4.3 Thread Safety

- `Counter.add()` and `Histogram.record()` are thread-safe (OTEL SDK uses lock-free accumulators internally)
- `Tags` is immutable — safe to share across threads
- Metric instruments (Counter, Histogram) are created once at startup and reused

### 4.4 Export Configuration

The OTEL SDK is configured via `opensearch.yml`:

```yaml
telemetry.otel.metrics.exporter.class: io.opentelemetry.exporter.otlp.metrics.OtlpGrpcMetricExporter
telemetry.otel.metrics.exporter.endpoint: http://otel-collector:4317
telemetry.otel.metrics.export.interval: 60s
```

## 5. How k-NN Plugin Integrates

### 5.1 Getting MetricsRegistry

```mermaid
sequenceDiagram
    participant OS as OpenSearch Core
    participant KNN as KNNPlugin
    participant MR as MetricsRegistry

    Note over OS: Node startup
    OS->>OS: Initialize telemetry framework
    OS->>OS: Load telemetry-otel plugin (if present)
    OS->>MR: Create DefaultMetricsRegistry(otelTelemetry)

    Note over OS: Plugin initialization
    OS->>KNN: createComponents(metricsRegistry, ...)
    KNN->>MR: createCounter("knn.search.vectors_scored", ...)
    KNN->>MR: createHistogram("knn.search.total_bytes_read", ...)
    KNN->>KNN: Store metric instruments for query-time use
```

### 5.2 Recording at Query Time

```java
// In KNNWeight, after search completes:
KNNSearchMetrics metrics = queryMetrics.toSearchMetrics(...);

Tags tags = Tags.ofStringPairs(
    "index", knnQuery.getIndexName(),
    "algorithm", algorithm
).addTag("shard", (long) knnQuery.getShardId());

// Type A: Data Read
vectorBytesPrefetchedCounter.add(metrics.getVectorBytesPrefetched(), tags);
neighborBytesReadCounter.add(metrics.getNeighborBytesRead(), tags);

// Type B: Traversal (as histogram for distribution)
vectorsScoredHistogram.record(metrics.getVectorsScored(), tags);
edgesTraversedHistogram.record(metrics.getEdgesTraversed(), tags);
```

## 6. What Happens Without telemetry-otel Plugin

```mermaid
graph LR
    KNN[k-NN Plugin] -->|"counter.add()"| NOOP[NoopCounter]
    NOOP -->|"empty method body"| NOTHING[Nothing happens]
```

The abstraction guarantees zero overhead when telemetry is not configured. The k-NN plugin doesn't need to check if telemetry is enabled — it always calls `counter.add()` / `histogram.record()`, and the no-op implementation discards the call.

## 7. Summary for k-NN Search Metrics Integration

| Concern | How it's handled |
|---|---|
| Metric creation | Once at plugin startup via `MetricsRegistry.createCounter/Histogram` |
| Metric recording | Per-query via `counter.add(value, tags)` / `histogram.record(value, tags)` |
| Thread safety | Built into OTEL SDK (lock-free accumulators) |
| Export | Automatic via OTEL SDK periodic export (configurable interval) |
| Aggregation | Done by backend (Prometheus, CloudWatch) using tag dimensions |
| Zero overhead when disabled | No-op implementations in abstraction layer |
| Tag dimensions | `index`, `shard`, `algorithm`, `node_id` |
| No custom log files | Metrics go directly through OTEL SDK export pipeline |
