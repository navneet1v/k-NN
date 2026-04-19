# Cluster-based ANN Algorithm — Design Document

## 1. Overview

This document describes the design for integrating a cluster-based ANN (Approximate Nearest Neighbor) algorithm into the OpenSearch k-NN plugin. The algorithm is **engine-less** — it does not tie to any `KNNEngine` (Faiss, Lucene, NMSLIB). The plugin should not accept or resolve an engine for this algorithm.

### Two Paths in the Plugin

- **Engine-based path** (existing): User picks engine → engine resolves algorithm → engine owns indexing + search
- **Algorithm-based path** (new): User picks algorithm directly via `method.name` → algorithm-specific mapper owns indexing + search end-to-end. No `KNNEngine`, no `KNNMethodContext`, no `EngineResolver`.

## 2. Mapping API Design

### 2.1 Alternatives Considered

#### Alternative A: New `algorithm` top-level parameter with separate `algorithm_parameters`

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "algorithm": "cluster",
    "space_type": "l2",
    "algorithm_parameters": {
      "num_clusters": 256,
      "sample_size": 10000
    }
  }
}
```

Pros:
- Clean separation — `algorithm` picks the mapper, `algorithm_parameters` is a generic bag each algorithm validates independently
- Easy to extend for future algorithms

Cons:
- Two new top-level fields to manage
- `algorithm_parameters` name is generic and doesn't hint at what's inside
- Introduces a new concept alongside the existing `method` block, increasing cognitive load for users
- Doesn't reuse any existing parsing infrastructure

#### Alternative B: Nested `algorithm` object with `name` and `parameters` sub-object

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "algorithm": {
      "name": "cluster",
      "parameters": {
        "num_clusters": 256,
        "sample_size": 10000
      }
    },
    "space_type": "l2"
  }
}
```

Pros:
- Self-contained — algorithm name and its parameters live together
- Mirrors the existing `method` block structure, familiar to users
- Each algorithm defines its own parameter schema inside `parameters`

Cons:
- Similar shape to `method` which could confuse users about when to use which
- Two levels of nesting (`algorithm.parameters.<param>`)
- Introduces a parallel concept to `method` — users must learn when to use `algorithm` vs `method`

#### Alternative C: Nested `algorithm` object, parameters as siblings of `name` (no `parameters` wrapper)

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "algorithm": {
      "name": "cluster",
      "num_clusters": 256,
      "sample_size": 10000
    },
    "space_type": "l2"
  }
}
```

Pros:
- One level less nesting than Alternative B
- Algorithm owns its config cleanly
- No `parameters` wrapper that adds depth without meaning

Cons:
- `name` is a reserved key inside the block, so no algorithm can have a parameter called `name`
- Still introduces a new `algorithm` concept parallel to `method`

#### Alternative D: Algorithm name as the key itself

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "space_type": "l2",
    "cluster": {
      "num_clusters": 256,
      "sample_size": 10000
    }
  }
}
```

Pros:
- Most concise, zero boilerplate
- No reserved keys inside the block

Cons:
- Parser must recognize `cluster` as an algorithm name vs an unknown field
- Mutual exclusion (can't have both `cluster` and `hnsw` blocks) needs explicit enforcement
- Doesn't scale cleanly if you want to add metadata alongside the algorithm
- Not extensible — every new algorithm pollutes the top-level namespace

#### Alternative E: DSL string (like Faiss index descriptions)

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "algorithm": "cluster(num_clusters=256, sample_size=10000)",
    "space_type": "l2"
  }
}
```

Pros:
- Extremely compact, power users love it

Cons:
- Terrible for programmatic generation
- Hard to validate with good error messages
- Not JSON-native, breaks tooling that expects structured data
- No precedent in OpenSearch mapping APIs

#### Alternative F (Recommended): Reuse existing `method` block, engine-less routing via `name`

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "space_type": "l2",
    "method": {
      "name": "cluster",
      "parameters": {
        "num_clusters": 256,
        "sample_size": 10000
      }
    }
  }
}
```

Pros:
- **Zero new concepts** — reuses the `method` block users already know
- `name` becomes the routing key: engine-based names (hnsw, ivf) go through engine path; engine-less names (cluster) go through algorithm path
- `parameters` scoping already exists and each algorithm defines its own schema
- `space_type` stays where it is
- Backward compatible — existing mappings with `engine` still work unchanged
- No new top-level fields, no parallel concepts
- Reuses existing parsing infrastructure (`KNNMethodContext.parse()`)

Cons:
- The `method` block now serves two purposes (engine-based and engine-less), which could be confusing if not documented clearly
- `engine` becomes optional/forbidden depending on `name`, adding conditional validation

### 2.2 Industry Comparison

| System | How algorithm is specified | Parameters |
|---|---|---|
| **Elasticsearch** | `index_options.type` (e.g., `hnsw`, `int8_hnsw`, `bbq_disk`) | Siblings of `type` inside `index_options` |
| **Milvus** | `index_type` in separate `create_index` API | `params` sub-object |
| **Qdrant** | Collection-level `hnsw_config` (HNSW only) | Direct fields in config object |
| **OpenSearch k-NN (existing)** | `method.engine` + `method.name` | `method.parameters` sub-object |
| **OpenSearch k-NN (proposed)** | `method.name` (engine-less for new algorithms) | `method.parameters` sub-object |

Elasticsearch's approach is the closest to what we're building — per-field, engine-less, algorithm type as the routing key. Our proposed approach (Alternative F) aligns with this pattern while reusing the existing OpenSearch k-NN `method` block structure.

### 2.3 Recommended Approach

**Alternative F** is recommended because:

1. **No new concepts** — Users already understand the `method` block. Adding a new algorithm is just a new `name` value.
2. **Industry-aligned** — Follows the same pattern as Elasticsearch's `index_options.type` where the algorithm type is the single routing key.
3. **Backward compatible** — Existing engine-based mappings work unchanged. The `engine` field simply becomes irrelevant for engine-less algorithms.
4. **Minimal code change** — Reuses existing `KNNMethodContext` parsing. The main change is in `TypeParser` validation and `Builder.build()` routing.
5. **Extensible** — Future engine-less algorithms just register a new `name` value. No structural changes needed.

### 2.4 Chosen Design

Reuse the existing `method` block. The `name` field routes to the algorithm.

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "space_type": "l2",
    "method": {
      "name": "cluster"
    }
  }
}
```

Note: `method.parameters` for the cluster algorithm (e.g., `num_clusters`, `sample_size`) are not included in the initial implementation. They will be designed and added in a follow-up.

For comparison, existing engine-based mapping:

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "space_type": "l2",
    "method": {
      "engine": "faiss",
      "name": "hnsw",
      "parameters": { "m": 16, "ef_construction": 512 }
    }
  }
}
```

### 2.5 Validation Rules

When `method.name` is an engine-less algorithm (e.g., `cluster`):

Reject (with clear error):
- `engine` (in method or top-level) — "Engine cannot be specified for algorithm 'cluster'"
- `model_id` — "model_id cannot be used with algorithm 'cluster'"
- `mode` / `compression` — "mode/compression cannot be used with algorithm 'cluster'"

Accept:
- `dimension` (required)
- `space_type` (in method or top-level, optional, default L2)
- `data_type` (optional, default float)
- `parameters` — algorithm-specific, validated by the algorithm

## 3. Mode and Compression Parameters

### Current Behavior

The existing `mode` and `compression` top-level mapping parameters are **engine-resolution hints**, not algorithm parameters.

- **`mode`** (`in_memory` | `on_disk`): Tells the system whether to optimize for memory or disk. Feeds into `EngineResolver.resolveEngine()` to pick the right engine (e.g., `on_disk` → Faiss) and determines whether rescoring is applied after quantized search.
- **`compression`** (`1x`, `2x`, `4x`, `8x`, `16x`, `32x`): Tells the system how aggressively to quantize vectors. Influences engine choice (e.g., `4x` → Lucene) and encoder selection within the engine (e.g., SQ, PQ, BBQ).

Both exist as shortcuts so users don't have to manually specify the full engine + method + encoder configuration. They feed into `EngineResolver` and `engine.resolveMethod()` to auto-configure the right combination.

Example of how they work today:

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "mode": "on_disk",
    "compression": "32x"
  }
}
```

This resolves to Faiss HNSW with binary quantization, rescoring enabled — without the user specifying any of that explicitly.

### Behavior with Engine-less Algorithms

`mode` and `compression` are **rejected** when `method.name` is an engine-less algorithm (e.g., `cluster`). Reasons:

1. **No engine to resolve** — These parameters exist to help `EngineResolver` pick an engine. Engine-less algorithms bypass `EngineResolver` entirely.
2. **Algorithm owns its own storage** — The cluster algorithm controls its own memory/disk tradeoffs and quantization strategy directly. These are internal implementation details of the algorithm, not external hints.
3. **Avoids ambiguity** — If `compression: 32x` were accepted alongside `method.name: cluster`, it's unclear what it means. Does the cluster algorithm quantize to 1 bit? Does it use a specific encoder? The algorithm should define these explicitly if needed.

This keeps the algorithm self-contained — all configuration lives inside `method.parameters`, validated by the algorithm itself, with no dependency on engine-level abstractions.

### Future: Quantization via Encoder

If the cluster algorithm needs quantization in the future (e.g., reducing bits per dimension), it will use the **encoder** abstraction inside `method.parameters`. This reuses the existing pattern from Faiss HNSW rather than inventing a new mechanism.

```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "method": {
      "name": "cluster",
      "space_type": "l2",
      "parameters": {
        "num_clusters": 256,
        "encoder": {
          "name": "sq",
          "parameters": {
            "bits": 8
          }
        }
      }
    }
  }
}
```

For comparison, existing Faiss HNSW with encoder:

```json
{
  "method": {
    "engine": "faiss",
    "name": "hnsw",
    "parameters": {
      "m": 16,
      "encoder": {
        "name": "sq",
        "parameters": { "bits": 8 }
      }
    }
  }
}
```

This is consistent because:
- Encoder is already a well-understood abstraction in the codebase (`Encoder` interface, `FaissSQEncoder`, `FaissFlatEncoder`, etc.)
- Users already know the `encoder` parameter pattern
- The cluster algorithm doesn't need an engine to use an encoder — it just validates and applies the encoder itself

## 4. Mapper Routing

### Builder.build() Decision Tree

```
1. modelId set?                        → ModelFieldMapper
2. method.name is engine-less (e.g.,   → ClusterVectorFieldMapper
   "cluster")?
3. resolvedKnnMethodContext == null?    → FlatVectorFieldMapper
4. otherwise                           → EngineFieldMapper
```

## 4. Integration Points

### 4.1 Mapper Layer
- New `ClusterVectorFieldMapper` extending `KNNVectorFieldMapper`
- New routing branch in `Builder.build()` (check if `method.name` is engine-less, before FlatVectorFieldMapper check)
- New validation in `TypeParser.parse()`:
  - Reject `engine` + engine-less algorithm with error: "Engine cannot be specified for algorithm 'cluster'"
  - Reject `model_id` + engine-less algorithm
  - Reject `mode` / `compression` + engine-less algorithm

### 4.2 Codec Layer
- Custom format that builds cluster structures on segment flush/merge
- Reads vectors from doc values, no native engine index files

### 4.3 Query Layer
- Custom query implementation that searches the cluster index
- No delegation to any `KNNEngine` search path

### 4.4 Performance Constraints (for codec and query design)

The mapper layer is index-time setup code and not performance-sensitive. The codec and query layers are on the hot path and must follow these principles:

- **Vectors stored in flat contiguous memory, not as objects.** Cluster centroids and member vectors should be flat `float[]` / `byte[]` with offset arithmetic. No `Vector` wrapper objects — each object header + pointer chase costs more than the distance computation on compressed vectors.
- **Cluster membership as raw `int[]` arrays, not collections.** Mapping from cluster ID → member doc IDs should be packed `int[]`, not `List<Integer>`. Cache-line-friendly layout dominates when scanning cluster members.
- **Fused score-and-filter over separate stages.** During search, fuse filtering into the distance loop within a cluster so vectors are skipped early. Don't materialize intermediate candidate lists per cluster.
- **Batch distance computation.** When scanning vectors within a cluster, compute distances in batches to enable SIMD register reuse. Don't call `distance(query, vector)` per vector.
- **Memory-mapped vector storage.** Map vector data via `mmap` / `MemorySegment` and compute distances against the mapped region. Don't copy vectors into heap objects.
- **Pre-computed norms for cosine similarity.** Store L2 norms alongside vectors at index time. Don't compute norms lazily during search.
- **Lock-free result collection.** Use per-thread local result buffers during multi-cluster search with a final merge. No shared synchronized priority queues on the hot path.

## 5. Tasks

- [x] Understand `KNNVectorFieldMapper` hierarchy and routing
- [x] Understand `KNNEngine` / `KNNLibrary` / `EngineResolver` layer
- [x] Define the algorithm-based abstraction (engine-less, algorithm-driven mapper)
- [x] Design mapping API (evaluated 6 alternatives, chose Alternative F)
- [x] Implement validation: reject `engine` + engine-less algorithm in `TypeParser.parse()`
- [x] Implement `ClusterVectorFieldMapper`
- [ ] Implement codec format for cluster index
- [ ] Implement query path for cluster-based search
- [ ] **Refactor**: Decouple `MemoryOptimizedKNNWeight` from HNSW-specific concepts. Currently it hardcodes `KnnSearchStrategy.Hnsw`, ACORN filtering thresholds, and HNSW-specific search strategies. Engine-less algorithms (e.g., cluster) route through this weight via `alwaysUseMemoryOptimizedSearch`, but the HNSW assumptions don't apply. Refactor to extract a strategy/provider interface so each algorithm can supply its own search strategy, collector configuration, and filtering behavior without inheriting HNSW defaults.
- [ ] **Fix**: Add algorithm-specific search parameter validation for engine-less methods in `KNNQueryBuilder.doToQuery()`. Currently skipped with `knnEngine != KNNEngine.UNDEFINED` guard to avoid NPE on null `knnLibrary`. Engine-less algorithms need their own validation path for method parameters.
- [ ] **Fix**: Support radial search (`min_score` / `max_distance`) for engine-less methods. Currently blocked by "Engine [UNDEFINED] does not support radial search" check in the query path.
- [ ] **Follow-up**: Design and add `method.parameters` for cluster algorithm (e.g., `num_clusters`, `sample_size`, `encoder`)
