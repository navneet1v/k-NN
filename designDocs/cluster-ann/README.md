# Cluster ANN Design Documents

## Reading Order

1. **KNN_VECTOR_FIELD_MAPPER.md** — Background on the existing `KNNVectorFieldMapper` hierarchy, how fields are mapped, and how engines are resolved. Read this first to understand the baseline architecture.

2. **KNN_ENGINE.md** — How `KNNEngine`, `KNNLibrary`, and `EngineResolver` work. Explains why engine-less algorithms need a different path.

3. **CLUSTER_ANN_INTEGRATION_PLAN.md** — The main design document. Covers the mapping API design (6 alternatives evaluated), validation rules, mapper routing, codec/query integration points, performance constraints, and the full task list with status.

4. **CLUSTER_VECTOR_FIELD_MAPPER_LLD.md** — Low-level design for `ClusterANNVectorFieldMapper`, the engine-less mapper that owns indexing for cluster fields.

5. **CLUSTER_VECTOR_FIELD_MAPPER_TEST_PLAN.md** — Test plan for the mapper layer (unit tests and validation).

6. **CLUSTER_ANN_SEARCH_PATH_LLD.md** — Low-level design for the search/query path: how cluster fields route through `MemoryOptimizedKNNWeight` and the Lucene reader.

## Key Concepts

- **Engine-less algorithm**: An algorithm that doesn't tie to any `KNNEngine` (Faiss, NMSLIB, Lucene). It owns its own indexing and search end-to-end via Lucene's native `KnnVectorsFormat`.
- **`method.name: cluster`**: The routing key in the mapping API. When the method name is engine-less, it bypasses `EngineResolver` entirely.
- **`alwaysUseMemoryOptimizedSearch`**: Flag set on `KNNVectorFieldType` that forces the query path through `MemoryOptimizedKNNWeight` → Lucene reader, skipping native JNI calls.
