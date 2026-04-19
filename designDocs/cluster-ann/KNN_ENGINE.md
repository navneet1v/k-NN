# KNNEngine

## Overview

`KNNEngine` is an enum that acts as the entry point for all engine-specific operations. Each enum value wraps a `KNNLibrary` implementation and delegates to it.

```
KNNEngine (enum, implements KNNLibrary)
  ├── FAISS   → Faiss.INSTANCE
  ├── LUCENE  → Lucene.INSTANCE
  ├── NMSLIB  → Nmslib.INSTANCE  (deprecated, blocked from 3.0.0)
  └── UNDEFINED
```

Default engine: `FAISS`

## KNNLibrary Interface

Every engine must implement `KNNLibrary` (extends `MethodResolver`):

| Method | Purpose |
|---|---|
| `getVersion()` | Library version for compatibility checks |
| `getExtension()` / `getCompoundExtension()` | File extensions for native index files |
| `score()` | Converts raw library score to Lucene score |
| `distanceToRadialThreshold()` / `scoreToRadialThreshold()` | Radial search conversions |
| `validateMethod()` | Validates a `KNNMethodContext` against the engine |
| `isTrainingRequired()` | Whether the method needs a training step |
| `estimateOverheadInKB()` | Estimates fixed overhead of the index |
| `getKNNLibraryIndexingContext()` | Returns indexing context (validators, processors, library params) |
| `getKNNLibrarySearchContext()` | Returns search-time context for a method |
| `resolveMethod()` | Fills in default parameters for a method |
| `getVectorSearcherFactory()` | Factory for memory-optimized vector searchers |

## KNNMethodContext

Holds the full method configuration parsed from the mapping:

- `knnEngine` — which engine (Faiss, Lucene, etc.)
- `spaceType` — distance function (L2, cosine, inner product, etc.)
- `methodComponentContext` — algorithm name + parameters (e.g., `hnsw` with `m=16, ef_construction=512`)

## EngineResolver

Determines the engine when the user doesn't explicitly set one. Resolution logic:

1. If user set engine in `method` or top-level → use that (validate no conflict)
2. If training required → `FAISS` (only engine supporting training)
3. If method is `flat` → `LUCENE`
4. Based on `mode` and `compressionLevel`:
   - `x4` compression → `LUCENE`
   - Otherwise → `FAISS` (default)

## How Engine Connects to the Mapper

```
Mapping JSON
  → TypeParser.parse()
    → KNNMethodContext.parse()          // engine + space_type + method name + params
    → EngineResolver.resolveEngine()    // pick engine if not explicit
    → engine.resolveMethod()            // fill in default algo params
    → Builder.build()
      → EngineFieldMapper.createFieldMapper()
        → engine.getKNNLibraryIndexingContext()  // get validators, processors, library params
        → library params written into FieldType attributes
          → codec reads these at flush/merge to build native index
```

## Engine Method Registration

Each engine registers its supported methods. For example, Faiss:

```java
Map<String, KNNMethod> METHODS = ImmutableMap.of(
    METHOD_HNSW, new FaissHNSWMethod(),
    METHOD_IVF,  new FaissIVFMethod()
);
```

A new algorithm would be added as a new `KNNMethod` entry in the engine's method map.
