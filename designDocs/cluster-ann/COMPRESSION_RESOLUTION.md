# Compression Resolution for Engine-Less Algorithms

## 1. Problem Statement

The cluster ANN codec (`ClusterANN1040KnnVectorsFormat`) needs a `docBits` value (1, 2, or 4) to configure scalar quantization. This value can originate from two different user inputs:

**Path 1: Top-level `compression_level`**
```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "compression_level": "32x",
    "method": { "name": "cluster", "space_type": "l2" }
  }
}
```

**Path 2: Encoder in `method.parameters`**
```json
{
  "my_vector": {
    "type": "knn_vector",
    "dimension": 128,
    "method": {
      "name": "cluster",
      "space_type": "l2",
      "parameters": {
        "encoder": {
          "name": "sq",
          "parameters": { "bits": 4 }
        }
      }
    }
  }
}
```

**Path 3: Both specified**
```json
{
  "compression_level": "32x",
  "method": {
    "name": "cluster",
    "parameters": {
      "encoder": { "name": "sq", "parameters": { "bits": 4 } }
    }
  }
}
```
This is a conflict — `32x` implies 1-bit but encoder says 4-bit. Must be validated and rejected.

**Path 4: Neither specified**
```json
{
  "method": { "name": "cluster", "space_type": "l2" }
}
```
Use default `docBits`.

### The Two Inputs

| Input | Where it lives | What it means |
|---|---|---|
| `compression_level` | Top-level mapping parameter → `KNNMethodConfigContext.compressionLevel` → `KNNMappingConfig.getCompressionLevel()` | High-level hint: "compress my vectors 32x" |
| `encoder` | `KNNMethodContext.methodComponentContext.parameters["encoder"]` | Explicit encoder config: "use SQ with 4 bits" |

### What the Codec Needs

A single `docBits` value (1, 2, or 4) passed to `ClusterANN1040KnnVectorsFormat(docBits)`.

### The Resolution Problem

Someone needs to:
1. Read both inputs
2. If only one is set, derive `docBits` from it
3. If both are set, validate they agree or reject
4. If neither is set, use default
5. Make the resolved `docBits` available to the codec format resolver

For engine-based fields, this is handled by the engine's `MethodResolver`. For engine-less fields, there is no engine. Who does this?

## 2. How Engine-Based Resolution Works

### The Players

| Component | What it does |
|---|---|
| `KNNMethodConfigContext` | Carries `compressionLevel` from the top-level mapping parameter. Transient — only lives during mapping parsing. |
| `KNNMethodContext` | Carries `methodComponentContext.parameters` including encoder. Persisted in index metadata. |
| `EngineResolver` | Picks the engine based on mode + compression. |
| `KNNEngine.resolveMethod()` | Delegates to engine-specific `MethodResolver`. |
| `FaissMethodResolver` | The actual resolver for FAISS. |
| `ResolvedMethodContext` | Output: resolved `KNNMethodContext` + final `CompressionLevel`. |

### The Flow

```
TypeParser.parse()
    │
    ├── Sets KNNMethodConfigContext.compressionLevel from top-level param
    │
    ├── EngineResolver.resolveEngine(configContext, methodContext)
    │   └── Picks engine (e.g., FAISS) based on mode + compression
    │
    └── KNNEngine.resolveMethod(methodContext, configContext)
        └── FaissMethodResolver.resolveMethod()
            │
            ├── resolveEncoder()
            │   │
            │   ├── Reads configContext.compressionLevel (Path 1)
            │   ├── Reads methodContext.parameters.encoder (Path 2)
            │   │
            │   ├── If only compression set:
            │   │   └── Maps compression → encoder + params
            │   │       (e.g., x32 → QFrameBitEncoder with bits=1)
            │   │
            │   ├── If only encoder set:
            │   │   └── Calculates compression from encoder
            │   │       (e.g., SQ bits=4 → x8)
            │   │
            │   ├── If both set:
            │   │   └── Validates they agree, rejects if conflict
            │   │
            │   └── If neither set:
            │       └── Uses default (flat encoder, x1 compression)
            │
            ├── Writes resolved encoder INTO methodContext.parameters
            │   (e.g., sets encoder name, bits, type params)
            │
            └── Returns ResolvedMethodContext(methodContext, compressionLevel)
                │
                └── TypeParser sets configContext.compressionLevel = resolved
```

### Key Observations

1. **`FaissMethodResolver.resolveEncoder()` is the single reconciliation point** — it sees both `compressionLevel` (from `configContext`) and `encoder` (from `methodContext.parameters`), reconciles them, and produces a single resolved state.

2. **The resolved encoder is written back into `KNNMethodContext.parameters`** — this is how the codec sees it. The codec never reads `compressionLevel` directly; it reads the encoder config from the persisted method context.

3. **`compressionLevel` on `KNNMethodConfigContext` is also updated** — so both the transient config and the persisted method context agree after resolution.

4. **The resolver is engine-specific** — FAISS maps compression to QFrameBitEncoder/SQ, Lucene maps to its own SQ format. Each engine knows its own encoder vocabulary.

### What Gets Persisted

After resolution, `KNNMethodContext` (persisted in index metadata) contains:
```
methodComponentContext:
  name: "hnsw"
  parameters:
    m: 16
    ef_construction: 512
    encoder:
      name: "qframe_bit"
      parameters:
        bits: 1
```

The codec reads this at segment write/read time. `compressionLevel` is NOT persisted — it's only on the transient `KNNMethodConfigContext`.

## 3. The Gap for Engine-Less Algorithms

For cluster ANN:
- There is no `KNNEngine` → no `resolveMethod()` call
- There is no `FaissMethodResolver` → no `resolveEncoder()`
- There is no engine-specific encoder vocabulary (no `QFrameBitEncoder`, no `FaissSQEncoder`)
- `validateFromEngineLessAlgorithm()` currently sets `compressionLevel` on `KNNMethodConfigContext` but does NOT reconcile it with encoder params
- The codec format resolver (`EngineLessCodecFormatResolver`) receives `KNNMethodContext` and `params` but not `compressionLevel`

### What Needs to Be Designed

1. **Who reconciles** `compression_level` and `encoder` for engine-less algorithms?
2. **Where does the resolved `docBits` live** so the codec can read it?
3. **What encoder vocabulary** does cluster ANN use? (It has its own: 1-bit, 2-bit, 4-bit scalar quantization — not FAISS encoders)
4. **How does the codec read it** at segment write/read time?

## 4. Option 1: ClusterANNMethodResolver extending AbstractMethodResolver

Create a `ClusterANNMethodResolver` that follows the same pattern as `FaissMethodResolver` and `LuceneHNSWMethodResolver`. The key insight: `AbstractMethodResolver` and `MethodResolver` are not engine-specific — they're method resolution abstractions. Engine-based methods happen to be resolved by engines, but the resolution pattern (reconcile compression + encoder → resolved config) applies equally to engine-less algorithms.

### What ClusterANNMethodResolver Does

1. Reads `compressionLevel` from `KNNMethodConfigContext` (Path 1)
2. Reads `encoder` from `KNNMethodContext.methodComponentContext.parameters` (Path 2)
3. If encoder is specified, uses cluster ANN's own `Encoder` implementations to calculate compression level
4. Calls `validateCompressionConflicts()` (inherited from `AbstractMethodResolver`) to reject mismatches
5. If only compression is set, resolves it to encoder params (compression → docBits → encoder config)
6. If only encoder is set, derives compression from it
7. Returns `ResolvedMethodContext` with the reconciled result

### Cluster ANN Encoder Vocabulary

Cluster ANN has its own encoder, not FAISS's:

| Encoder name | Parameter | docBits | CompressionLevel |
|---|---|---|---|
| `sq` | `bits: 4` | 4 | x8 |
| `sq` | `bits: 2` | 2 | x16 |
| `sq` | `bits: 1` | 1 | x32 |

This maps to an `Encoder` implementation (e.g., `ClusterANNSQEncoder`) that implements `calculateCompressionLevel()`.

### Where It's Called

In `validateFromEngineLessAlgorithm()`, after basic validation, call the algorithm's method resolver:

```java
// In validateFromEngineLessAlgorithm():
EngineLessMethod method = EngineLessMethod.fromName(methodName);
ResolvedMethodContext resolvedMethodContext = method.getMethodResolver()
    .resolveMethod(
        builder.originalParameters.getKnnMethodContext(),
        builder.knnMethodConfigContext,
        false,  // shouldRequireTraining
        resolvedSpaceType
    );
builder.originalParameters.setResolvedKnnMethodContext(resolvedMethodContext.getKnnMethodContext());
builder.knnMethodConfigContext.setCompressionLevel(resolvedMethodContext.getCompressionLevel());
```

### How EngineLessMethod Provides the Resolver

```java
public interface EngineLessMethod {
    String getName();
    EngineLessMapperFactory getMapperFactory();
    KnnVectorsFormat createFormat(int docBits);
    MethodResolver getMethodResolver();
}

public class ClusterANNMethod implements EngineLessMethod {
    public static final ClusterANNMethod INSTANCE = new ClusterANNMethod();
    public MethodResolver getMethodResolver() { return new ClusterANNMethodResolver(); }
    // ...
}
```

### Flow

```
validateFromEngineLessAlgorithm()
    │
    ├── Basic validation (reject engine, mode, non-float)
    │
    ├── Set compressionLevel on KNNMethodConfigContext
    │
    └── ClusterANNMethodResolver.resolveMethod()
        │
        ├── resolveEncoder()
        │   ├── Reads configContext.compressionLevel (Path 1)
        │   ├── Reads methodContext.parameters.encoder (Path 2)
        │   ├── Reconciles using ClusterANNSQEncoder.calculateCompressionLevel()
        │   ├── validateCompressionConflicts() (inherited)
        │   └── Writes resolved encoder into methodContext.parameters
        │
        ├── resolveMethodParams() (inherited from AbstractMethodResolver)
        │
        └── Returns ResolvedMethodContext(methodContext, compressionLevel)
            │
            └── Caller sets configContext.compressionLevel = resolved
```

### What Changes

| Component | Change |
|---|---|
| `AbstractMethodResolver` | Update javadocs — not engine-specific, used by any algorithm |
| `MethodResolver` | Update javadocs — same |
| `ClusterANNMethodResolver` | New class extending `AbstractMethodResolver` |
| `ClusterANNSQEncoder` | New `Encoder` implementation for cluster ANN's SQ |
| `EngineLessMethod` | Convert from enum to interface |
| `validateFromEngineLessAlgorithm()` | Call `method.getMethodResolver().resolveMethod()` |
| `EngineLessCodecFormatResolver` | Read resolved encoder from `KNNMethodContext.parameters` |

### Why This Works

- **Same reconciliation pattern** as engine-based path — `AbstractMethodResolver.validateCompressionConflicts()` handles conflict detection
- **Each algorithm owns its encoder vocabulary** — cluster ANN defines `ClusterANNSQEncoder`, future algorithms define theirs
- **Resolved encoder is persisted in `KNNMethodContext`** — codec reads it the same way FAISS codec reads FAISS encoders
- **No special-casing in the codec resolver** — it just reads params like any other method

### Answers to the Four Questions

1. **Who reconciles?** `ClusterANNMethodResolver`, extending `AbstractMethodResolver` — same pattern as `FaissMethodResolver`
2. **Where does resolved docBits live?** In `KNNMethodContext.methodComponentContext.parameters` as encoder config — persisted, readable by codec
3. **What encoder vocabulary?** `ClusterANNSQEncoder` with `bits` parameter (1, 2, 4) — cluster ANN's own, not FAISS's
4. **How does the codec read it?** `EngineLessCodecFormatResolver` reads encoder from `params`, derives `docBits` from `bits` parameter
