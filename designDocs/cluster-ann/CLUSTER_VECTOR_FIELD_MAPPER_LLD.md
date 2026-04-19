# ClusterANNVectorFieldMapper — Low Level Design

## 1. Overview

This document covers the mapper layer changes needed to integrate the cluster-based ANN algorithm. It does not cover codec or query path.

### Files to Modify

| File | Change |
|---|---|
| `KNNConstants.java` | Add `METHOD_CLUSTER` constant |
| `KNNVectorFieldMapper.java` (TypeParser) | Add engine-less algorithm detection and validation |
| `KNNVectorFieldMapper.java` (Builder) | Add registry-based routing for engine-less mappers |
| `KNNMethodContext.java` | Skip engine in toXContent/stream serialization when not configured |

### Files to Create

| File | Purpose |
|---|---|
| `EngineLessMethod.java` | Enum of engine-less algorithm names |
| `EngineLessMapperFactory.java` | Factory interface for creating engine-less mappers |
| `ClusterANNVectorFieldMapper.java` | New mapper for cluster algorithm |

## 2. Class Diagram — Mapper Hierarchy

```mermaid
classDiagram
    direction TB

    class ParametrizedFieldMapper {
        <<abstract>>
        <<OpenSearch Core>>
    }

    class KNNVectorFieldMapper {
        <<abstract>>
        #useLuceneBasedVectorField : boolean
        #originalMappingParameters : OriginalMappingParameters
        +parseCreateField(context) void
        +getVectorValidator()* VectorValidator
        +getPerDimensionValidator()* PerDimensionValidator
        +getPerDimensionProcessor()* PerDimensionProcessor
    }

    class FlatVectorFieldMapper {
        +createFieldMapper(...)$ FlatVectorFieldMapper
    }

    class EngineFieldMapper {
        -isLuceneEngine : boolean
        +createFieldMapper(...)$ EngineFieldMapper
    }

    class ModelFieldMapper {
        -modelId : String
        +createFieldMapper(...)$ ModelFieldMapper
    }

    class ClusterANNVectorFieldMapper {
        <<new>>
        -vectorValidator : VectorValidator
        +createFieldMapper(...)$ ClusterANNVectorFieldMapper
    }

    class EngineLessMethod {
        <<new>>
        <<enum>>
        CLUSTER
        +fromName(name)$ EngineLessMethod
        +isEngineLess(name)$ boolean
    }

    class EngineLessMapperFactory {
        <<new>>
        <<interface>>
        +create(...) KNNVectorFieldMapper
    }

    ParametrizedFieldMapper <|-- KNNVectorFieldMapper
    KNNVectorFieldMapper <|-- FlatVectorFieldMapper
    KNNVectorFieldMapper <|-- EngineFieldMapper
    KNNVectorFieldMapper <|-- ModelFieldMapper
    KNNVectorFieldMapper <|-- ClusterANNVectorFieldMapper
    KNNVectorFieldMapper --> EngineLessMapperFactory : ENGINE_LESS_MAPPER_FACTORIES
    EngineLessMapperFactory ..> ClusterANNVectorFieldMapper : creates

    note for FlatVectorFieldMapper "Engine-less\nNo search structure\nBinary doc values only"
    note for EngineFieldMapper "Engine-bound\nFaiss / Lucene / NMSLIB"
    note for ClusterANNVectorFieldMapper "Engine-less\nLucene KnnFloatVectorField\nFloat only\nSpace type aware"
```

### Legend

| Style | Meaning |
|---|---|
| `<<abstract>>` | Abstract class |
| `<<OpenSearch Core>>` | Class from OpenSearch core |
| `<<new>>` | New class/interface to be created |
| `<<enum>>` | Enum type |
| `$` suffix | Static method |
| `*` suffix | Abstract method |

## 3. Class Diagram — Validation & Parsing Flow

```mermaid
classDiagram
    direction LR

    class TypeParser {
        +parse(name, node, parserContext) Builder
        -validateFromEngineLessAlgorithm(builder) void
        -validateFromFlat(builder) void
        -validateFromModel(builder) void
        -validateFromKNNMethod(builder) void
    }

    class Builder {
        +build(context) KNNVectorFieldMapper
    }

    class EngineLessMethod {
        <<enum>>
        CLUSTER
        +fromName(name)$ EngineLessMethod
        +isEngineLess(name)$ boolean
    }

    class EngineLessMapperFactory {
        <<interface>>
        +create(...) KNNVectorFieldMapper
    }

    TypeParser --> Builder : creates & validates
    TypeParser --> EngineLessMethod : checks method name
    Builder --> EngineLessMapperFactory : registry lookup
    TypeParser ..> EngineResolver : engine path only
    TypeParser ..> SpaceTypeResolver : all paths

    note for TypeParser "validateFromEngineLessAlgorithm:\n- Rejects engine\n- Rejects mode/compression\n- Rejects non-float data types\n- Requires dimension\n- Resolves space type"
```

## 4. Sequence Diagram — Mapping Parse & Mapper Creation

```mermaid
sequenceDiagram
    participant User as Mapping JSON
    participant TP as TypeParser.parse()
    participant ELM as EngineLessMethod
    participant B as Builder
    participant SR as SpaceTypeResolver
    participant REG as ENGINE_LESS_MAPPER_FACTORIES
    participant ER as EngineResolver

    User->>TP: {"method": {"name": "cluster", "space_type": "l2"}}
    TP->>B: parse(name, parserContext, node)
    TP->>TP: Check: method + model mutual exclusion

    TP->>ELM: isEngineLess("cluster")
    ELM-->>TP: true

    TP->>TP: validateFromEngineLessAlgorithm(builder)
    TP->>TP: Reject if engine configured
    TP->>TP: Reject if mode/compression set
    TP->>TP: Reject if non-float data type
    TP->>TP: Require dimension
    TP->>SR: resolveSpaceType(...)
    SR-->>TP: L2
    TP-->>B: return builder (skip engine resolution)

    Note over B: Builder.build() called later

    B->>ELM: fromName("cluster")
    ELM-->>B: CLUSTER
    B->>REG: get(CLUSTER).create(...)
    REG-->>B: ClusterANNVectorFieldMapper
```

## 5. Class Diagram — Validators & Processors

```mermaid
classDiagram
    direction TB

    class VectorValidator {
        <<interface>>
    }

    class PerDimensionValidator {
        <<interface>>
    }

    class PerDimensionProcessor {
        <<interface>>
    }

    class SpaceVectorValidator {
        -spaceType : SpaceType
    }

    class NOOP_VECTOR_VALIDATOR {
        <<singleton>>
    }

    VectorValidator <|.. SpaceVectorValidator
    VectorValidator <|.. NOOP_VECTOR_VALIDATOR

    FlatVectorFieldMapper ..> NOOP_VECTOR_VALIDATOR : uses
    FlatVectorFieldMapper ..> NOOP_PROCESSOR : uses

    ClusterANNVectorFieldMapper ..> SpaceVectorValidator : uses
    ClusterANNVectorFieldMapper ..> DEFAULT_FLOAT_VALIDATOR : uses (float only)
    ClusterANNVectorFieldMapper ..> NOOP_PROCESSOR : uses

    note for ClusterANNVectorFieldMapper "Float only\nSpaceVectorValidator for\ncosine zero-vector rejection etc."
```

## 6. Builder.build() Routing — Engine-less Mapper Resolution

### Alternatives Considered

#### Alternative A: Hardcoded if-else per algorithm (rejected)

Each engine-less algorithm gets its own `if` block in `Builder.build()`.

**Rejected because:** Every new algorithm adds another branch. Violates open-closed principle.

#### Alternative B: Single shared EngineLessFieldMapper for all algorithms (rejected)

One mapper class for all engine-less algorithms. Differentiation in codec only.

**Rejected because:** If a future algorithm needs different ingestion behavior (different validators, data types, field attributes), this forces conditionals inside a single mapper.

#### Alternative C: Registry of EngineLessMethod → mapper factory (chosen)

A static `Map<EngineLessMethod, EngineLessMapperFactory>`. `Builder.build()` does a single lookup.

**Chosen because:**
1. `Builder.build()` has one branch that never grows
2. Each algorithm owns its own mapper class
3. Factory signature is naturally shared
4. New algorithms: add enum value + register in map

### Routing Decision Tree

```mermaid
flowchart TD
    A[Builder.build called] --> B{modelId set?}
    B -->|Yes| C[ModelFieldMapper]
    B -->|No| D{method.name in\nENGINE_LESS_MAPPER_FACTORIES?}
    D -->|Yes| E[Registry lookup → algorithm-specific mapper]
    D -->|No| F{resolvedKnnMethodContext == null\nAND version >= 2.17?}
    F -->|Yes| G[FlatVectorFieldMapper]
    F -->|No| H[EngineFieldMapper]

    style E fill:#4da6ff,color:#000
    style C fill:#f0f0f0,color:#000
    style G fill:#f0f0f0,color:#000
    style H fill:#f0f0f0,color:#000
```

## 7. ClusterANNVectorFieldMapper — Key Design Decisions

### Vector Storage: Lucene KnnFloatVectorField

Uses `useLuceneBasedVectorField = true` with Lucene's native `KnnFloatVectorField`. Vectors are stored in Lucene's vector format. The codec layer will handle cluster index building separately.

### FieldType Attributes

The `FieldType` carries metadata for the codec to identify and configure cluster fields:

| Attribute | Value | Purpose |
|---|---|---|
| `knn_method` | `"cluster"` | Codec identifies this as a cluster algorithm field |
| `space_type` | e.g., `"l2"` | Codec/query uses for distance computation |
| `dimension` | e.g., `"128"` | Codec uses for vector layout |
| `data_type` | `"float"` | Always float — only supported type |

Plus Lucene's vector attributes (dimension, `FLOAT32` encoding, similarity function) set via `fieldType.setVectorAttributes()`.

### Data Type: Float Only

Only `VectorDataType.FLOAT` is supported. Byte and binary are rejected during validation in `validateFromEngineLessAlgorithm`. The mapper hardcodes `DEFAULT_FLOAT_VALIDATOR` and `VectorEncoding.FLOAT32`.

### Validation: SpaceVectorValidator

Uses `SpaceVectorValidator` (not NOOP) to validate vectors against the space type at ingestion time (e.g., cosine rejects zero-magnitude vectors).

## 8. XContent & Stream Serialization Changes

### Parsing (JSON → Java) — No Changes

`KNNMethodContext.parse()` handles the cluster mapping without modification.

### toXContent (Java → JSON) — Skip engine when not configured

Only write `engine` field when `isEngineConfigured == true`. Cluster mapping serializes without `engine` field.

### Stream Serialization — Version-gated at V_3_7_0

New format writes `isEngineConfigured` boolean before engine string. Old format preserved for BWC.

| Concern | Change | BWC Impact |
|---|---|---|
| Parsing (JSON → Java) | None | N/A |
| toXContent (Java → JSON) | Skip `engine` when `!isEngineConfigured` | None |
| Stream writeTo/readFrom | Version-gated boolean flag | Old nodes see old format |

## 9. Differences from FlatVectorFieldMapper

| Aspect | FlatVectorFieldMapper | ClusterANNVectorFieldMapper |
|---|---|---|
| Purpose | Store vectors only (no search structure) | Store vectors + cluster index via codec |
| Vector storage | Binary doc values (`DocValuesType.BINARY`) | Lucene `KnnFloatVectorField` |
| `useLuceneBasedVectorField` | `false` | `true` |
| Data types | Float, byte, binary | Float only |
| VectorValidator | NOOP | SpaceVectorValidator |
| FieldType attributes | None | `knn_method`, `space_type`, `dimension`, `data_type` |
| Space type awareness | No | Yes |

## 10. What This Does NOT Cover

- **Codec layer**: How the cluster index is built during segment flush/merge
- **Query layer**: How cluster-based search is executed
- **Algorithm parameters**: `method.parameters` (e.g., `num_clusters`, `sample_size`) — follow-up
- **Encoder support**: Future quantization via `encoder` in `method.parameters` — follow-up
