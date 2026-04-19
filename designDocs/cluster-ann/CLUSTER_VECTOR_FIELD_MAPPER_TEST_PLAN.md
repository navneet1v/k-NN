# ClusterVectorFieldMapper — Test Plan

## 1. Unit Tests

### 1.1 TypeParser Validation Tests

**File**: `KNNVectorFieldMapperTests.java` (add to existing file)

These tests verify that `TypeParser.parse()` correctly validates engine-less algorithm mappings.

#### Happy Path

| Test | Input | Expected |
|---|---|---|
| Parse cluster method with space_type | `method: {name: "cluster", space_type: "l2"}` | Builder created, `knnMethodContext.name == "cluster"`, `spaceType == L2`, `isEngineConfigured == false` |
| Parse cluster method with minimal config | `method: {name: "cluster"}, dimension: 128` | Builder created with defaults (space_type defaults via SpaceTypeResolver) |
| Parse cluster method with top-level space_type | `space_type: "l2", method: {name: "cluster"}` | Builder created, space type resolved from top-level |
| Parse cluster method with different data types | `data_type: "byte", method: {name: "cluster"}` | Builder created with `vectorDataType == BYTE` |
| Build produces ClusterVectorFieldMapper | Parse + `builder.build()` | `instanceof ClusterVectorFieldMapper` |

#### Rejection — Engine

| Test | Input | Expected Exception |
|---|---|---|
| Reject engine in method block | `method: {name: "cluster", engine: "faiss"}` | MapperParsingException: "Engine cannot be specified for algorithm 'cluster'" |
| Reject top-level engine | `engine: "faiss", method: {name: "cluster"}` | MapperParsingException: "Engine cannot be specified for algorithm 'cluster'" |
| Reject engine even if set to lucene | `method: {name: "cluster", engine: "lucene"}` | MapperParsingException |

#### Rejection — Model

| Test | Input | Expected Exception |
|---|---|---|
| Reject model_id with cluster | `model_id: "my-model", method: {name: "cluster"}` | Already caught by existing "Method and model can not be both specified" validation |

#### Rejection — Mode and Compression

| Test | Input | Expected Exception |
|---|---|---|
| Reject mode with cluster | `mode: "on_disk", method: {name: "cluster"}` | MapperParsingException: "mode/compression cannot be used with algorithm 'cluster'" |
| Reject compression with cluster | `compression: "16x", method: {name: "cluster"}` | MapperParsingException: "mode/compression cannot be used with algorithm 'cluster'" |
| Reject both mode and compression | `mode: "on_disk", compression: "16x", method: {name: "cluster"}` | MapperParsingException |

#### Rejection — Missing Dimension

| Test | Input | Expected Exception |
|---|---|---|
| Reject missing dimension | `method: {name: "cluster"}` (no dimension) | IllegalArgumentException: "Dimension value missing" |

#### Space Type Validation

| Test | Input | Expected |
|---|---|---|
| Conflicting space types rejected | `space_type: "l2", method: {name: "cluster", space_type: "cosinesimil"}` | MapperParsingException (space type conflict) |
| Method space_type used when top-level absent | `method: {name: "cluster", space_type: "innerproduct"}` | spaceType == INNER_PRODUCT |

### 1.2 Builder.build() Routing Tests

**File**: `KNNVectorFieldMapperTests.java`

These tests verify that `Builder.build()` routes to the correct mapper.

| Test | Setup | Expected Mapper |
|---|---|---|
| Cluster method produces ClusterVectorFieldMapper | `method: {name: "cluster", space_type: "l2"}, dimension: 128` | `instanceof ClusterVectorFieldMapper` |
| HNSW method still produces EngineFieldMapper | `method: {name: "hnsw", engine: "faiss"}, dimension: 128` | `instanceof EngineFieldMapper` |
| No method still produces FlatVectorFieldMapper | `dimension: 128` (KNN disabled) | `instanceof FlatVectorFieldMapper` |
| Model still produces ModelFieldMapper | `model_id: "test"` | `instanceof ModelFieldMapper` |

### 1.3 ClusterVectorFieldMapper Tests

**File**: New `ClusterVectorFieldMapperTests.java`

#### Construction

| Test | Verify |
|---|---|
| createFieldMapper returns non-null | Factory method works |
| fieldType returns KNNVectorFieldType | Correct field type |
| fieldType.typeName == "knn_vector" | Content type correct |
| dimension matches configured value | Dimension propagated |
| useLuceneBasedVectorField is false | No Lucene KNN vector field |
| fieldType docValuesType is BINARY | Binary doc values set |

#### Validators

| Test | Verify |
|---|---|
| getVectorValidator returns SpaceVectorValidator | Not NOOP — validates against space type |
| getPerDimensionValidator for float returns DEFAULT_FLOAT_VALIDATOR | Float validation |
| getPerDimensionValidator for byte returns DEFAULT_BYTE_VALIDATOR | Byte validation |
| getPerDimensionValidator for binary returns DEFAULT_BIT_VALIDATOR | Binary validation |
| getPerDimensionProcessor returns NOOP_PROCESSOR | No processing |

#### parseCreateField — Float Vectors

| Test | Input | Verify |
|---|---|---|
| Valid float vector indexed | float[] of correct dimension | Document contains VectorField with BINARY doc values |
| Null vector skipped | VALUE_NULL token | No fields added |
| Wrong dimension rejected | float[] of wrong dimension | Exception thrown |
| Invalid float value rejected (NaN) | float[] containing NaN | Exception from PerDimensionValidator |

#### parseCreateField — Byte Vectors

| Test | Input | Verify |
|---|---|---|
| Valid byte vector indexed | byte[] of correct dimension | Document contains VectorField with BINARY doc values |
| Out-of-range byte value rejected | value > 127 | Exception from PerDimensionValidator |

#### Space Type Validation via VectorValidator

| Test | Input | Verify |
|---|---|---|
| Cosine space type rejects zero vector | float[] of all zeros, space_type=cosinesimil | Exception from SpaceVectorValidator |
| L2 space type accepts zero vector | float[] of all zeros, space_type=l2 | No exception |

### 1.4 KNNMethodContext Serialization Tests

**File**: `KNNMethodContextTests.java` (add to existing file)

#### toXContent

| Test | Input | Verify |
|---|---|---|
| Engine-configured context writes engine field | `KNNMethodContext(FAISS, L2, ..., isEngineConfigured=true)` | JSON contains `"engine": "faiss"` |
| Engine-not-configured context skips engine field | `KNNMethodContext(UNDEFINED, L2, ..., isEngineConfigured=false)` | JSON does NOT contain `"engine"` |
| Cluster method round-trips through toXContent + parse | Create → toXContent → parse → compare | Original equals parsed |

#### Stream Serialization

| Test | Input | Verify |
|---|---|---|
| Engine-less context round-trips through streams (new version) | `KNNMethodContext(UNDEFINED, L2, ..., isEngineConfigured=false)` with current version stream | `writeTo` → `StreamInput` → equals original, `isEngineConfigured == false` |
| Engine-configured context round-trips through streams (new version) | `KNNMethodContext(FAISS, L2, ..., isEngineConfigured=true)` with current version stream | `writeTo` → `StreamInput` → equals original |
| BWC: old version stream reads engine-less as engine-configured | Write with old version format (always writes engine string) | Reads back with `isEngineConfigured == true` (old behavior preserved) |

### 1.5 Merge Tests

**File**: `KNNVectorFieldMapperTests.java`

| Test | Verify |
|---|---|
| Merge two ClusterVectorFieldMappers with same config | No exception, merged mapper is ClusterVectorFieldMapper |
| Merge ClusterVectorFieldMapper with different space type | Exception — incompatible merge |
| getMergeBuilder returns Builder that rebuilds ClusterVectorFieldMapper | Round-trip through merge builder |

### 1.6 Engine-less Detection Utility Tests

**File**: `KNNVectorFieldMapperTests.java` or new utility test

| Test | Input | Expected |
|---|---|---|
| "cluster" is engine-less | `isEngineLessMethod("cluster")` | true |
| "hnsw" is not engine-less | `isEngineLessMethod("hnsw")` | false |
| "ivf" is not engine-less | `isEngineLessMethod("ivf")` | false |
| null is not engine-less | `isEngineLessMethod(null)` | false |
| unknown name is not engine-less | `isEngineLessMethod("unknown")` | false |

## 2. Integration Tests

### 2.1 Index Creation

**File**: New `ClusterVectorFieldMapperIT.java`

| Test | Action | Verify |
|---|---|---|
| Create index with cluster method | PUT index with cluster mapping | Index created, mapping returned matches input |
| Create index — engine rejected | PUT index with `engine: "faiss"` + `name: "cluster"` | 400 error with clear message |
| Create index — mode rejected | PUT index with `mode: "on_disk"` + `name: "cluster"` | 400 error with clear message |
| Create index — compression rejected | PUT index with `compression: "16x"` + `name: "cluster"` | 400 error with clear message |
| Create index — missing dimension rejected | PUT index with `name: "cluster"` but no dimension | 400 error |

### 2.2 Mapping Retrieval (Round-trip)

| Test | Action | Verify |
|---|---|---|
| Get mapping returns correct structure | Create index → GET mapping | `method.name == "cluster"`, no `engine` field, `space_type` and `parameters` present |
| Get mapping does not contain engine field | Create index → GET mapping | `engine` key absent from method block |

### 2.3 Document Indexing

| Test | Action | Verify |
|---|---|---|
| Index float vector document | PUT doc with float vector | 200 OK, document indexed |
| Index byte vector document | PUT doc with byte vector (data_type: byte) | 200 OK |
| Index wrong dimension rejected | PUT doc with vector of wrong dimension | 400 error |
| Index null vector | PUT doc with null vector field | 200 OK, field absent |

### 2.4 Existing Engine-based Mappings Unaffected

| Test | Action | Verify |
|---|---|---|
| HNSW + Faiss still works | Create index with `method: {name: "hnsw", engine: "faiss"}` | Index created, EngineFieldMapper used |
| HNSW + Lucene still works | Create index with `method: {name: "hnsw", engine: "lucene"}` | Index created |
| Flat mapping still works | Create index with just dimension, KNN disabled | Index created, FlatVectorFieldMapper used |

## 3. Test Coverage Summary

| Area | UT Count | IT Count |
|---|---|---|
| TypeParser validation (happy path) | 5 | 1 |
| TypeParser validation (rejection) | 8 | 4 |
| Builder routing | 4 | — |
| ClusterVectorFieldMapper construction | 6 | — |
| ClusterVectorFieldMapper validators | 5 | — |
| ClusterVectorFieldMapper parseCreateField | 6 | 4 |
| KNNMethodContext serialization | 6 | 2 |
| Merge | 3 | — |
| Engine-less detection utility | 5 | — |
| Backward compatibility (existing paths) | — | 3 |
| **Total** | **48** | **14** |

## 4. Follow-up: Tests for method.parameters

Once `method.parameters` for the cluster algorithm are designed (e.g., `num_clusters`, `sample_size`, `encoder`), the following test areas will be added:

- UT: Parse cluster method with valid parameters
- UT: Reject unknown parameters
- UT: Validate parameter types and ranges (e.g., `num_clusters` must be positive integer)
- UT: Default values applied when parameters omitted
- IT: Create index with parameters, verify in GET mapping response
- IT: Reject invalid parameter values at index creation
