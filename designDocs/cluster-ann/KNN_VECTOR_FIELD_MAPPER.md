# KNNVectorFieldMapper

### Class Hierarchy

```
ParametrizedFieldMapper (OpenSearch core)
  └── KNNVectorFieldMapper (abstract base)
        ├── FlatVectorFieldMapper    — stores vectors as doc values only (no KNN index structure)
        ├── EngineFieldMapper        — builds KNN index using an engine (Lucene, Faiss, NMSLIB)
        └── ModelFieldMapper         — uses a pre-trained model to define the index structure
```

### Key Responsibilities

#### TypeParser

Parses the mapping JSON (`"type": "knn_vector"`) and decides which concrete mapper to create:

- `modelId` set → `ModelFieldMapper`
- KNN disabled or no resolved method (≥2.17) → `FlatVectorFieldMapper`
- Otherwise → `EngineFieldMapper`

#### Builder

Holds all mapping parameters: `dimension`, `vectorDataType`, `knnMethodContext`, `modelId`, `mode`, `compressionLevel`, `topLevelSpaceType`, `topLevelEngine`. The `build()` method routes to the correct mapper.

#### Parsing & Indexing (`parseCreateField`)

Reads vectors from the document, validates them (per-dimension + whole-vector), applies transformations, then creates Lucene `Field` objects:

- Lucene engine → `KnnFloatVectorField` / `KnnByteVectorField`
- Faiss/NMSLIB → `VectorField` with binary doc values

#### Extension Points

Each concrete mapper provides:

| Method | Purpose |
|---|---|
| `getVectorValidator()` | Validates the whole vector (e.g., space type constraints) |
| `getPerDimensionValidator()` | Validates each dimension value |
| `getPerDimensionProcessor()` | Processes each dimension (e.g., clipping) |
| `getVectorTransformer()` | Transforms the vector before indexing (e.g., normalization) |


