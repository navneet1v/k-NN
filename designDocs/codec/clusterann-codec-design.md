# ClusterANN Codec Design Document

## Overview

ClusterANN is an IVF (Inverted File Index) based approximate nearest neighbor search implementation integrated into the OpenSearch k-NN plugin via Lucene's codec extension points. It implements a two-phase search: coarse quantization (centroid selection) followed by fine-grained asymmetric distance computation (ADC) using scalar quantization.

The design is inspired by Google's ScaNN (Scalable Nearest Neighbors) and FAISS, optimized for Lucene's immutable segment model.

---

## File Format

ClusterANN uses two per-segment files:

| File | Extension | Purpose |
|------|-----------|---------|
| Metadata | `.clam` | Centroids, centroid norms, posting sizes, offset table, random rotation, transformed centroids |
| Postings | `.clap` | Per-centroid posting lists + block-columnar quantized vectors |

### Codec Identity

```
CODEC_NAME      = "ClusterANN1040"
VERSION_START   = 0
VERSION_CURRENT = 0
```

Both files start with a standard Lucene `IndexHeader` and end with a `CodecFooter`.

---

## .clam Layout (Metadata)

Per-field, sequentially until `END_OF_FIELDS (-1)`:

```
fieldNumber:          int
numVectors:           int
dimension:            int
numCentroids:         int
metricName:           String (L2 | INNER_PRODUCT | COSINE)
docBits:              byte   (1, 2, or 4)
postingsOffset:       long   (file pointer into .clap)

centroidDocCounts:    numCentroids x int      (vectors per cluster)
centroidNorms:        numCentroids x float    (||c||^2 for decomposed L2)
postingSizes:         numCentroids x int      (exact bytes per centroid for prefetch)

centroids:            numCentroids x dimension x float
centroidOffsets:      numCentroids x long     (file pointer per centroid in .clap)
randomRotation:       serialized RandomRotation matrix
transformedCentroids: numCentroids x dimension x float (post-rotation, for ADC)
```

### Design Rationale

- **centroidNorms** enable the ScaNN decomposed L2 optimization: `||q-c||^2 = ||q||^2 + ||c||^2 - 2*dot(q,c)`, avoiding per-dimension subtraction.
- **postingSizes** allow exact byte-level prefetch of upcoming centroid data during search.
- **transformedCentroids** are pre-rotated so that query-time ADC uses the same vector space as indexed codes.
- **Eager loading**: centroidDocCounts, norms, and postingSizes are small (per-centroid) and loaded upfront. Centroids and offset tables are larger but also loaded eagerly since they're needed for every query.

---

## .clap Layout (Postings)

Aligned to 16-byte boundaries (SIMD register width). Per centroid, the primary posting list and SOAR posting list are stored adjacently for sequential I/O:

```
[Primary Posting List]
[SOAR Posting List]
```

### Per Posting List

```
docIds:     PostingListCodec encoded (adaptive: CONTINUOUS | DELTA_FIXED16 | PACKED_32)
ordinals:   VInt count + count x int (fixed-width bulk readable)
quantized:  block-columnar layout (BLOCK_SIZE=32)
```

### Block-Columnar Quantized Layout

Each block of 32 vectors is stored corrections-first, codes-last:

```
lower[0..31]:   32 x int (float bits) -- correction lower intervals
upper[0..31]:   32 x int (float bits) -- correction upper intervals
add[0..31]:     32 x int (float bits) -- additional corrections
sum[0..31]:     32 x int              -- quantized component sums
codes[0..31]:   32 x packedBytes      -- quantized vector codes (contiguous for SIMD)
```

**Why corrections first**: Enables block-level early skip. The reader computes an upper bound from corrections alone; if the block cannot beat the current competitive threshold, the codes are skipped entirely without reading them from disk.

---

## Scalar Bit Encoding

Three quantization levels supported via `ScalarBitEncoding`:

| Encoding | Bits | Compression | Packed Layout |
|----------|------|-------------|---------------|
| ONE_BIT  | 1    | 32x         | MSB-first binary packing (8 values/byte) |
| TWO_BIT  | 2    | 16x         | 2 transposed stripes (lower bits, upper bits) |
| FOUR_BIT | 4    | 8x          | 4 nibble stripes (SIMD-friendly) |

All encodings use 4-bit query quantization (asymmetric: query has more precision than documents).

The transposed stripe layout enables SIMD-friendly dot products: each stripe is a contiguous byte array processed via `bitCount`-based operations without unpacking individual values.

---

## Posting List Codec

`PostingListCodec` adaptively selects the tightest encoding per posting list:

| Encoding | Condition | Storage |
|----------|-----------|---------|
| CONTINUOUS | IDs are sequential | 1 VInt (start) |
| DELTA_FIXED16 | All deltas fit in 16 bits | 1 int (start) + (n-1) shorts |
| PACKED_32 | General case | n ints (bulk readable) |

---

## Write Path

### Flush (`ClusterANN1040KnnVectorsWriter.flush`)

1. Collect vectors from `FlatFieldVectorsWriter` (on-heap `List<float[]>`)
2. Wrap as `ClusterANNVectorValues.fromList()`
3. Call `writeIVF()` with `initialCentroids = null`

### Merge (`ClusterANN1040KnnVectorsWriter.mergeOneField`)

1. Read all vectors from source segments via `ClusterANNVectorValues.fromMergeState()`
   - Under 10M vectors: ordinal mapping to source segment readers (no temp file)
   - Over 10M vectors: writes to temp file, reads back via mmap
   - Reservoir-samples 4096 vectors during the read pass
2. Estimate centroid count: `max(2, min(4096, (n + 256) / 512))`
3. Use reservoir sample as `initialCentroids` to seed k-means (faster convergence)
4. Call `writeIVF()` -- **full rebuild** of the IVF index

### writeIVF (Shared Path)

```
1. Cluster: HierarchicalKMeans.cluster(vectors, config, initialCentroids)
2. Create RandomRotation matrix
3. Transform centroids into quantization space
4. Spatial sort centroids (project onto axis of max variance, sort by projection)
5. Write .clap: for each centroid in spatial order:
   a. Write primary posting list (docIds + ordinals + quantized blocks)
   b. Write SOAR posting list (adjacent)
6. Write .clam: field metadata + stats + centroids + offsets + rotation + transformed centroids
```

### Clustering Algorithm

`HierarchicalKMeans` implements adaptive hierarchical k-means:

- Top-level: flat k-means with `k = n / targetSize` (capped at 128 per level)
- Oversized clusters (>1.5x target) are recursively split
- Uses reservoir-sampling initialization (fast, no k-means++ overhead)
- Merge path seeds with reservoir-sampled centroids (skip init entirely)
- Max depth: 10 levels, max total centroids: 4096

### SOAR (Secondary Orthogonal Augmented Retrieval)

Each vector gets a secondary cluster assignment based on residual-aware distance:

```
SOAR_dist(v, c_secondary) = L2(v, c_secondary) + lambda * (residual_projection^2 / residual_norm^2)
```

- Looks at top-10 nearest centroids to the primary centroid
- Selects the secondary centroid that minimizes SOAR distance
- Secondary assignments create overlapping posting lists, improving recall
- Native SIMD acceleration via `SimdVectorComputeService.bulkSOARDistance()`

---

## Read Path

### Initialization (`ClusterANN1040KnnVectorsReader`)

1. Open `.clam` and `.clap` files
2. Read all field states from `.clam` (eager: centroids + offsets + rotation + transformed centroids)
3. Build field name to number mapping

### Search Pipeline

```
ClusterANN1040KnnVectorsReader.search()
    |
    v
[Brute force fallback if numVectors < 100]
    |
    v
NearestProbeScheduler (centroid selection)
    |
    v
OptimizedProbeScheduler (I/O optimization + early termination)
    |
    v
ClusterANNCentroidScanner (per-centroid scan)
    |
    v
QuantizedVectorReader.scoreBlock() (ADC scoring per block of 32)
    |
    v
KnnCollector (direct collection)
```

### Phase 1: Centroid Selection (`NearestProbeScheduler`)

1. Compute distances from query to all centroids:
   - L2: decomposed via `||q||^2 + ||c||^2 - 2*dot(q,c)` (centroid norms precomputed)
   - IP/Cosine: direct metric distance
2. Sort centroids by distance using primitive `long[]` packing (no boxing)
3. Compute adaptive nprobe: `max(10, NPROBE_MULTIPLIER * sqrt(numCentroids))`
4. Build `ProbeTarget[]` array with centroid index, file offset, posting size, distance

### Phase 2: I/O Optimization (`OptimizedProbeScheduler`)

1. **Window reordering**: Reorder probes by file offset within sliding windows of 8 for sequential I/O
2. **Read-ahead prefetch**: Issue `IndexInput.prefetch()` for upcoming probes (lookahead = 8), skip postings > 2MB (would thrash L2 cache)
3. **Filter-aware skip**: If filter selectivity < 10%, skip centroids where expected valid docs < 0.5
4. **Hybrid early termination**:
   - Soft budget = `max(k*4, numVectors * nprobe / (2 * numCentroids))`
   - Contribution tracking: if threshold isn't improving AND past budget → stop
   - Distance ratio: stop if current centroid is >1.5x farther than closest

### Phase 3: Per-Centroid Scan (`ClusterANNCentroidScanner`)

For each centroid, scans both primary and SOAR posting lists:

1. Read posting list (docIds via `PostingListCodec`, ordinals via bulk `readInts`)
2. Filter: skip visited docs (BitSet), skip deleted docs (Bits)
3. Route to ADC or exact scoring:
   - **ADC path**: `QuantizedVectorReader.scoreBlock()` in blocks of 32
   - **Exact path**: skip quantized bytes, batch `bulkScore()` via `RandomVectorScorer`

### Phase 4: ADC Block Scoring (`QuantizedVectorReader`)

Per block of 32 vectors:

1. **Query quantization** (cached per centroid): Quantize query to 4-bit using `OptimizedScalarQuantizer`, transpose into stripe layout
2. **Read corrections** (lower, upper, add, sum)
3. **Block-level early skip**: Compute upper bound from corrections alone; if block can't beat threshold, skip code bytes entirely
4. **Read codes** (only if potentially competitive)
5. **Bulk dot product**:
   - Native: `SimdVectorComputeService.bulkQuantizedDotProduct()` (SIMD)
   - Java fallback: `VarHandle`-based long reads for 1-bit, byte-level for 2-bit/4-bit
6. **Score computation**: Apply corrections to raw dot products, convert to similarity:
   - Euclidean: `1 / (1 + max(corrected_score, 0))`
   - MIP: `rawDot >= 0 ? rawDot + 1 : 1 / (1 - rawDot)`
   - Cosine/DP: `max((1 + rawDot) / 2, 0)`
7. **Direct collection** into `KnnCollector` (no intermediate buffer)

### Candidate Collection

`CandidateCollector` (used in earlier codepaths, now superseded by direct collection):
- Amortized O(1) insertion inspired by ScaNN's `TopNAmortizedConstant`
- Buffer of 2x capacity; when full, quickselect-partition to keep top half
- No heap data structure (avoids O(log n) per insert)

---

## Random Rotation

`RandomRotation` applies a random orthogonal transformation to vectors before quantization (L2 only). This redistributes variance across dimensions, improving scalar quantization quality when the original data has high variance concentration in a few dimensions.

- Created at index time, stored in `.clam`
- Applied to both centroids (stored as `transformedCentroids`) and vectors (during posting list write)
- Applied to query at search time before ADC scoring
- Not applied for IP/Cosine (these metrics are rotation-invariant for normalized vectors)

---

## Vectorization / SIMD

Two levels of SIMD acceleration:

### JNI Native (`SimdVectorComputeService`)

- `bulkQuantizedDotProduct()`: SIMD dot product between transposed quantized vectors
- `bulkSOARDistance()`: SIMD SOAR distance computation
- Probed once at class-load time; falls back to Java if unavailable

### Panama Vector API (`ClusterANNVectorizationProvider`)

- `PanamaBulkVectorOps` on Java 21+ with `jdk.incubator.vector` module
- `DefaultBulkVectorOps` scalar fallback (uses `Math.fma`)
- Used by k-means clustering (distance computation in batch)

---

## Flat Vectors Layer

ClusterANN delegates raw vector storage to Lucene's `FlatVectorsWriter`/`FlatVectorsReader`:

- **Write**: `flatVectorsWriter.flush()` / `flatVectorsWriter.mergeOneField()` stores raw float vectors
- **Read**: `flatVectorsReader.getRandomVectorScorer()` provides exact scoring for:
  - Brute-force fallback (tiny segments)
  - Exact rescore (if ADC is disabled)
  - `ordToDoc()` mapping

This separation means ClusterANN's IVF index is an **overlay** on top of the standard flat vector format. The `.clam`/`.clap` files contain only the IVF structure; raw vectors live in Lucene's standard `.vec` files.

---

## Segment Merge Strategy (Current)

The current merge is a **full rebuild**:

```
mergeOneField(fieldInfo, mergeState):
  1. Read all live vectors from source segments
  2. Reservoir-sample 4096 vectors
  3. Estimate numCentroids = max(2, min(4096, (n+256)/512))
  4. Run full HierarchicalKMeans with reservoir as initial centroids
  5. Compute SOAR secondary assignments
  6. Write fresh .clam + .clap
```

**Implications**:
- Every merge re-clusters from scratch (O(n*k*iterations))
- Reservoir seeding provides faster convergence than random init
- No centroid inheritance from source segments
- No posting list concatenation optimization for small merges

---

## Configuration Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| TARGET_CLUSTER_SIZE | 512 | `ClusterANNFormatConstants` |
| SOAR_LAMBDA | 1.0 | `ClusterANNFormatConstants` |
| BLOCK_SIZE | 32 | `ClusterANNFormatConstants` |
| SECTION_ALIGNMENT | 16 bytes | `ClusterANNFormatConstants` |
| MIN_ADC_VECTORS | 32 | `ClusterANNFormatConstants` |
| MIN_IVF_VECTORS | 100 | `ClusterANN1040KnnVectorsReader` |
| NPROBE_MULTIPLIER | 2 (configurable via system property) | `NearestProbeScheduler` |
| MAX_K_PER_LEVEL | 128 | `HierarchicalKMeans` |
| SPLIT_THRESHOLD | 1.5x target | `HierarchicalKMeans` |
| SOAR_CANDIDATE_LIMIT | 10 | `IVFIndexBuilder` |
| TEMP_FILE_THRESHOLD | 10M vectors | `ClusterANNVectorValues` |

---

## Class Diagram

```
ClusterANN1040KnnVectorsWriter (KnnVectorsWriter)
  ├── FlatVectorsWriter (delegates raw vector storage)
  ├── IVFIndexBuilder (clustering + SOAR)
  │     ├── HierarchicalKMeans
  │     │     └── KMeans
  │     └── ClusterANNVectorValues (vector access abstraction)
  ├── QuantizedVectorWriter (block-columnar quantized output)
  ├── PostingListCodec (adaptive doc-ID encoding)
  └── RandomRotation (variance redistribution)

ClusterANN1040KnnVectorsReader (KnnVectorsReader)
  ├── FlatVectorsReader (raw vector access + exact scoring)
  ├── ClusterANNFieldState (per-field metadata from .clam)
  ├── NearestProbeScheduler (centroid selection)
  ├── OptimizedProbeScheduler (I/O reordering + early termination)
  ├── ClusterANNCentroidScanner (per-centroid scan)
  └── QuantizedVectorReader (ADC block scoring)
        └── CandidateCollector (amortized top-N)
```

---

## Distance Metrics

| Lucene VectorSimilarityFunction | ClusterANN DistanceMetric | ADC Similarity Formula |
|---------------------------------|---------------------------|------------------------|
| EUCLIDEAN | L2 | `1 / (1 + max(score, 0))` |
| DOT_PRODUCT | INNER_PRODUCT | `max((1 + rawDot) / 2, 0)` |
| MAXIMUM_INNER_PRODUCT | INNER_PRODUCT | `rawDot >= 0 ? rawDot + 1 : 1/(1-rawDot)` |
| COSINE | COSINE | `max((1 + rawDot) / 2, 0)` |

For COSINE, vectors and centroids are normalized before quantization. For L2, random rotation is applied.
