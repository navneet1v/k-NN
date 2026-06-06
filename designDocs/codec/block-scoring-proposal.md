# Block Scoring: Current Design & Future Optimization

## Current Scoring Logic

### Overview

`QuantizedVectorReader.scoreBlock()` uses a **corrections-first** layout where correction values are read before quantized codes. This enables a block-level early skip and fuses the dot product and correction application into a single interleaved loop that skips invalid vectors early.

### On-Disk Block Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Block of 32 vectors                                              │
├──────────────────────────────────────────────────────────────────┤
│  lower[0..31]    ← 32 x int (float bits)                         │ ← corrections
│  upper[0..31]    ← 32 x int (float bits)                         │    read first
│  add[0..31]      ← 32 x int (float bits)                         │
│  sum[0..31]      ← 32 x int                                      │
├──────────────────────────────────────────────────────────────────┤
│  code[0]         ← packedBytes (e.g., 96B at 768-dim/1-bit)      │ ← codes read
│  code[1]         ← packedBytes                                    │    second
│  code[2]         ← packedBytes                                    │
│  ...                                                              │
│  code[31]        ← packedBytes                                    │
│                                                                    │
│  Total codes: 32 x packedBytes (e.g., 32 x 96B = 3072B)          │
│  Stored contiguously — vec[j] at offset j * packedBytes           │
└──────────────────────────────────────────────────────────────────┘
```

`packedBytes` per vector:
- 1-bit: `(dim + 7) / 8` → 96B at dim=768
- 2-bit: `((dim + 7) / 8) * 2` → 192B at dim=768
- 4-bit: `((dim + 7) / 8) * 4` → 384B at dim=768

### Scoring Flow

```
scoreBlock(input, blockStart, blockSize, ...):

  1. Check if any vector in block is valid (not deleted/visited)
     → if none valid, skip entire block

  2. Ensure query is quantized against this centroid (cached)

  3. Read corrections: lower[], upper[], add[], sum[]

  4. Block-level early skip (IP/Cosine only):
     - Compute score upper bound from corrections alone
     - If upper bound <= collector threshold → skip codes, return

  5. Read codes: flatCodesBuf[0..blockSize*packedBytes]

  6. Bulk dot product (native SIMD or Java fallback):
     - Compact valid entries into validOffsets[]
     - For each valid vector j: rawDotBuf[j] = dotProduct(queryTransposed, codes[j])
     - Only valid vectors are computed (skipping ~20% invalid saves significant work)

  7. For each valid vector j (FUSED correction + similarity):
     a. docScale = (upper[j] - lower[j]) * docBitScale
     b. score = lower[j]*qLowerDim
              + qLower*docScale*sum[j]
              + lower[j]*qScaleCompSum
              + docScale*qScale*rawDotBuf[j]
     c. Convert to similarity (metric-dependent)
     d. Collect into bulkDocs/bulkScores arrays

  8. Bulk collect: if bulkMaxScore > threshold, drain all into KnnCollector
```

### Why This Design Works

1. **Dot product dominates cost**: At 768-dim/1-bit, each dot product processes 96 bytes of packed data. The correction phase is ~4 float multiply-adds per vector — negligible in comparison.

2. **Skipping invalid vectors matters**: With ~20% invalid vectors per block, skipping them in the dot product phase saves ~20% of the most expensive operation. JMH benchmarks confirmed this is a net win over branchless-all-32 approaches.

3. **Corrections-first layout enables block skipping**: Reading corrections before codes lets us compute an upper bound and skip the entire codes read (~3KB at 768-dim/1-bit) if the block can't compete.

4. **Fused loop is cache-friendly**: The correction arrays are small (4 × 32 × 4B = 512B) and hot in L1 when the scoring loop runs immediately after reading.

### Scoring Formula Reference

The ADC score for vector `j` in a block assigned to centroid `c`:

```
rawDotProduct = dotProduct(queryQuantized_transposed, docCodes[j])

docScale = (upper[j] - lower[j]) / (2^docBits - 1)

correctedScore = lower[j] * queryLower * dim
               + queryLower * docScale * componentSum[j]
               + lower[j] * queryScale * queryComponentSum
               + docScale * queryScale * rawDotProduct

EUCLIDEAN:
  distance = queryAdditionalCorrection + additionalCorrection[j] - 2 * correctedScore
  similarity = 1 / (1 + max(distance, 0))

INNER_PRODUCT / MIP:
  rawDot = correctedScore + additionalCorrection[j] + centroidDotProduct - centroidNormSq
  MIP:    similarity = rawDot >= 0 ? rawDot + 1 : 1 / (1 - rawDot)
  COSINE: similarity = max((1 + rawDot) / 2, 0)
```

Where:
- `queryLower`, `queryScale`, `queryComponentSum`, `queryAdditionalCorrection` — from 4-bit query quantization against centroid
- `lower[j]`, `upper[j]`, `componentSum[j]`, `additionalCorrection[j]` — per-document correction values from `OptimizedScalarQuantizer`
- `centroidDotProduct` = `dot(query, centroid)`
- `centroidNormSq` = `dot(centroid, centroid)` (precomputed in `.clam`)

---

## Implemented: MemorySegment + Panama Vector API Bulk Scoring

### Problem

The baseline dot product path copies doc codes from disk into a heap `byte[]` via `input.readBytes()`, then computes one vector at a time using `VarHandle`-based long reads + `Long.bitCount()`. Two inefficiencies:

1. **Copy overhead**: `readBytes` copies from the mmap'd page cache into a heap buffer — unnecessary when MMapDirectory already has the data mapped.
2. **Per-vector query reload**: Each `int4BitDotProductOffset` call re-loads the query's 4 transposed stripes from `byte[]` into registers.

### Solution: Zero-Copy MemorySegment + Bulk4 Panama Vector API

Two changes combined for a **2.4x throughput improvement**:

#### 1. Zero-copy via MemorySegment

Instead of `input.readBytes(flatCodesBuf, ...)`, obtain the underlying `MemorySegment` directly from Lucene's `MMapDirectory`:

```java
MemorySegment memorySegment = ((MemorySegmentAccessInput) input)
    .segmentSliceOrNull(input.getFilePointer(), (long) blockSize * packedBytes);
```

Doc vectors are accessed at computed offsets without any heap copy.

#### 2. Panama Vector API (jdk.incubator.vector) for SIMD loads

Both single-vector and bulk4 functions use `ByteVector.fromMemorySegment()` for doc loads and `ByteVector.fromArray()` for query loads, with vectorized `BIT_COUNT`:

```java
// Single vector — used for tail (0-3 remaining vectors)
static float int4BitDotProductOffset(byte[] query, MemorySegment docs, long offset, int len) {
    LongVector acc0 = LongVector.zero(LONG_SPECIES);
    // ... acc1, acc2, acc3
    for (final int upperBound = BYTE_SPECIES.loopBound(len); r < upperBound; r += VECTOR_BYTE_SIZE) {
        LongVector d = ByteVector.fromMemorySegment(BYTE_SPECIES, docs, offset + r, LITTLE_ENDIAN)
            .reinterpretAsLongs();
        LongVector q0 = ByteVector.fromArray(BYTE_SPECIES, query, r).reinterpretAsLongs();
        // q1, q2, q3 from query stripes
        acc0 = acc0.add(q0.and(d).lanewise(VectorOperators.BIT_COUNT));
        // acc1, acc2, acc3
    }
    // reduceLanes + scalar tail
    return sum0 + sum1 * 2L + sum2 * 4L + sum3 * 8L;
}

// Bulk4 — processes 4 doc vectors per iteration, query loaded once
static void int4BitDotProductBulk4(
    byte[] query, MemorySegment docs,
    long off0, long off1, long off2, long off3,
    int len, float[] results, int ri0, int ri1, int ri2, int ri3
) {
    // 16 LongVector accumulators (4 query stripes × 4 doc vectors)
    // Main loop: load query stripe once, AND+BIT_COUNT against all 4 doc vectors
    // reduceLanes + scalar tail for remainder
}
```

#### 3. Query quantization hoisted to caller

`ensureQueryQuantized(centroid)` is now public and called once in `ClusterANNCentroidScanner.scoreADC()` before the block loop, rather than being checked inside every `scoreBlock` call. The `centroid` parameter was removed from `scoreBlock`.

### Integration in scoreBlock (useBulkSIMD path)

```java
if (useBulkSIMD) {
    int validCount = 0;
    for (int j = 0; j < blockSize; j++) {
        if (validBuf[blockStart + j]) validOffsets[validCount++] = j;
    }
    final MemorySegment memorySegment = ((MemorySegmentAccessInput) input)
        .segmentSliceOrNull(input.getFilePointer(), (long) blockSize * packedBytes);

    int v = 0;
    for (; v + 3 < validCount; v += 4) {
        int j0 = validOffsets[v], j1 = validOffsets[v+1], j2 = validOffsets[v+2], j3 = validOffsets[v+3];
        int4BitDotProductBulk4(currentTransposed, memorySegment,
            (long) j0 * packedBytes, (long) j1 * packedBytes,
            (long) j2 * packedBytes, (long) j3 * packedBytes,
            packedBytes, rawDotBuf, j0, j1, j2, j3);
    }
    for (; v < validCount; v++) {
        int j = validOffsets[v];
        rawDotBuf[j] = int4BitDotProductOffset(currentTransposed, memorySegment, (long) j * packedBytes, packedBytes);
    }
    input.skipBytes((long) blockSize * packedBytes);
}
```

### Benchmark Results

**768-dim, 1-bit, MAXIMUM_INNER_PRODUCT, 20 blocks, single centroid, Apple Silicon (ARM NEON 128-bit):**

#### End-to-end (includes query quantization)

| Path | ops/ms | Speedup |
|------|--------|---------|
| Baseline (byte[] + single-vector VarHandle) | 12.4 | 1.0x |
| MemorySegment + Bulk4 Panama Vector API | 32.1 | **2.6x** |

#### Pure scoring (query quantization pre-computed in setup)

| Path | ops/ms | Speedup |
|------|--------|---------|
| Baseline (byte[] + single-vector VarHandle) | 18.4 | 1.0x |
| MemorySegment + Bulk4 Panama Vector API | 73.9 | **4.0x** |

#### Stack profile (pure scoring, bulk SIMD path)

| % of RUNNABLE | Method | Category |
|---|---|---|
| 19.7% | `int4BitDotProductBulk4` | Dot product |
| 17.4% | `scoreBlock` (corrections, upper-bound, orchestration) | Framework |
| 6.5% | `TernaryLongHeap.downHeap` | Collector |
| 1.7% | `int4BitDotProductOffset` (tail) | Dot product |
| 1.6% | `readFloatsFromInts` | I/O |
| 0.8% | `MemorySegment.asSliceNoCheck` | I/O |

The dot product (`bulk4` + tail) is ~21% of pure scoring time. The `scoreBlock` orchestration (corrections read, upper-bound check, valid-offset gathering) is 17%. Collector insertion is 8%.

### Requirements

- JDK 21+ with `--enable-preview` and `--add-modules=jdk.incubator.vector`
- `MMapDirectory` (required for `MemorySegmentAccessInput`)
- `SPECIES_PREFERRED` = 128-bit on Apple Silicon (ARM NEON), 256/512-bit on x86 AVX2/AVX-512

### Future Work

- On x86 with AVX-512, `SPECIES_PREFERRED` is 64 bytes — the bulk4 function processes 64 bytes/iter instead of 16, potentially yielding further gains.
- Consider 8-vector bulk variant for x86 where register file is larger.
- Reduce `scoreBlock` orchestration overhead (17%) — potential to inline corrections read or batch upper-bound checks.
