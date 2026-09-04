# Single-Pass `.veqlo` Write from Native Faiss SQ Storage

**Date:** 2026-09-04
**Branch:** `hnsw_locality`
**Goal:** Eliminate the double-write of quantized vectors during the reordered SQ build by writing the
reordered store (`.veqlo`) **directly from the native Faiss SQ flat storage** (via a `MemorySegment` over
its memory address), and by sourcing quantization from the raw `.vec` file — so the quantized data is
written to disk exactly **once**.

---

## Problem: quantized vectors are written twice today

For the reordered format (`Faiss1040SQHNSWReorderedWriter`), on flush/merge:

1. The SQ flat writer quantizes the vectors and persists them to **`.veq`/`.vemq`** — *write #1*.
2. The native graph build reads those codes to build the HNSW graph (`.faiss` uses a *null* storage
   placeholder — it does not persist vectors).
3. `writeReorderedLocalityStore` re-reads the codes from the SQ flat store and re-emits them in physical
   (reordered) order to **`.veqlo`/`.vemlo`** — *write #2 of the same quantized data*.
4. `deleteRedundantSqQuantizedFiles()` deletes `.veq`/`.vemq`.

Net today: **1 persistent copy** (`.veqlo`), but **2 writes + 2 reads + 1 delete** of the quantized data
at index time (write amplification). The raw `.vec` (full precision, for rescore) is written once and is
not part of the duplication.

---

## Key enablers (verified in the C++)

### 1. The native SQ storage is already in the exact `.veqlo` shape
`knn_jni::FaissSQFlat` (`jni/include/sq/faiss_sq_flat.h:264`) stores all quantized data in a single member:

```cpp
// faiss_sq_flat.h:270 — one contiguous, 8-byte-aligned buffer
std::vector<uint8_t, knn_jni::NBytesAlignedAllocator<uint8_t, 8>> quantizedVectorsAndCorrectionFactors;
```

- **Contiguous, ordinal-ordered:** vector `i` at `data() + i * oneElementSize` (distance computer:
  `data + i * oneElementByteSize`, lines 122/159).
- **Stride** = `oneElementSize` = `quantizedVectorBytes + (3×f32 + 1×i32)` = **112 B** for 768-dim.
- **Record layout is byte-identical to `.veqlo`:**
  `[code | lowerInterval f32 | upperInterval f32 | additionalCorrection f32 | quantizedComponentSum i32]`,
  little-endian.
- **8-byte aligned** base (`NBytesAlignedAllocator<…,8>`) — safe for float reads.
- **Base pointer** = `quantizedVectorsAndCorrectionFactors.data()`.

So writing `.veqlo` from this store is a **pure byte-copy per record in permutation order** — no
re-encoding, no re-quantization.

### 2. Phase 1 ingests by ordinal (no mmap needed)
`passQuantizedVectorsAndCorrectionFactors` reads codes purely via
`binarizedVectorValues.vectorValue(ord)` + `getCorrectiveTerms(ord)`
(`MemOptimizedScalarQuantizedIndexBuildStrategy.java:361/371`) — no `getSlice()`. So the quantized source
can be produced in Java (from `.vec`) without materializing `.veq`.

### 3. Lifetime constraint — native store is freed at `writeIndex`
`buildAndWriteIndex` sequence (`MemOptimizedScalarQuantizedIndexBuildStrategy.java`):

```
doBuildIndex()   Phase 1 transfer + Phase 2 insert   → fills FaissSQFlat storage
ordering         permutation + hubs                  → native store ALIVE
writeIndex()     unique_ptr FREES the whole index    → FaissSQFlat storage GONE (:215)
```

The `FaissSQFlat` address is valid **only between ordering and `writeIndex`**. Today
`writeReorderedLocalityStore` runs *after* the native build returns (i.e. after `writeIndex`), so the
native address would already be dangling. **The `.veqlo` write must move to inside `buildAndWriteIndex`,
after ordering and before `writeIndex`.**

---

## Proposed single-pass flow

```
addField:  collect raw float vectors (raw flat writer buffers them)
flush/merge:
  1. raw flat writer → .vec                          (raw floats, once — rescore AND quant source)
  2. compute centroid over .vec                       (pre-pass; must match search-time centroid)
  3. for each batch: read .vec → quantize (OptimizedScalarQuantizer + centroid) → [code|corr]
        → batch-transfer to native FaissSQFlat        (op 1, unchanged)
        → batch-insert into HNSW                      (op 2, unchanged)
     (no .veq; only batch-sized Java memory)
  4. ordering (greedy) from the graph → permutation + hubs
  5. write .veqlo from FaissSQFlat address via MemorySegment, in permutation order (byte copy);
     write .vemlo (centroid, ordToPhysicalOrdMap, hubs)          ← BEFORE writeIndex
  6. writeIndex → .faiss graph (null storage); frees native
```

Result: quantized data written **once** (`.veqlo`), raw **once** (`.vec`); no `.veq`/`.vemq`, no delete,
no full-N Java buffer (only per-batch during quantization; the native store is the accumulation buffer).

Decisions locked in:
- **Keep the two native operations** (batch-transfer + batch-insert) exactly as today.
- **Quantization source = the raw `.vec` file** (we quantize ourselves; drop the SQ flat writer).
- **`.veqlo` is written from the native `FaissSQFlat` memory address** (not by re-reading a flat store).

---

## Work items

1. **Own the quantization (largest piece).** A `QuantizedByteVectorValues` that reads raw floats from
   `.vec` and quantizes on access via `OptimizedScalarQuantizer` + the centroid, feeding `doBuildIndex`'s
   existing batch-transfer unchanged. Requires a **centroid pre-pass** over `.vec` (quantization can't
   start until the centroid is known). Reuse Lucene104's centroid computation + OSQ so the centroid and
   the 112‑B record are byte-identical to what search expects.
2. **JNI getter** — `getSQFlatStorageAddress(indexMemoryAddress)`: down-cast
   `IndexBinaryIDMap → FaissSQHnsw → FaissSQFlat`, return `quantizedVectorsAndCorrectionFactors.data()`
   as `jlong`. (Stride and count are already known in Java; `.data()` is stable post-build.)
3. **Write `.veqlo`/`.vemlo` from the native address, before `writeIndex`.** Thread a locality-write
   callback into `BuildIndexParams` (same pattern as the `Layer0LocalityOrdering` sink); the build
   strategy invokes it after ordering with the native address; the impl wraps it as
   `MemorySegment.ofAddress(addr).reinterpret(count*stride)` and byte-copies each record in permutation
   order, then writes `.vemlo`.
4. **Drop** the SQ flat writer, `.veq`/`.vemq`, and `deleteRedundantSqQuantizedFiles`; keep the raw flat
   writer (`.vec`).

---

## Risks & constraints

- **Fidelity (primary risk):** the self-computed centroid and quantized records must exactly match the
  search-time query quantization + reader expectations, or recall breaks. Mitigate by reusing
  `OptimizedScalarQuantizer` + Lucene104 centroid logic, and add a **round-trip byte-equality test**
  (`.veqlo` from the current path vs the new path) plus a recall check.
- **Lifetime:** the `MemorySegment` read must occur before `writeIndex` frees the store (item 3’s
  placement enforces this).
- **Native access:** the reinterpret is a restricted FFM operation; the module needs native access
  granted (the mmap search path already does this).
- **Memory:** only per-batch Java memory during quantization; the native `FaissSQFlat` store is the
  full-N accumulation buffer (unavoidable — the graph build needs it), read back via `MemorySegment`.

---

## Suggested implementation order
1. `getSQFlatStorageAddress` JNI method + `FaissService`/`JNIService` binding (isolated, low-risk).
2. Thread the `BuildIndexParams` locality-write callback; write `.veqlo` from the native address via
   `MemorySegment` (verify byte-equality while still using today's quantization source).
3. Own the quantization from `.vec` (centroid pre-pass + OSQ), drop `.veq`.

---

## Explicitly out of scope / unchanged
- Batch-transfer and batch-insert native ops (unchanged).
- Reordering the `.vec` file (decided not worth it — ~3% of total I/O).
- Graph remap in `.faiss` (ruled out).
- Strict-page/sparse layout, BFS variant.
