# Deep Dive: Reordering the Quantized Vector File

How we physically reorder the SQ 1-bit quantized vectors so graph neighbors are co-located on disk. This is the file that actually stores the vectors and must be reordered to match the permuted graph.

---

## 1. The File We're Reordering: `.veq` / `.vemq`

The k-NN SQ 1-bit path does **not** define its own byte format — `KNN1040ScalarQuantizedVectorsFormat` extends Lucene's `Lucene104ScalarQuantizedVectorsFormat` and uses Lucene's writer/reader verbatim (only swapping in a SIMD scorer). So the on-disk bytes are 100% Lucene 10.4 format.

**Two files per field:**
- **`.veq`** — vector data (records)
- **`.vemq`** — metadata (offsets, dimension, centroid, ordinal→doc config)

(There is also a `.vec` full-precision file used for optional rescoring — see §6.)

### `.veq` record layout (1-bit, `SINGLE_BIT_QUERY_NIBBLE`)

```
offset(ord) = vectorDataOffset + ord * recordSize
recordSize  = ceil(dim/8) + 16

record = [ ceil(dim/8) bytes : packed 1-bit code (8 dims/byte, MSB-first) ]
         [ 4 bytes : lowerInterval        (LE float bits) ]
         [ 4 bytes : upperInterval        (LE float bits) ]
         [ 4 bytes : additionalCorrection (LE float bits) ]
         [ 4 bytes : quantizedComponentSum (LE int32)     ]
```

Records are **contiguous, indexed purely by ordinal** — access is a single multiply, no indirection table. For dim=768: recordSize = 96 + 16 = 112 bytes → 292 records per 32KB page. Byte order is little-endian.

### File envelope (CodecUtil)
Both files are wrapped: index header (`magic + codecName + version + segmentId + segmentSuffix`) at the front, footer (`magic + algorithmId + CRC32`) at the end. The reader enforces the header (including **matching segmentId**), version, and full-file CRC on open (`checkIndexHeader`, `checkFooter`, `checksumEntireFile`).

### The ordinal→docId mapping (in `.vemq`, + appended to `.veq` if sparse)
Three cases, encoded in `docsWithFieldOffset`:
- `-2` empty
- `-1` **dense** (`count == maxDoc`): **ordinal == docId**, no table stored
- `>=0` **sparse**: a `DirectMonotonicReader` over **increasing** docIds + a DISI bitset

---

## 2. The Fundamental Constraint (the crux)

> **Lucene's flat vector format assumes storage ordinals are in increasing-docId order.** Its ordinal→docId mapping is always monotonic — dense (`ord == doc`) or sparse (`DirectMonotonic`, strictly increasing).

A locality permutation **deliberately breaks docId ordering** — ordinal 5 might hold doc 100 and ordinal 6 hold doc 3. This cannot be represented as a valid Lucene `.vemq` ordToDoc (DirectMonotonic requires monotonic input).

**Consequence:** we cannot reorder records inside a standard Lucene `.veq`/`.vemq` and still produce a reader-openable file with a correct ordToDoc. We also cannot use Lucene's `sortMap` flush path (it produces a valid *sorted* segment with monotonic ordToDoc, and the permutation isn't known until after the graph is built anyway — chicken-and-egg).

**Therefore: we write our own container, indexed purely by ordinal, and delegate docId resolution to the FAISS `id_map` (which we permute anyway).**

This is *why* we write our own quantized vector file (`.veqo`) rather than reordering Lucene's `.veq` — not convenience, but a hard format constraint.

---

## 3. The Design: Own the Write Path — Write the Vector File ONCE, Permuted

**Decision:** Rather than letting Lucene write `.veq` in original order and then writing a second reordered file (leaving dead weight), we **own the quantized write path**: quantize in memory, build the graph, compute the permutation, and write the quantized vector file **once, already in permuted order**. Lucene's `Lucene104ScalarQuantizedVectorsWriter` is not used for the quantized file.

This is enabled by two facts (verified against Lucene 10.4 source):

- **`OptimizedScalarQuantizer` is public and fully deterministic/self-contained.** Given `(vector, centroid, similarityFunction, bits)` it returns byte-identical packed code + `QuantizationResult(lowerInterval, upperInterval, additionalCorrection, quantizedComponentSum)`. No hidden per-vector or global state. Write **order** does not affect **values** — only which ordinal a record lands at. So writing in permuted order is safe.
- **We don't need to match Lucene's centroid.** The centroid must only be *consistent* between data quantization (build) and query quantization (search). Since we own the write path, we also own the metadata the query side reads (`getCentroid()` / `getCentroidDP()`). We compute the centroid ourselves (arithmetic mean; L2-normalize for COSINE) and use the same value on both sides. This eliminates the need to replicate Lucene's two-branch merge-centroid logic — any reasonable, consistently-applied centroid is correct.

### File layout (our format — metadata first, then vectors)
```
[CodecUtil index header — KNN codec name, version, segmentId, segmentSuffix]
[metadata: dimension, count, recordSize, centroid (dim × f32), centroidDp, similarityFunction]
[records, contiguous, in PERMUTED order: record at ordinal newOrd occupies newOrd * recordSize]
[CodecUtil footer + CRC32]
```
Metadata precedes the records, so the reader parses in one sequential pass and the record region is `[dataOffset, fileLength − footerLength)` — no trailer/offset directory. No inverse permutation is stored: docId is the join key for anything cross-file (§6). (The Milestone 0 standalone file additionally stores an in-file docId map for a self-contained reader; the final design resolves docId via the FAISS `id_map`.)

- **Same record layout** as Lucene's `.veq` (`[packed code | lowerInterval | upperInterval | additionalCorrection | quantizedComponentSum]`, `ceil(dim/8) + 16` bytes) so the native SIMD offset arithmetic (`base + ord * recordSize`) is byte-identical — **zero scoring-path changes**.
- **Purely ordinal-indexed** — no ordToDoc. Physical position `newOrd` holds the vector of graph node `newOrd`. docId resolution is delegated to the FAISS `id_map`.
- **CodecUtil envelope** with the segment's `getId()` for integrity checking on open.

### The single invariant that ties everything together
Let `inv[newOrd] = oldOrd` (inverse of the page-assignment permutation `perm[oldOrd] = newOrd`). Three things are written from the **same** `inv`:

1. **Graph** — `hnsw.permute_entries(inv)` reorders node blocks + remaps neighbor values so node `newOrd` = old node `inv[newOrd]`.
2. **Vector file** — physical position `newOrd` gets the quantized record of old ordinal `inv[newOrd]`.
3. **FAISS `id_map`** — `newIdMap[newOrd] = oldIdMap[inv[newOrd]]`, so result ordinal `newOrd` → correct docId.

At search: graph yields `newOrd` → native reads `vectorFile[newOrd * recordSize]` (correct vector, no indirection) → collector maps `newOrd` → docId via FAISS id_map. Fully consistent.

---

## 4. Where It's Written (build pipeline — owned path)

The float vectors are buffered in memory during indexing (the raw-float `FlatFieldVectorsWriter`, which also produces the `.vec` full-precision file). At flush/merge we take over:

```
1. Buffer float vectors                # existing (raw float writer → .vec, original order)
2. Compute centroid = mean(vectors)    # NEW (ours; l2normalize for COSINE); centroidDp = dot(c,c)
3. Quantize in memory:                 # NEW — OptimizedScalarQuantizer.scalarQuantize(copy, scratch, 1, centroid)
     for each ordinal: packed code + 4 correction factors   (packAsBinary)
4. Build native graph from quantized values   # existing pass/add path, fed from our in-memory values
5. extractSQHnswGraphData → Java              # NEW: neighbors + levels
6. LocalityAwarePageAssigner → perm, inv      # NEW (Java)
7. permute_entries(inv) + permute id_map      # NEW (JNI)
8. writeIndex(skipFlat=true)                  # existing — serializes permuted graph (.faiss)
9. Write quantized vector file ONCE in permuted order:   # NEW writer
     header; for newOrd in 0..N-1: emit record for inv[newOrd]; centroid+meta; footer
```

No original-order quantized file is ever written — a single write in permuted order. `scalarQuantize` mutates its input in place (subtracts centroid), so quantize on a **copy** of each float vector.

---

## 5. Where It's Read (search path)

Today `FaissScalarQuantizedFlatIndex` wraps Lucene's `FlatVectorsReader`; the scorer extracts the mmap slice + base address (`MemorySegmentAddressExtractorUtil`) and native SIMD does `base + ord * recordSize`.

For locality indices, a **custom reader** opens our vector file:
- verifies CodecUtil header/footer, slices the record region, exposes the base address the same way (reusing `MemorySegmentAddressExtractorUtil`),
- exposes `getCentroid()` / `getCentroidDP()` from our metadata (query quantization uses these),
- **scoring path unchanged** — still `base + ord * recordSize`; only the file/base pointer differs,
- **ordToDoc** — FAISS `id_map` (permuted), already the mechanism used in the memory-opt path (`OrdinalTranslatedKnnCollector` + `scorer::ordToDoc`).

Backward compatibility: pre-locality segments (Lucene `.veq`/`.vemq`) continue through the existing path; the codec detects which format a segment was written with (version/marker) and routes accordingly.

---

## 6. Full-Precision Rescoring (`.vec`) — decoupled by docId

Rescoring is a **separate flow keyed on docIds, not ordinals.** The ANN pass produces docIds (physical ordinal → docId via the id_map); the rescore step then re-scores those docIds against full-precision vectors. So the locality reordering and the rescore flow are fully decoupled by the docId join key — the reordering never touches the rescore path.

Consequently `.vec` stays in **original** order (written by the raw-float writer) and is read **by docId**, exactly as today. We do **not** store an inverse permutation, and we do not reorder `.vec` (reordering ≈3 KB/vector buys nothing — rescore reads are scattered regardless).

**Decision: `.vec` unchanged (original order, read by docId); no inverse permutation; rescore is an independent docId-based flow.**

---

## 7. Storage Cost & Open Items

- **No dead-weight quantized file.** Because we own the write path and quantize in memory, the quantized vector file is written **once**, already permuted. There is no original-order Lucene `.veq` left on disk. (This resolves Design Critique #6 — the earlier "write `.vep` alongside Lucene's `.veq`" approach is superseded.)
- **`.vec` full-precision retained.** The raw float file is still produced (original order) by the raw-float format for optional rescoring.
- **`inv` is build-time only.** The inverse permutation drives the graph permute, vector-file order, and id_map at write time, but is **not persisted** — nothing at search needs physical→original ordinal (rescore is docId-based, §6).
- **What we take on by owning the path:** (1) centroid computation (mean; weighted or recompute at merge — but any consistent centroid is correct, §3); (2) a quantization orchestrator calling `OptimizedScalarQuantizer`; (3) our own vector-file writer + a reader that extends `FlatVectorsReader` and implements `QuantizedVectorsReader` (plugs into the codec framework like Lucene's SQ reader — see §8); (4) codec wiring to route SQ-locality fields to our writer/reader instead of `Lucene104ScalarQuantized*`; (5) segment-file registration for the new extension.
- **Sparse segments.** The Lucene ordinal is already a compacted 0..N-1 space; our permutation operates on that compacted space, and the FAISS id_map carries compacted-ordinal→docId. No extra handling beyond permuting id_map.
- **Strict page padding** (from the earlier decision) is deferred — it makes the ordinal space gapped and is incompatible with `permute_entries`. MVP uses dense packing (see Feasibility doc §"Discovered Design Interaction").
- **COSINE normalization.** For COSINE, L2-normalize vectors (and centroid) before quantization, matching Lucene. (Cohere IP test data needs no normalization.)

---

## 8. Alternative Considered: Reuse Lucene's `.veq` format + writer, with an inverse map on top — REJECTED

**Idea:** store quantized vectors via Lucene's `Lucene104ScalarQuantizedVectorsWriter`/`Reader` and keep a separate inverse permutation (physical → original ordinal) on top, instead of writing our own reordered file.

The verdict depends entirely on **what physical order the `.veq` bytes are in** — because locality lives in the byte layout, not in any map layered over it. An inverse map only provides *correctness* (finding the right vector); it moves no bytes on disk.

**Interpretation A — Lucene `.veq` in original order + inverse map.** Rejected. Vectors stay in insertion order, so graph-neighbors remain scattered across pages → **zero locality benefit** (the entire ~52% page reduction is lost). Worse, search would do `physical ordinal → inverse map → original ordinal → read .veq[original]`: a dependent lookup per vector on the hot path (the rejected "Approach B" indirection), and the vector it finally reads is in a scattered location anyway. Strictly worse — indirection cost with none of the gain.

**Interpretation B — Lucene `.veq` written in reordered order.** Lucene's writer cannot do this directly: its `ordToDoc` must be monotonic (dense `ord==doc` or sparse increasing docId), and a locality permutation is non-monotonic (§2). The only way to force it is a **fake-docId hack**: write the physical ordinal *as* the docId (so Lucene sees a dense file), reorder vectors as you feed them, and keep the real docId in a side map. This "works" but costs:
- **Fake-docId hazard** — Lucene's `ordToDoc` now returns the physical ordinal; any code path trusting it silently breaks. Real docId must always come from the side map.
- **Writer timing** — Lucene's writer runs during flush, before the graph/permutation exists; feeding reordered input means running it *after* the graph and re-quantizing from floats in physical order (its writer only accepts floats).
- **Format coupling** — to Lucene's `.veq`/`.vemq` quirks and the monotonic workaround.

**What we DO reuse.** The genuinely valuable target is Lucene's *reader machinery* (`OffHeapScalarQuantizedVectorValues`: SIMD-friendly `getSlice()`, correction-factor reads). Our `.veqo` **record layout is already byte-identical to Lucene's `.veq` record**, so our small values view exposes the exact same `getSlice()` + `base + ord*recordSize` access the SIMD scorer needs — we get the reader benefit *without* adopting the full format or the monotonic workaround.

**Decision.** Keep the **reordered custom `.veqo` file** (the only thing that delivers locality without hot-path indirection), and integrate its reader into the codec framework by extending `FlatVectorsReader` and implementing `QuantizedVectorsReader` (so it plugs in like Lucene's own SQ reader). Reuse Lucene's reader *interface/framework* — yes; reuse its *file format + writer* with an inverse map — no.

---

## 9. Summary

| Question | Answer |
|----------|--------|
| Can we reorder inside Lucene's `.veq`? | **No** — its ordToDoc must be monotonic; a locality permutation is not. |
| So what do we do? | **Own the write path**: quantize in memory, build graph, permute, write the quantized file **once** in permuted order (custom format, ordinal-indexed, CodecUtil-wrapped, no ordToDoc). |
| Is owning quantization safe? | **Yes** — `OptimizedScalarQuantizer` is public and deterministic; write order doesn't affect values. We compute the centroid ourselves and use it consistently on both build and query sides (no need to match Lucene's centroid). |
| How is docId resolved? | FAISS `id_map` (permuted with the same `inv`), as already done in memory-opt search. |
| Scoring-path change? | None — same record layout, still `base + ord * recordSize`; only the file/base pointer changes. |
| When is it written? | Once, after graph build + permutation. No original-order file is written first — no dead weight. |
| Rescoring? | Separate docId-based flow; `.vec` unchanged (original order, read by docId). No inverse permutation stored. |
| Reuse Lucene's `.veq` writer + inverse map? | **No** (§8) — original-order `.veq` = no locality; reordered via Lucene needs a fake-docId hack. We reuse Lucene's reader *framework* (`FlatVectorsReader`/`QuantizedVectorsReader`), not its file format/writer. |
| Consistency guarantee? | Graph (`permute_entries`), vector file, and `id_map` all built from the same inverse permutation `inv`. |
