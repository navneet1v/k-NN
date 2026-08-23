# Implementation Guide: Packed docId / sortedOrds in `.veqo` (off-heap)

**Goal:** move the O(N log N) doc-sort from read-time to write-time. Store two permutations in the
`.veqo` file, bit-packed with Lucene's `DirectWriter`, and read them lazily off-heap with
`DirectReader`. No sort, no upfront decode at read.

**Decisions locked in:**
- Off-heap reads via `DirectReader` (`LongValues`), not materialized `int[]`.
- Store **both** `docIdMap` and `sortedOrds` (zero read-time computation; no inversion pass).

---

## Data to store (both are permutations → `ceil(log2 N)` bits/entry, not further compressible)
- `docIdMap`: `physicalOrd → docId`. Used by `ordToDoc(ord)`. (Replaces today's raw-int docId map.)
- `sortedOrds`: `rank → physicalOrd`, rank = ascending-doc order. Used by iterator `index()`.
  Computed at **write time** as `argsort(docIdMap)`.
- Do NOT store the ascending docId list; derive `docID(rank) = docIdMap[sortedOrds[rank]]`.

## Key APIs + gotchas
- Write: `int bits = DirectWriter.bitsRequired(maxValue);`
  `DirectWriter w = DirectWriter.getInstance(output, count, bits);` → `w.add(long)` × count → `w.finish()`.
  - `bitsRequired` rounds up to a supported width (1,2,4,8,12,16,…). Use its return for BOTH writer and reader.
  - Guard `count == 0` and `maxValue == 0` (use `Math.max(1, maxValue)` or skip the section).
- Read: `RandomAccessInput ras = input.randomAccessSlice(offset, byteLen);`
  `LongValues v = DirectReader.getInstance(ras, bits);` → `(int) v.get(index)`.
  - Lazy/off-heap → no upfront cost. Keep `input` open for the values' lifetime.
  - `LongValues` from `DirectReader` does positional reads → safe to share across `copy()` instances.

## Rule of thumb: unsorted → `DirectWriter`, monotonic → `DirectMonotonic`
`DirectWriter` is a **fixed-width packed-int** writer — every value takes `bitsPerValue` bits, so it
handles **arbitrary/unsorted** values (permutations) perfectly. Sortedness is irrelevant to it.
`DirectMonotonicWriter` only helps a **non-decreasing** sequence (it delta-encodes against a linear
trend); on a permutation its residuals blow up to ~N, so it's the wrong tool. Both `docIdMap` and
`sortedOrds` are permutations → use `DirectWriter`.

`DirectWriter` is **positional**: the i-th value you `add()` is what `DirectReader.get(i)` returns.
So add in the order you'll index by (physicalOrd for `docIdMap`, rank for `sortedOrds`).

## Worked example — writing an unsorted docId map with `DirectWriter`

Write side (inside the writer), `docIdMap[p] = source.ordToDoc(order[p])`:
```java
// 1) bit width must cover the largest value written (the max docId). Guard 0.
int maxDocId = 0;
for (int p = 0; p < count; p++) maxDocId = Math.max(maxDocId, docIdMap[p]);
int bits = DirectWriter.bitsRequired(Math.max(1, maxDocId));

// 2) writer for exactly `count` values at `bits`
long start = output.getFilePointer();
DirectWriter w = DirectWriter.getInstance(output, count, bits);

// 3) add all values in physicalOrd order (positional: get(p) returns docIdMap[p])
for (int p = 0; p < count; p++) {
    w.add(docIdMap[p]);
}
w.finish();                                   // may pad to a block boundary

// 4) capture the packed byte length (NOT count*bits/8 — finish() pads) for the reader's slice
long byteLen = output.getFilePointer() - start;
// persist `bits` and `byteLen` (framing) so the reader can slice + decode
```

Read side (off-heap), `ordToDoc(physicalOrd)`:
```java
RandomAccessInput ras = input.randomAccessSlice(offset, byteLen);
LongValues docIds = DirectReader.getInstance(ras, bits);
int docId = (int) docIds.get(physicalOrd);
```

Same recipe for `sortedOrds`, only the max differs (values are physical ordinals):
```java
int bits = DirectWriter.bitsRequired(count - 1);  // ordinals are 0..count-1
DirectWriter w = DirectWriter.getInstance(output, count, bits);
for (int rank = 0; rank < count; rank++) w.add(sortedOrds[rank]);
w.finish();
// read: index(rank) = (int) sortedReader.get(rank)
```

Constraints to remember: values must be **non-negative** and `≤` the max passed to `bitsRequired`;
you must `add()` exactly `count` values before `finish()`; use the **same `bits`** on read and write.

## Per-section on-disk framing (do this for docIdMap, then sortedOrds)

Target framing per section:
```
[ bits (vInt) ][ byteLen (vLong) ][ packed bytes (byteLen) ]
```

### The framing-order problem (read before Phase 1)
`byteLen` is unknown until AFTER `DirectWriter.finish()`, but the framing wants it before the bytes.
`DirectWriter.finish()` may also pad to a block boundary, so `byteLen` must be the actual bytes
written (capture via file-pointer delta), and `randomAccessSlice(offset, byteLen)` over the padded
region is fine because `DirectReader` addresses by bit index.

Three ways to resolve it — pick one and keep writer/reader symmetric:

1. **Trailer of lengths (recommended).** Write `bits` + the packed bytes for both sections inline,
   and record each section's `(offset, byteLen)` in a small fixed trailer at the end of the file
   (like the offsets Lucene keeps in `.vemq`). Reader reads the trailer first, then slices. Cleanest
   for random access; no buffering.
2. **Scratch buffer.** Write each `DirectWriter` into an in-memory buffer to learn its length, then
   write `bits` + `byteLen` + buffer to the real output. Simple, sequential; small extra memory.
3. **Derive length (avoid).** Don't store `byteLen`; recompute from `bits` and `count` at read.
   Risky — `DirectWriter` block-padding makes the byte length non-obvious.

**Chosen framing:** _TBD — decide between (1) trailer and (2) scratch buffer before starting Phase 1._

## Write-time argsort note
`sortedOrds` sorts ordinals by `docIdMap[ord]`. docId↔ordinal is a bijection within the field, so all
docIds are distinct → the sort is total, no tie-break needed.

## Phases
1. **Writer** — build `docIdMap[]`; compute `sortedOrds = argsort(docIdMap)`; pack both (framing above); ensure record region still starts right after all metadata (`dataOffset`).
2. **Version** — bump `VERSION_CURRENT`. We haven't shipped v0, so a hard cut to v1 is fine (no dual-read branch needed).
3. **Reader / `FieldEntry`** — read each section's `bits` (+ `byteLen`), build `randomAccessSlice` + `DirectReader`, seek past it; store the two `LongValues` in `FieldEntry` (drop raw `int[] docIdMap`).
4. **Values** — `ordToDoc(ord) = (int) docIdReader.get(ord)`; delete `docSortedOrds()`; iterator uses `sortedOrds` reader: `index(rank)=(int) sortedOrds.get(rank)`, `docID(rank)=(int) docIdReader.get(sortedOrds.get(rank))`; `advance` binary search unchanged.
5. **Verify** — existing `LocalityOrderedQuantizedVectorsRoundTripTests` is the gate (ascending docs, `index()` mapping, coverage, `advance`). Should pass unchanged.

## Gotchas checklist
- [ ] Same `bits` value used on write and read (from `bitsRequired`).
- [ ] `count == 0` guarded.
- [ ] `input` kept open for values lifetime; `close()` closes it.
- [ ] `dataOffset` (records) recomputed after the new packed sections.
- [ ] Iterator `advance` binary search still probes `docIdMap[sortedOrds[mid]]` via the readers.
- [ ] `copy()` shares the same `LongValues` readers (they're positional/stateless).
