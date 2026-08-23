# Faiss SQ 1-bit Integration Feasibility (Checklist Item #4)

**Verdict: FEASIBLE.** The single-layer build, graph extraction, and graph remap are all achievable on the custom `FaissSQHnsw` path. One design interaction was discovered (strict page padding vs. Faiss's built-in permute) — flagged below, not a blocker.

This validates the *integration point*, not just the algorithm (which the simulation already validated).

---

## Access Path (confirmed)

From the native index memory address (a `faiss::IndexBinaryIDMap*`):

```cpp
auto binaryIdMap  = (faiss::IndexBinaryIDMap*) indexMemoryAddress;
auto faissSQHnsw  = (knn_jni::FaissSQHnsw*) binaryIdMap->index;   // extends IndexBinaryHNSW
auto& hnsw        = faissSQHnsw->hnsw;                            // faiss::HNSW, all members public
auto& idMap       = binaryIdMap->id_map;                         // std::vector<int64_t>, ordinal -> docId
```

`FaissSQHnsw final : faiss::IndexBinaryHNSW`, which holds `HNSW hnsw` **by value**. All of `hnsw.neighbors`, `hnsw.offsets`, `hnsw.levels`, `hnsw.cum_nneighbor_per_level`, `hnsw.assign_probas`, `hnsw.entry_point`, `hnsw.max_level` are public, mutable members. Same access pattern already used by `addDocsToSQIndex`.

---

## (a) Force Single-Layer Build — FEASIBLE

The `HNSW` constructor calls `set_default_probas(M, 1/log(M))`, producing a geometric level distribution. `random_level()` walks the `assign_probas` CDF to pick each node's top level.

**To force all nodes to level 0:** override `assign_probas` after constructing `FaissSQHnsw` but before any `add`:

```cpp
hnsw.assign_probas.assign(1, 1.0);              // random_level() always returns 0
hnsw.cum_nneighbor_per_level.assign({0, 2*M});  // only the 2*M level-0 block is allocated
```

Any `f in [0,1)` satisfies `f < assign_probas[0]=1.0`, so `random_level()` returns 0 for every node. `prepare_level_tab` then sets `levels[i]=1` (base level) for all, `max_level=0`, and each node reserves exactly one `2*M` neighbor block → **uniform offset stride** (`offsets[i] = i * 2M`).

**Integration point:** add a `singleLayer` flag to `BinaryIndexService::initFaissSQIndex` (gated on the locality feature). This runs before any `add`, satisfying the "cum_nneighbor_per_level must not change after first add" constraint.

**Bonus:** aligns with the existing search path — `FaissCagraHNSW.load()` already sets `maxLevel=1`, `entryPoint=-1` and uses provided entry points. Single-layer SQ indices flow through that path unchanged.

---

## (b) Extract Graph — FEASIBLE (trivial)

New JNI method `extractSQHnswGraphData(indexMemoryAddress)`: read `hnsw.neighbors` (`MaybeOwnedVector<int32_t>`), `hnsw.offsets` (`std::vector<size_t>`), `hnsw.levels` (`std::vector<int>`) and copy to Java arrays. All public members; straightforward primitive-array JNI copies.

For the single-layer case, `offsets` is uniform stride so we can even skip returning it — Java knows every node has `2*M` slots.

---

## (c) Remap Graph — FEASIBLE, and cheaper than planned

**Faiss ships a built-in `HNSW::permute_entries(const idx_t* map)`** (`impl/HNSW.cpp:1087`) where `map[newIndex] = oldIndex`. It:
- reorders per-node neighbor blocks (node at new position `i` gets old node `map[i]`'s list),
- remaps neighbor *values* through the inverse mapping,
- fixes `entry_point`.

This replaces the custom `remapSQHnswNeighbors` we had planned — no hand-written remap needed for the dense case.

**Convention alignment:** Faiss `map[new]=old` equals our `inversePermutation[newOrd]=oldOrd`, which we already compute to write `.vep`. Consistent.

**One addition — the ID map must be permuted too** (permute_entries does NOT touch it):
```cpp
std::vector<int64_t> newIdMap(idMap.size());
for (size_t i = 0; i < idMap.size(); i++) newIdMap[i] = idMap[map[i]];
idMap.swap(newIdMap);
```

---

## (d) Storage Buffer Does NOT Need Permuting

`FaissSQFlat::quantizedVectorsAndCorrectionFactors` is used **only during construction** for distance computation. At write time the storage is skipped (`IO_FLAG_SKIP_STORAGE` → `fourcc("null")`); vectors are served at search from Lucene's `.veb`/`.vep`. So we never permute the native buffer — the `.vep` file is written separately in Java (`PermutedVectorWriter`) using the same inverse permutation. Clean separation between graph (native → `.faiss`) and vectors (Java → `.vep`).

---

## (e) Write Path Already Supports Graph-Only Serialization

`write_HNSW` serializes exactly `assign_probas`, `cum_nneighbor_per_level`, `levels`, `offsets`, `neighbors`, `entry_point`, `max_level`. After `permute_entries` all reflect the new ordering. The existing `skipFlat=true` path writes only the graph. **No write-path changes needed.**

---

## Discovered Design Interaction: Strict Page Padding vs. permute_entries

`permute_entries` requires `map` to be a **dense bijection over [0, ntotal)** — it asserts `map[i] >= 0 && map[i] < ntotal` and builds an inverse assuming every ordinal is used exactly once.

Our recent decision (strict page boundaries + zero-padding) makes the physical ordinal space **larger and gapped**: `[0, num_pages * page_capacity)` with holes at partial-page tails. That is NOT a dense bijection, so `permute_entries` cannot be used directly with padding.

Two resolutions (a design decision, both feasible):

1. **Dense packing → use `permute_entries` as-is (simplest).** No padding; greedy-assigned neighborhoods are still contiguous, occasionally straddling a 32KB boundary. This is what the simulation's 52% result actually used (dense permutation). Zero custom C++.

2. **Strict padding → write a small custom remap (~30 lines).** Generalize `permute_entries` to a sparse target: allocate `neighbors`/`offsets`/`levels` sized for `num_pages * page_capacity`, place each node's block at its padded position, remap neighbor values, mark padding slots as empty (level 0, all `-1`). `ntotal` grows to include padding. Also permute `id_map` with padding entries pointing to a sentinel.

**Recommendation:** Start MVP with **dense packing (#1)** — it uses the built-in permute, matches the simulation's validated 52% number, and defers the padding complexity. Re-introduce strict padding only if the ".vep page also carries edges/metadata" layout (which co-locates graph + vectors per page) is pursued in a later phase — that layout is what truly needs strict alignment, and it is a larger format change beyond the current separate-file design.

---

## Revised Integration Plan (deltas from Implementation_Plan)

| Original plan | Revised after feasibility |
|---------------|---------------------------|
| `remapSQHnswNeighbors` JNI (custom neighbor rewrite) | Thin wrapper over `hnsw.permute_entries(map)` **+ permute `id_map`** |
| Force single-layer: TBD | Set `assign_probas={1.0}` + `cum_nneighbor_per_level={0,2M}` in `initFaissSQIndex` (gated on locality flag) |
| Strict page padding assumed | MVP uses **dense packing** (permute_entries needs a bijection); strict padding deferred to co-located-page format phase |
| Reorder vectors + graph | Graph via native permute; vectors via Java `PermutedVectorWriter` from `.veb`→`.vep`; native SQ buffer untouched |

### Build pipeline (updated ordering)
1. `initFaissSQIndex` — force single-layer (locality enabled)
2. `passSQVectors` + `addDocsToSQIndex` — build graph (unchanged)
3. `extractSQHnswGraphData` → Java (neighbors + levels)
4. Java: `LocalityAwarePageAssigner` → permutation + inverse
5. JNI remap: `hnsw.permute_entries(inverse)` + permute `id_map`
6. `writeIndex(skipFlat=true)` — serializes remapped graph (unchanged)
7. Java: `PermutedVectorWriter` writes `.vep` from `.veb` using inverse permutation

---

## Remaining Unknowns (small)

- **Cagra reader vs. plain single-layer:** confirm `FaissCagraHNSW.load()` is selected for our written index, or whether we need a distinct index-type marker. (The write path emits a standard `IBHf`/`IBMp` structure; the reader detection needs a quick check.)
- **efConstruction with single-layer:** verify graph quality (recall) of a natively single-layer SQ build matches the simulation, which forced single-layer via `assign_probas` in Python Faiss. Expected identical, but worth a build-and-search smoke test.
