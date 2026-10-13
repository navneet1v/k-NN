# Rejected Ideas — HNSW Locality Reordering

Ideas we evaluated and dropped, kept here so they aren't re-litigated. Companion to
`HNSW_Locality_Reordering_Design.md` (which retains only the *live* alternatives — greedy, BFS, RCM,
Gorder, the `aff/deg` tiebreak, and the deferred METIS min-cut). All simulation numbers are M=16,
ef_construction=256, ef_search=256 unless noted; see `Simulation_Results_M16.md`.

| Idea | Result | Verdict |
|---|---|---|
| Greedy + **degree as a selection weight** | intra-page 17.7% → 5.3% as degree weight rises | Rejected — global degree ≠ page affinity (see below) |
| Greedy + **2-hop refill** | intra-page unchanged vs greedy; pages/query +0.0–0.8 pp | Rejected — adds edgeless nodes (see below) |
| **Reorder the `.vec`** (full-precision) | ~3% of total I/O; shadow measure ~11–29% of that | Rejected — not worth it |
| **Strict-page padding** (pad each cluster to a page boundary) | 92% padding overhead at 100K; ~3× more pages | Rejected — dense packing instead |
| **Graph remap in `.faiss`** | complexity, no locality gain | Ruled out |

---

## 1. Greedy + degree as a selection weight

**Idea:** among candidates, prefer high-degree nodes — score = `affinity + α·degree` — on the intuition
that well-connected hubs make good page members (RCM-flavored).

**Why it fails:** degrades locality monotonically as the degree weight rises — intra-page edge ratio
**17.7% → 16.3% → 10.1% → 5.3%** for α = 0 → 0.05 → 0.2 → 1.0 (M=32 sweep; the mechanism is
degree-independent). A high-degree node's edges mostly point *outside* any single page (they can't
all fit), so pulling it in early adds cross-page edges and displaces a node more tightly bound to
*this* page. Global degree ≠ page affinity. Even as a pure tiebreak it's slightly negative.

**The one degree use that *helps* is the opposite:** `aff/deg` as a *containment* tiebreak (prefer the
*lower*-degree candidate at equal affinity, so its neighborhood is contained on the page) — that is
adopted and lives in the main design doc, not here.

## 2. Greedy + 2-hop refill

**Idea:** when a page's 1-hop affinity frontier empties before the page is full, refill it from the
2-hop neighbors of the page (neighbors-of-neighbors), rather than closing the page early.

**Why it fails (instructive):** any 2-hop node that *has* an edge to the page is already a 1-hop
candidate; the ones 2-hop actually adds have **zero edges to the page**. Co-locating an edgeless node
adds no intra-page edges — locality is about edges, not membership. Measured: 2-hop places ~2.4–3.9%
of nodes but the intra-page edge ratio is byte-identical to plain greedy, and pages/query improves
≤0.8 pp across 10K–1M.

**Note on the "52–56%" figure once attributed to 2-hop:** that was strict-page *padding overhead* (a
strict-page-layout artifact, below), not a real query-benefit number — a layout we don't ship.

## 3. Reorder the `.vec` (full-precision) file

**Idea:** apply the same locality permutation to the full-precision `.vec` file (used for rescore), so
rescore reads are also co-located.

**Why rejected:** the `.vec` full-precision reads are only ~3% of total query I/O, and a measured
"shadow" (reordered-`.vec` amplification without actually reordering the file) showed only ~11–29% of
*that* small slice is recoverable. Not worth the write amplification / complexity. The `.vec` is
deliberately left in original order; rescore reads are layout-invariant by design (this is also why
rescore-on can *mask* the graph-read win end-to-end at high dim — see the main doc §8.4/§10).

## 4. Strict-page padding (one cluster per physical page)

**Idea:** pad each greedy cluster out to the next page boundary so every logical cluster maps to
exactly one physical page (no cluster ever straddles a boundary).

**Why rejected:** clusters are mostly partially filled (by cluster count ~66% close early), so
one-cluster-per-page wastes enormous space — **92% padding overhead at 100K, ~3× more physical pages**
(1301 pages vs 441 dense) — which *increases* read amplification. Dense packing (concatenate clusters,
no gaps) reclaims all of it. Follow-up analysis also showed cluster→page alignment gives ≈0 gain on
its own (intra-cluster edge ratio == intra-page edge ratio; the inter-cluster edges dense packing
gains offset the intra-cluster edges it splits), so there is no locality reason to pad either.

## 5. Graph remap in `.faiss`

**Idea:** additionally permute the node ids inside the HNSW graph structure (`.faiss`) to match the
vector reordering.

**Why ruled out:** substantial added complexity (rewriting graph adjacency + entry points) with no
locality gain on the read path that matters — the win comes from co-locating the *quantized vectors*
a traversal reads (`.veqlo`), not from relabeling graph ids. The permutation is handled at read time
via `ordToPhysicalOrdMap`; the graph itself does not need remapping.
