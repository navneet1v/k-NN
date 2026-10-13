# Cluster Benchmarks — HNSW Locality Reordering (AOSS V2, gamma)

Consolidated latency (p50/p90/p99) and the **5 reordering-benefit BufferPool metrics** across all
cluster runs (baseline = stock SQ codec, candidate = reordered codec; field-name-gated, single-layer
HNSW, SQ 1-bit 32× compression, ef_construction=256, ef_search=256, innerproduct). Source docs in
`benchmarks/`: `hnsw-locality-benchmark-results.md` (1M/10M), `airbnb-100M-run-benchmarks.md` (100M),
`hnsw-locality-113m-benchmark-results.md` + `steady_state_113M_run.md` (113M).

Recall is at parity everywhere (layout-only change) — see `benchmarks/recall_across_datasets.md`.

The 5 metrics = **Allocation.BytesDelta** (bytes S3→pool), **Cache.Misses**, **Cache.Loads**,
**WssRecentMB** (resident working set), **Prefetch.BlocksLoaded** (`JunoSearch/Worker` NodeStats,
Beholder acct 372537475017). Lower = better on candidate. Also tracked from round 2 on: **`EFS.ReadBytes` /
`EFS.ReadOperations`** (actual backing-store reads, `populateDiskIOMetrics`) — the closest node-level analog of PageTouch.

> ⚠ **Confound in all runs before §0 (pre-2026-09-25).** A `search-deploy --knn-jar-only` bug skipped the per-arch
> KNN native-lib patch, so candidate deploys failed on Graviton hosts and the candidate fleet stayed on an **older
> build with KNN prefetch enabled** (993K prefetch calls vs baseline 65K). Sections §1, §1a, §2 therefore compare
> *reordering + prefetch* vs *stock*, not reordering alone. The direction agrees with the clean run (§0), but treat
> their magnitudes as confounded. Fix + details: `benchmarks/knn_jar_only_patch_guide.md`.
>
> ⚠ **Aggregation correction.** `Allocation.BytesDelta`, `Cache.Hits/Misses/Loads/Evictions`, `Prefetch.*` are
> emitted as **per-interval deltas** — aggregate with **SUM** over the window, not Max. `WssRecentMB` is a gauge →
> Max. `HitRate` must be derived as ΣHits/(ΣHits+ΣMisses). §1a round 1 used per-minute Max; §0 and round 2 use SUM.

---

## 0. Clean A/B — prefetch disabled on both fleets, 100M (the reference result)

Source: `benchmarks/single_query_100m_no_prefetch.md` (2026-09-25+). Both fleets on the **same KNN jar**
(sha `99fae3e6…`) with `KNNFeatureFlags.isPrefetchEnabled()` hard-coded `false` (Prefetch.Calls 296 vs 356 —
negligible). `target_index_100m` (100M × 1024-dim), rescore OFF, k=100, both fleets run simultaneously.
**This is the first A/B that isolates reordering alone.**

### 0.1 Summary across regimes

| Regime | OCU | Workload | Read reduction (EFS.ReadBytes) | Latency / throughput | Recall |
|---|---|---|---|---|---|
| **Cold, single query ×10** | 32 | 1 query, 10× serial, empty pool | **−19.8%** (Misses −19.8%, WSS −23.6%) | cold p50 **−21.5%**, cold mean **−30.3%**; warm p90 −33.7%; warms 1 iteration sooner | — |
| **10k distinct, cache fits** | 32 | full sweep, 0 evictions | −0.4% (total) | parity (p50 +5.4%, p99 −3.1%) | 0.6644 / 0.6647 |
| **Stress run 1 (cache thrash)** | 16 | 10k sweep, cold start | **−7% per successful query** | **+43% successful queries**, p90 **−23%**, p99 **−39%**, mean −11% | parity |
| **Stress run 2 (still thrashing)** | 16 | 10k sweep, pool carried over | **−21% per successful query** | +31% successes, +36% QPS, p90 **−38%**, p99 **−38%**, mean **−24%** | parity |
| Stress round 3 (low load) | 16 | first 100 queries | −9% | p50 +41%, p99 −8% (EFS RTT +20% on candidate, n≈95 — noisy) | — |
| Round 4a (100 distinct cold) | 32 | first 100 queries | −3.7% | p50 +64%, p99 −21% (replica-routing noise, see 0.4) | parity |
| **Round 7b — KNN prefetch ON, cold query 1** | 16 | 1 query, same jar both fleets, prefetch enabled | **−24.2% total EFS blocks** (PageTouch 32 KB −27.8%) | not comparable (per-call stack logging) | — |

### 0.2 Cold single query ×10 (32 OCU) — the cleanest per-query signal

| Metric | Agg | Baseline | Candidate | Δ |
|---|---|---|---|---|
| **EFS.ReadBytes** | SUM | 17.77 GB | 14.25 GB | **−19.8%** |
| **EFS.ReadOperations** | SUM | 543.7K | 436.9K | **−19.6%** |
| Cache.Loads | SUM | 541.0K | 434.2K | −19.7% |
| Cache.Misses | SUM | 1.08M | 868.1K | −19.8% |
| Allocation.BytesDelta | SUM | 17.73 GB | 14.23 GB | −19.7% |
| WssRecentMB | Max | 6,550 | 5,000 | **−23.6%** |
| Cache.Evictions | SUM | 0 | 0 | — |
| EFS.ReadKBPerOp | avg | 31.92 | 31.76 | ≈32 KB page ✓ |
| EFS.AverageRTT | avg | 3.53 ms | 3.61 ms | flat — gain is *fewer* reads, not faster ones |

| Latency (client E2E) | Baseline | Candidate | Δ |
|---|---|---|---|
| Cold phase (iters 1–5) p50 | 10,625 ms | 8,343 ms | **−21.5%** |
| Cold phase mean | 12,539 ms | 8,738 ms | **−30.3%** |
| Warm phase (iters 6–10) p90 | 98.6 ms | 65.4 ms | −33.7% |
| Warm phase mean | 78.0 ms | 62.0 ms | −20.5% |
| Time in slow (>1 s) iterations | 62.7 s | 43.6 s | −30% |

**PageTouch (same window):** distinct 32 KB pages **2.00M → 1.45M (−27.7%)**, 8 KB −22.6%, vectors/32 KB page
1.61 → 2.30 (+43%), read-amp ~141× → ~99× (−30%), for the same graph work (3.23M vs 3.33M vectors scored). Per
segment search: p50 −16%, avg −28%, p90 −27%, p99 −29% — savings grow with segment size. The node-level EFS −20%
and PageTouch −28% agree in direction and rough magnitude.

### 0.3 Why the 10k sweep with cache-fit shows ~0

With **0 evictions** each block is read from EFS at most once, so total reads = the **union** of pages touched by
all queries. 10,000 distinct queries touch nearly the whole index on both layouts (~62 GB), so the union is the
same. Reordering cuts pages **per query**; it only shows up in totals/latency when those pages would otherwise be
**re-fetched** (cold start, or a cache too small for the working set → eviction churn). That is exactly the
16-OCU stress regime, where it converts into +31–43% throughput and −23–39% tail latency.

### 0.4 Caveat — replica routing dominates small-n latency

Each index has 16 shards × 2 copies across 2 workers; a query is fast only if **every** copy it hits is warm (one
cold copy ⇒ ~10 s). Small-n p50 comparisons (round 3, 4a) are dominated by which copies happened to be cold.
Compare cold iterations, normalize per successful query, or pin replica selection for fair small-sample latency.

### 0.5 Round 7 (2026-09-29, 16 OCU) — KNN prefetch ON, and a fetch-phase IO fix

Source: `benchmarks/single_query_100m_no_prefetch.md` §Round 7. Both fleets at 16 OCU (2 workers × 24 GCU), query
`{"size":100,"_source":false,"stored_fields":"_none_", knn k=100, rescore:false}`. Instrumentation logs every
`CachedMemorySegmentIndexInput.prefetch` call with its caller stack.

**7b — same jar on both fleets, KNN vector prefetch ENABLED** (first clean prefetch-on A/B). Query 1 only, both
fleets cold on fresh workers, isolated by `request.id`:

| Query 1 | Baseline | Candidate | Δ |
|---|---|---|---|
| KNN prefetch calls | 284,895 | 263,859 | −7.4% |
| Blocks loaded via prefetch | 177,107 | 137,900 | **−22.1%** |
| Blocks loaded on demand | 109,704 | 79,436 | **−27.6%** |
| **Total blocks loaded from EFS** | **286,811 (≈8.75 GiB)** | **217,342 (≈6.63 GiB)** | **−24.2%** |
| PageTouch segment searches / distinct vectors | 314 / 307,433 | 312 / 315,582 | same work (+2.7%) |
| **PageTouch distinct 32 KB pages** | **191,734** | **138,426** | **−27.8%** |
| PageTouch distinct 8 KB pages | 257,028 | 198,279 | −22.9% |
| Pages per segment avg / max (32 KB) | 610.6 / 2,251 | 443.7 / 1,634 | −27% / −27% |

→ **The read saving holds with prefetch on** (−24% EFS blocks, vs −20% with prefetch off in §0.2), and EFS blocks
track PageTouch pages closely (−24% vs −28%). Reordering and prefetch are complementary, not overlapping.

**How much does each KNN prefetch call actually cover?**

| | Baseline | Candidate |
|---|---|---|
| Calls spanning 1 block / 2 blocks / ≥3 | 97.35% / 2.65% / 0 | 96.64% / 3.36% / 0 |
| Prefetch length p50 / p90 / avg | 144 / 144 / 880 B | 144 / **432** / 1,105 B |
| Calls whose block was already resident (0 loaded) | 38% | **48%** |

- The **median prefetch call covers one 144 B record**: `prefetchBasedOnOrdinals`' block-aligned grouping almost
  never coalesces neighbors into one call. Even after reordering, most of a node's neighbors land in *different*
  32 KB blocks, consistent with the simulation's ~17–19% intra-page edge ratio.
- The candidate's saving comes mainly from **more calls landing on already-resident blocks (48% vs 38%)**: neighbors
  from different expansions reuse pages loaded earlier in the same query. Actual coalescing is only slightly better
  (p90 3 records vs 1).
- Implication: the shipped ordering captures *cross-expansion* reuse, not *within-neighbor-list* contiguity. A layout
  that packs a node's full neighbor list into its block (higher intra-page / the sibling term, §9 of the design doc)
  is the remaining headroom for turning prefetch into fewer, larger reads.

**7a — fetch-phase `.fdt` prefetch bug, fixed.** `FetchPhase.execute` (JunoSearchWorker 3.3, L223–265) prefetched
`.fdt` for every hit even with `stored_fields: _none_`, where no stored field is ever read — pure wasted IO. Fix:
add `&& fieldsVisitor != null` to the gate (setting `juno.search.prefetch_docs.fetch_phase.enabled`).

| | Before fix | After fix |
|---|---|---|
| `.fdt` prefetch calls (baseline / candidate) | 527 / 707 | **0 / 0** |
| All prefetch calls (both fleets) | 1,302 | 72 (none from `FetchPhase`) |

The 72 remaining calls are candidate-only, from its 2 segments with soft-deleted docs (`docs.deleted = 4,056`):
sparse ord↔doc `IndexedDISI` reads on the reordered `.vec`, plus a one-time soft-deletes check on `.dvd` at shard open.
Not a reordering cost, but they make the two fleets' indices not byte-equivalent (candidate also has +637 live docs).

**Caveats:** per-call stack logging produced ~100–120K log lines/s, so **only query 1 counts are reliable**
(later iterations lost most lines), and **latency is not comparable** (stack capture dominates: warm ~1 s vs
60–150 ms normally). Use counters or sampled logging for performance runs; use `?preference=<fixed>` for repeats.

### 0.6 Four-questions run (2026-09-30) — prefetch ON, current build, 100M

Source: `benchmarks/benchmark_results_100m_four_questions.md` (plan: `benchmarks/benchmark_plan_100m_four_questions.md`).
Same KNN + storage jar on both fleets, **KNN prefetch ON** (shipping config), fetch-phase `.fdt` fix present,
rescore OFF, 1 serial client, runner `bench_100m.py`. Idle EFS reads verified 0 before start. Data = CloudWatch +
client side only: worker logs (PageTouch, storage-plugin per-call logs) were **not used** because the per-call
prefetch log (~150K lines/s) caused ~80% log loss. E1 (rescore-ON recall), E4b, E4c not run (decision).

| Question | Experiment | Result | Status |
|---|---|---|---|
| **Q1 Recall** | E2 (10k, rescore OFF) | recall@1 0.9771 / 0.9781, @10 0.7367 / 0.7380, @100 0.6644 / 0.6647 | ✅ parity (rescore-ON only from the older confounded run: 0.9449 / 0.9458) |
| **Q2 Per-query reads** | E6 (10k, PageTouch per query, §0.7) | **median query −37.2% distinct 32 KB pages** (p10 −27.2%, p90 −47.7%); **all 10,000 queries** improved; read-amp 141× → 82× | ✅ closed (single cold queries: EFS bytes −19.8% / −24.2%, from a p10-type query) |
| **Q3 Steady state** | E3a (warm, 10k looped, 20 min, 32 OCU) | **p50 −18%, p90 −16.5%, p99 −15%, mean −17.8%, QPS +21.6%** with 0 EFS reads / 0 evictions on both | ⚠ **not reproducible as a layout effect** — the E6r repeat on new hosts gave −3% warm p50 (§0.8); claim = **no slowdown** |
| **Q4 Heavy evictions** | E4a (16 OCU, 10k looped, 30 min) | **+50% successful queries**, p90 −25%, p99 −37%, mean −21%; EFS bytes per successful query **−22%**, evictions per successful query **−29%** | ✅ accepted as stress result (17–28% errors, OOM throttle engaged on both) |

**E2 — 10k distinct queries, single serial pass, fresh 32 OCU (0 errors on both):**

| | Baseline | Candidate | Δ |
|---|---|---|---|
| p50 / p90 / p99 (ms) | 48.3 / 59.3 / 254.2 | 40.1 / 51.2 / 277.9 | −16.9% / −13.7% / +9.3% |
| mean (ms) / QPS | 64.0 / 15.60 | 56.6 / 17.64 | −11.6% / +13.1% |
| EFS.ReadBytes / ReadOperations (SUM) | 61.75 GB / 1.882M | 61.46 GB / 1.874M | −0.5% / −0.4% |
| EFS.AverageRTT | 4.44 ms | 3.96 ms | −10.7% (crosses the 10% confound threshold) |
| Cache.Evictions | 0 | 0 | — |
| Prefetch.Calls / BlocksLoaded / BlocksCacheHit | 826K / 812K / 23.1K | 929K / 808K / 128.4K | +12% / −0.5% / **+456%** |

Totals at parity again (bounded by the working set, as in §0.3). The latency gain here was flagged as
RTT-confounded, but E3a reproduces a similar gain with **zero** EFS reads, so the E2 gain is most likely the same
non-IO effect rather than RTT.

**E3a — steady state, 1 client, 20 min, 32 OCU, warm from E2.** The candidate re-read its whole working set from
EFS (61.4 GB, 0 evictions) in the first 4 minutes — most likely replica routing to the cold copy of each shard — so
the headline uses 06:15–end, where EFS = 0 on both:

| | Baseline | Candidate | Δ |
|---|---|---|---|
| requests | 18,958 | 23,073 | +21.7% |
| p50 / p90 / p99 (ms) | 49.0 / 56.9 / 67.6 | 40.2 / 47.5 / 57.5 | **−18.0% / −16.5% / −15.0%** |
| mean (ms) / QPS | 49.8 / 20.06 | 40.9 / 24.39 | **−17.8% / +21.6%** |
| recall@1 / @100 | 0.9765 / 0.6641 | 0.9789 / 0.6651 | parity |
| EFS.ReadBytes / Evictions | 0 / 0 | 0.36 MB / 0 | ≈ 0 |

Full 20-min window (raw, including the candidate's cold minutes): p50 −17%, p99 +124%, mean −2.6%, QPS +2.7%.

⚠ **Superseded by §0.8: did not reproduce.** A repeat single pass on fresh worker hosts (E6r) gave warm p50 −3%,
vs −17% in E6; host-to-host variance (10–30%) is as large as this gap, so it is not attributed to the layout. The
original note follows. With nothing read from EFS, the gain must come from in-memory work. Candidates:
(a) buffer-pool overhead — every 32 KB block access costs a cache lookup/pin even when resident, and the reordered
layout touches ~28% fewer distinct blocks per query; (b) CPU-cache/TLB locality. Neither is measured. Counter-weights:
the fleets run on different hosts, and the candidate has extra sparse segments (which should make it slower).
**Before using this as a design-doc headline:** rerun with a fixed `?preference=`, compare worker CPU and instance
types, and ideally swap hosts.

**E4a — heavy evictions, 16 OCU (2 × 24 GCU; 24 OCU is not a valid setting), 30 min, `--timeout-s 5 --retries 3`:**

| | Baseline | Candidate | Δ |
|---|---|---|---|
| successes / requests | 1,780 / 2,489 | 2,674 / 3,232 | **+50.2%** successes |
| error rate after retries | 28.5% | 17.3% | −11.2 pp |
| p50 / p90 / p99 (ms) | 567 / 1,356 / 3,959 | 496 / 1,022 / 2,480 | −12.6% / **−24.6% / −37.4%** |
| mean (ms) / successful QPS | 774 / 0.989 | 613 / 1.485 | **−20.9% / +50.2%** |
| recall@1 / @100 | 0.9775 / 0.6662 | 0.9839 / 0.6638 | parity |
| EFS.AverageRTT | 4.37 ms | 4.03 ms | −7.6% (not confounded) |
| Cache.Evictions (SUM) | 2.59M | 2.77M | +6.7% |
| **EFS bytes per successful query** | 331.4 MB | 257.4 MB | **−22.3%** |
| **Evictions per successful query** | 1.46K | 1.03K | **−29.0%** |
| EFS bytes per attempt (incl. failures/retries) | 126.3 MB | 138.4 MB | +9.6% |
| Oom.ThrottleEngagedSeconds | 94.2 | 283.2 | non-zero on both |

Both fleets are bounded by the same EFS throughput (~115 GB / 5 min when saturated); the candidate turns it into
50% more completed queries. Per *attempt* the candidate reads more (it gets further into a request before being
rejected), so the user-facing number is per successful query. Consistent with the earlier prefetch-OFF 16 OCU runs
(+31–43% successes, p99 −38/−39%).

Notes: `KNN.AnnLatencyCount` does not track request counts in any of these runs — not used for normalization.
`JunoSearchWorkerRejections` has no datapoints under this dimension combo.

### 0.7 E6 — per-query pages touched across all 10k queries (closes Q2)

Source: `benchmarks/benchmark_results_100m_four_questions.md` §6. Redeployed builds (2026-09-30 19:48 UTC), same KNN +
storage jars on both, **KNN prefetch ON**, storage-plugin per-call logs **removed** (0 per-call lines in the smoke
test), PageTouch ON. 32 OCU (2 × 48 GCU), **fresh** cold cache, all 10,000 queries, single serial pass in index order,
rescore OFF, 0 client errors on both.

**Attribution & completeness:** each PageTouch record carries `request.id` (one id per search request, across both
workers of a fleet); query index = order of first timestamp (serial run). PageTouch's per-JVM `query#[n]` counter
has **0 gaps on all 4 worker streams** (1.65M lines each, 6.61M total), so no log loss. Every query searches all
segments (329 baseline / 331 candidate — the candidate's 2 extra sparse segments); 1.3% of queries per fleet have more
lines (up to 418) because the optimistic search's phase 2 re-searches some segments. The plan's ±2% rule keeps
**9,854 complete pairs** (1.46% excluded); using all 10,000 pairs gives the same answer (p10/p50/p90 = 27.0 / 37.1 /
47.6%, aggregate −36.8%).

**Per-query distribution (9,854 complete pairs):**

| | Baseline | Candidate | Δ |
|---|---|---|---|
| 32 KB pages p10 / p50 / p90 / p99 / max | 132,908 / 164,382 / 185,674 / 198,818 / 214,832 | 72,966 / 101,976 / 131,079 / 149,324 / 163,130 | −45.1% / −38.0% / −29.4% / −24.9% / −24.1% |
| 32 KB pages mean | 161,200.5 | 101,793.6 | **−36.9%** |
| 8 KB pages p50 / p90 / mean | 213,772 / 247,004 / 210,099 | 145,040 / 186,723 / 144,814 | −32.2% / −24.4% / −31.1% |
| distinct vectors per query p50 / mean | 267,052 / 260,933 | 284,186 / 278,587 | **+6.4% / +6.8%** (see note) |
| segment searches per query (mean) | 329.1 | 331.1 | +0.6% |
| read amplification 32 KB p50 / mean | 140.6× / 141.2× | 82.8× / 82.4× | **−41.1% / −41.6%** |

**Paired reduction r(q) = 1 − cand.pages(q) / base.pages(q):**

| | p10 | p25 | **p50** | p75 | p90 | mean | queries with r > 0 | aggregate |
|---|---|---|---|---|---|---|---|---|
| 32 KB pages | 27.2% | 31.3% | **37.2%** | 42.9% | 47.7% | 37.4% | **100.0%** | −36.9% |
| 8 KB pages | 22.0% | 25.9% | **31.2%** | 36.7% | 41.4% | 31.6% | 100.0% | −31.1% |

Distinct vectors scored, cand/base − 1: p10 +4.3%, p50 +6.4%, p90 +10.8%.

**By segment size (per segment search):**

| Segment size (vectors) | Baseline searches | Candidate searches | Baseline pages32 / search | Candidate pages32 / search | Δ |
|---|---|---|---|---|---|
| < 10K | 740,963 | 781,223 | 10.8 | 12.1 | +12.5% |
| 10K – 100K | 1,091,069 | 1,070,880 | 142.9 | 123.0 | −13.9% |
| 100K – 500K | 741,258 | 691,091 | 578.3 | 363.7 | −37.1% |
| 500K – 1M | 160,066 | 230,112 | 1,114.5 | 651.9 | −41.5% |
| ≥ 1M | 560,352 | 540,551 | 1,487.8 | 872.2 | −41.4% |

The saving grows with segment size and plateaus at ~41% for segments ≥ 500K vectors. Tiny segments (< 10K
vectors, ~40–50 pages in total) change by about one page either way (+12.5% = 10.8 → 12.1).

**Histogram (32 KB pages per query, bucket 5,000):** baseline is centred at 160K–175K (peak 1,109 queries in
165–170K; nothing below 50K, max 215K); candidate is centred at 95K–105K (peak 800 in 100–105K; nothing above 165K).
The two distributions barely overlap: only ~1% of baseline queries are below the candidate's median. Full table in
the source doc.

**Cross-check with the single-query runs:** query 0 = 200,499 → 145,045 pages (**−27.7%**), identical to Round 4
(query 0 ×10, −27.7%) and Round 7b (first query, −27.8%) — PageTouch is deterministic for the same query and index.
Query 0 sits near the **p10** of the paired reduction, so the single-cold-query EFS byte savings (−19.8% / −24.2%)
come from an unfavourable query; the median query saves ~37% of pages.

**Client + CloudWatch (sanity):** 10,000 / 0 errors on both; p50 33.6 → 28.0 ms (−16.6%), p90 40.8 → 34.2 ms
(−16.2%), p99 −0.4%, mean −14.4%, QPS +16.8%; recall@1 0.9771 / 0.9781, @100 0.6644 / 0.6647 (parity); total
EFS.ReadBytes 61.745 → 61.462 GB (−0.5%, whole working set loaded once, 0 evictions); **EFS.AverageRTT +5.5% on the
candidate** (slower storage, yet faster queries); Prefetch.BlocksCacheHit 75.5K → 197.7K (+162%).

**Notes for the write-up:**
- **The two graphs are not identical.** The indexes were built separately (the candidate also has +637 docs and 2
  sparse segments), so the HNSW graphs and segment structure differ — see the vector counts (+6.4%) and the
  different per-bucket search counts. The candidate does *more* graph work and still touches 37% fewer pages, so
  per vector scored the page saving is ~41%. The simulation (design doc §10.1) is the same-graph comparison.
- The p50 −16.6% in this run did **not** reproduce on new hosts (E6r: warm p50 −3%, §0.8). Treat single-run
  warm-latency deltas as host variance.



### 0.8 E6r — repeat of E6 on fresh worker hosts (2026-09-30)

Source: `benchmarks/benchmark_results_100m_four_questions.md` §6.1–§6.2. Same config as E6 (32 OCU, 2 × 48 GCU,
prefetch ON, rescore OFF, all 10,000 queries, single serial pass), after a reset that brought up **new worker hosts**
(different log streams) with a cold cache. 10,000 / 10,000 successes, 0 errors on both; 0 `query#` gaps on all 4
streams (6.61M lines); 9,853 complete pairs (1.47% excluded for phase-2 re-searches).

**Pages: reproduced to within a few pages out of ~161K per query** (PageTouch is deterministic):

| | E6 | E6r |
|---|---|---|
| Paired 32 KB reduction p10 / p50 / p90 | 27.2% / **37.2%** / 47.7% | 27.2% / **37.2%** / 47.7% |
| Paired 8 KB reduction p50 | 31.2% | 31.2% |
| 32 KB pages mean, baseline / candidate | 161,200.5 / 101,793.6 | 161,199.0 / 101,792.1 |
| Aggregate Σcand/Σbase − 1 (32 KB) | −36.9% | −36.9% |
| Read amplification 32 KB mean | 141.2× / 82.4× | 141.2× / 82.4× |
| Distinct vectors p50, cand/base − 1 | +6.4% | +6.4% |
| Query 0 / query 1 (32 KB) | −27.7% / −36.0% | −27.7% / −36.0% |
| Queries with fewer pages on candidate | 100% | 100% |

**Latency: did NOT reproduce.** It depends on the worker hosts, not on the layout alone:

| | E6 baseline | E6 candidate | E6r baseline | E6r candidate |
|---|---|---|---|---|
| All queries: p50 / p90 / p99 (ms) | 33.6 / 40.8 / 254.1 | 28.0 / 34.2 / 253.0 | 36.8 / 45.3 / 242.0 | 35.9 / 46.0 / 254.2 |
| All queries: mean (ms) / QPS | 50.2 / 19.88 | 43.0 / 23.23 | 52.4 / 19.04 | 51.8 / 19.26 |
| Warm (queries 2,000–9,999): p50 / p90 / p99 / mean | 33.2 / 37.6 / 44.4 / 33.7 | 27.6 / 31.1 / 37.3 / 28.0 | 36.4 / 42.5 / 51.4 / 37.3 | 35.2 / 41.6 / 51.9 / 36.2 |
| Candidate Δ, warm p50 / mean | | −16.9% / −16.9% | | −3.3% / −2.9% |

Between runs the candidate's warm p50 moved 27.6 → 35.2 ms (+28%) and the baseline's 33.2 → 36.4 ms (+10%), just
from new hosts. Host variance (~10–30%) is as large as the warm-latency gaps seen in E2 (−17%), E3a (−18%) and E6
(−17%), so **none of those are quoted as a layout effect**. The page reduction (−37.2% median) is the robust Q2
number. The heavy-eviction result (§0.6 E4a) is unaffected: its gain comes from fewer bytes read per query (−22%),
which the page counts explain.

**CloudWatch, E6 vs E6r** (SUM for deltas, Max for gauges, Avg for RTT / KB-per-op, hit rate derived):

| Metric | E6 base | E6 cand | E6 Δ | E6r base | E6r cand | E6r Δ |
|---|---|---|---|---|---|---|
| **EFS.ReadBytes** | 61.745 GB | 61.462 GB | **−0.5%** | 61.745 GB | 61.462 GB | **−0.5%** |
| EFS.ReadOperations | 1.882M | 1.873M | −0.4% | 1.882M | 1.874M | −0.4% |
| EFS.ReadKBPerOp (avg) | 32.08 | 32.04 | −0.1% | 32.05 | 32.04 | 0.0% |
| EFS bytes per query | 6.174 MB | 6.146 MB | −0.5% | 6.174 MB | 6.146 MB | −0.5% |
| EFS.AverageRTT (avg, ms) | 4.20 | 4.43 | +5.5% | 4.19 | 4.45 | +6.2% |
| Cache.Misses | 3.214M | 3.236M | +0.7% | 3.141M | 3.225M | +2.7% |
| Cache.Loads | 1.879M | 1.870M | −0.5% | 1.838M | 1.870M | +1.7% |
| Cache.Hits | 920.6K | 1.021M | +10.9% | 901.2K | 1.021M | +13.3% |
| Derived hit rate | 22.3% | 24.0% | +1.7 pp | 22.3% | 24.1% | +1.8 pp |
| Cache.Evictions | 0 | 0 | — | 0 | 0 | — |
| Allocation.BytesDelta | 61.567 GB | 61.285 GB | −0.5% | 60.231 GB | 61.285 GB | +1.7% |
| WssRecentMB (max) | 26.49K | 27.31K | +3.1% | 26.52K | 27.22K | +2.6% |
| Prefetch.Calls (per query) | 826.2K (82.6) | 928.8K (92.9) | +12.4% | 807.2K (80.7) | 929.2K (92.9) | +15.1% |
| Prefetch.BlocksLoaded | 759.6K | 738.7K | −2.8% | 745.4K | 725.5K | −2.7% |
| **Prefetch.BlocksCacheHit** | 75.5K | 197.7K | **+162%** | 70.3K | 211.7K | **+201%** |
| Throttle / OOM | 0 / 0 | 0 / 0 | — | 0 / 0 | 0 / 0 | — |
| KNN.AnnLatencyCount (sum) | 1.419M | 1.543M | +8.7% | 1.648M | 1.645M | −0.2% |

EFS bytes are identical across the two runs to the byte: the cold-pass working set is as deterministic as the page
counts.

**Why CloudWatch shows −0.5% while PageTouch shows −37% per query:**
- CloudWatch counts each 32 KB block at most once per worker (first touch; with no evictions nothing is loaded twice).
- Over 10,000 distinct queries the union of blocks touched grows to essentially the whole index on both layouts
  (~61.5 GB per fleet), so the totals measure index size, not per-query work.
- PageTouch counts the distinct pages each query touches; most were already loaded by an earlier query, so they add
  no EFS reads in a single pass.
- The per-query saving is visible in CloudWatch as **more reuse within the pass** (prefetches landing on resident
  blocks ×2.6–3.0, cache hits +11–13%, prefetch blocks loaded −3%), and as **fewer bytes whenever blocks are
  re-read**: −22.3% EFS bytes and −29% evictions per successful query under sustained eviction (§0.6 E4a), and
  −19.8% / −24.2% EFS bytes for single cold queries (§0.2, §0.5).
- `KNN.AnnLatencyCount` is inconsistent (+8.7% in E6, −0.2% in E6r with identical request counts): not used.

---

## 1. Latency — p50 / p90 / p99 (baseline → candidate)

| Scale (dim) | OCU | rescore | query set / cache | p50 (ms) | p90 (ms) | p99 (ms) | p100 (ms) | notes |
|---|---|---|---|---|---|---|---|---|
| **1M** (768) | 32 | on | OSB prod-queries (cache-resident) | 25.9 → 26.1 | 35.6 → 36.0 | 59.3 → 59.1 | 2354 → 1990 | flat — fully cached, no headroom |
| **1M** (768) | 32 | on | 100 q × 10 loop (server `took`) | 11 → 8 (**−27%**) | 36 → 25 (**−31%**) | 248 → 149 (**−40%**) | — | tail win where working-set misses appear |
| **10M** (768) | 32 | on | 100 q × 10 loop (server `took`) | 14 → 11 (−21%) | 83 → 83 (~0) | 593 → 603 (+2%) | — | muted — cache-resident at 32-OCU |
| **10M** (768) | 32 | on | full 10k prod-queries | 34.6 → 33.0 (−5%) | 45.1 → 45.9 (~0) | 121.8 → 171.5 (**+41%**) | 3127 → 2958 | p99 worse (noise; WSS only ~5%) |
| **100M** (1024) | 128 | on | full 10k, E2E client | 47.0 → 46.3 (−1.4%) | 82.6 → 65.6 (**−20.6%**) | 300.3 → 296.9 (−1.1%) | 15159 → 3202 (**−78.9%**) | tail win; p50/p99 masked by rescore |
| **113M** (1024) | 32→48 | on | OSB prod-queries-cached (**warm, trustworthy**) | 50.0 → 50.5 (flat) | 65.8 → 61.2 (−7%) | 290.8 → **185.0 (−36%)** | 3155 → **408 (−87%)** | +14% throughput at iso-recall |
| 113M (1024) | 0/128 | on | OSB prod-queries (**cold — NOT trustworthy**) | 86.8 → 129.0 | 410.9 → 597.2 | 2034.9 → 1977.4 | 8636 → 7642 | ⚠ flush-buffer-pool failed → not cold-clean; **discard** |
| 113M (1024) | — | **off** | single-query ×20, warm | ~100 → ~84 (**−19%**) | *not measured* | *not measured* | — | rescore-off isolates graph path |

**Reading the latency picture:** the win is at the **tail and under cache pressure/scale**, and
**muted when the working set is cache-resident** (1M/10M @32-OCU). Best trustworthy results:
**113M warm p99 −36% / p100 −87% / throughput +14%** (rescore on, iso-recall) and **100M p90 −20.6%
/ p100 −78.9%**. At 1024-dim, rescore-on reads (layout-invariant) dominate p50/median and *mask* the
graph-read win there — rescore-off exposes it (113M single-query p50 −19%).

---

## 1a. Steady-state, rescore OFF — latency + read-amp across all 4 scales

The cleanest single sweep isolating the graph read path (`benchmarks/steady_state_results.md`).
Protocol: per index, scale OCU 0 → target, lazy shard placement, then **10 unique queries looped
serially for 600 s, rescore OFF**, baseline (`target_field`) and candidate (`reordering_target_field`)
run **simultaneously**. Latency = client-side E2E; the 5 metrics = per-minute **Maximum** (peak).
k=100, size=100, 1 serial client. This is the rescore-OFF / cache-pressured regime §3 flagged as missing.

| Index | scale (dim) | OCU | p50 (base→cand) | p90 | p99 | QPS Δ | BytesDelta | Cache.Loads | WssRecentMB |
|---|---|---|---|---|---|---|---|---|---|
| target_index | 1M (768) | 32 | 20.1 → 20.2 (~flat) | 23.2 → 23.3 | 33.0 → 37.6 | ~flat | +21% (worse) | +17% | +10% |
| target_index_10_shuffled | 10M (768) | 32 | 25.3 → 21.7 (**−14%**) | 28.9 → 25.5 (**−12%**) | 36.7 → 38.8 (+6%) | **+13%** | −2.8% | +0.2% | +4.5% |
| target_index_100m | 100M (1024) | 96 | 29.8 → 27.0 (**−10%**) | 34.3 → 31.4 (**−9%**) | 46.6 → 65.3 (tail noise) | **+12%** | **−10.5%** | **−13.4%** | **−13.5%** |
| target_index_113m | 113M (1024) | 96 | 28.8 → 27.1 (**−6%**) | 34.7 → 34.1 (−2%) | 53.7 → 81.5 (tail noise) | +2% | **−22.6%** | **−16.8%** | **−13.5%** |

**The benefit grows monotonically with scale / cache pressure, as designed:**
- **1M** — cache-resident: no benefit (candidate marginally worse, all noise; layout can't help when the working set fits).
- **10M** — clear **latency** win (p50 −14%, p90 −12%, QPS +13%); peak node metrics flat (only 10 repeated queries → small, similar distinct-page working set, so the win lands in per-query latency not peak read volume).
- **100M** — win in **both** latency (p50 −10%, QPS +12%) **and** all read-amp metrics (bytes −10.5%, loads −13.4%, WSS −13.5%, hit-rate +3.6 pp). Cleanest simultaneous latency+read-amp win.
- **113M** — strongest **read-amp** cut (bytes −22.6%, loads −16.8%, WSS −13.5%); latency win modest (p50 −6%).

**Round 2 — 100M @96 OCU, correct SUM aggregation + EFS reads** (`benchmarks/steady_state_results_round2.md`;
query body adds OSB fast-fetch `_source:false, docvalue_fields:[id], stored_fields:_none_`; prefetch-confounded
like round 1):

| Metric | Agg | Baseline | Candidate | Δ |
|---|---|---|---|---|
| p50 / p90 | — | 27.6 / 32.3 ms | 26.7 / 31.4 ms | −3% / −3% |
| Cache.Misses | SUM | 2.99M | 2.31M | **−22.8%** |
| Allocation.BytesDelta | SUM | 48.29 GB | 41.57 GB | −13.9% |
| Cache.Loads | SUM | 1.47M | 1.27M | −13.9% |
| HitRate (derived) | — | 59.56% | 63.72% | +4.2 pp |
| WssRecentMB | Max | 7,128 | 6,196 | −13.1% |
| **EFS.ReadOperations / ReadBytes** | SUM | 1.48M / 48.41 GB | 1.28M / 42.05 GB | **−13.1% / −13.1%** |
| EFS.ReadKBPerOp | avg | 31.86 KB | 31.45 KB | ≈32 KB page ✓ |
| PageTouch distinct 32 KB pages/query | avg | 485 | 305 | **−37%** (p90 −45%) |

Read-amp win is clear in both node reads and PageTouch; latency only −3% because 10 repeated queries are served warm.

Caveats: p99/p100 tails are noisy (10-query loop; p100 is the cold first query), so read p50/p90/mean as the stable latency signal. 113M `Cache.HitRate` peak (100%→54%) is a Maximum-of-rate artifact (baseline had one 100%-hit minute) — ignore. Recall not measured here (rescore off); parity established in §3 / `recall_across_datasets.md`.

---

## 2. The 5 reordering-benefit metrics (per scale)

| Scale (dim) | measurement regime | Allocation.BytesDelta | Cache.Misses | Cache.Loads | WssRecentMB | Prefetch.BlocksLoaded |
|---|---|---|---|---|---|---|
| **1M** (768) | node-cumulative, cache-resident (weak) | *not emitted* | −15% (10,048→8,530) | −13% (6,759→5,881) | ~1% (too coarse) | *not emitted* |
| **10M** (768) | controlled cold, per-query (÷N) | 253.0 → 194.4 MB (**−23.2%**) | 12,042 → 9,347 (−22.4%) | 7,722 → 5,932 (−23.2%) | 989 → 779 MB (−21.2%) | 4,579 → 3,506 (−23.4%) |
| **100M** (1024) | per-min Maximum (peak/worker), rescore ON | 4.26 → 3.54 GB (**−17.0%**) | 216,222 → 153,549 (−29.0%) | 130,132 → 100,539 (−22.7%) | 36,251 → 29,730 MB (−18.0%) | 103,166 → 82,410 (−20.1%) |
| **113M** (1024) | rescore OFF, warm single-query (run 4) | 74.55 → 26.01 MB (**−65.1%**) | 3,479 → 1,566 (−55.0%) | 2,277 → 953 (−58.1%) | 4,479 → 2,763 MB (−38.3%) | 1,318 → 387 (−70.7%) |
| **113M** (1024) | rescore OFF, reproducible cold (runs 5/6) | −19 to −21% | −15 to −17% | −19 to −21% | −27% | −31 to −34% |

Extras: **Cache.Evictions** — 100M candidate **83,844 → 0** (working set fit without eviction
pressure); Cache.HitRate roughly flat/slightly + at 10M/100M, ~−3 pp at 113M (noise at low cold hit
rates). At 100M the **Sum-over-window** aggregate looks flat (−0.5%) because rescore bytes (4096 B,
layout-invariant) swamp the total — the **peak (Maximum)** view above is where the graph-read win
shows.

**Consistent story:** candidate loads ~17–65% less data per query (magnitude scales with cache
warmth: ~20% on a fresh cold fault → ~65% warmer), with fewer misses/loads, smaller working set, and
fewer prefetch blocks — on both natural and shuffled data, so it's a codec-layout property. Corroborated
by PageTouch graph read-amp −28–42% (−36% @100M).

---

## 3. Gaps — latencies we do NOT have (clean)

Most rescore-OFF/pressured gaps below are now **filled by the steady-state sweep (§1a)**; remaining
weak cells noted:

1. ~~**1M & 10M cold/pressured latency**~~ — **FILLED (§1a):** steady-state rescore-OFF p50/p90/p99 at
   32-OCU. 1M confirmed cache-resident (flat); 10M shows the latency win (p50 −14%, QPS +13%).
2. **113M cold latency** — the OSB `prod-queries` (cold) run is **not trustworthy** (flush-buffer-pool
   AccessDenied → not guaranteed cold); discard those percentiles. Only the *warm* p50/p90/p99 are good.
   (Steady-state §1a gives a clean rescore-OFF profile instead: 113M p50 −6%, bytes −22.6%.)
3. ~~**113M rescore-OFF p90/p99**~~ — **FILLED (§1a):** p50 −6% / p90 −2% / p99 tail-noisy; read-amp
   bytes −22.6%, loads −16.8%, WSS −13.5%.
4. ~~**100M rescore-OFF latency + 5 metrics**~~ — **FILLED (§1a):** rescore-OFF p50 −10%, QPS +12%, and
   the peak metrics (bytes −10.5%, loads −13.4%, WSS −13.5%, hit-rate +3.6 pp) — the cleanest
   simultaneous latency+read-amp win. (100M rescore-OFF on the *Sum* view still not captured.)
5. **1M Allocation.BytesDelta & Prefetch.BlocksLoaded** — block-IO counters not emitted for the 1M
   test PCs; only Misses/Loads (directional) available.
6. **Recall under rescore-OFF steady-state** — not measured in §1a (latency/metrics focus); parity
   established separately (recall table below / `recall_across_datasets.md`).

**Remaining gaps after the clean A/B (§0), in priority order:**

7. **Cache-pressured but healthy regime** — **accepted as a stress result (§0.6 E4a, 16 OCU, 17–28% errors)**; a
   < 1%-error eviction regime was not reached (24 OCU is not a valid setting; E4b/E4c descoped). Original gap: the only clean cache-pressure data is 16 OCU at 74–88% errors (§0.1).
   Need 100M at 24–32 OCU, a few hundred–1k distinct queries looped, working set > cache (evictions > 0) but errors
   < 1%, paced multi-client load. This is the production-representative latency/throughput number.
8. **Clean coverage beyond 100M** — 10M and 113M are still prefetch-confounded. Repeat §0.2 and gap 7 at 113M (and 10M)
   to get an unconfounded benefit-vs-scale curve.
9. **Shipping config (prefetch ON + rescore ON)** — **prefetch ON closed (§0.6: all four-question runs are prefetch ON)**;
   rescore ON still open. Earlier status — partly closed (§0.5): prefetch-on reads −24% for one cold
   query. Still open: a prefetch-on *latency* run without stack logging, a 2×2 (stock/reordered × prefetch off/on),
   and a clean **rescore-ON** run (production default; rescore reads are layout-invariant and masked p50 before).
10. **Per-query vs aggregate saving (Q2)** — **CLOSED (§0.7 E6):** median query −37.2% distinct 32 KB pages
    (p10 −27.2%, p90 −47.7%), all 10,000 queries improved, with 0 log gaps. Total EFS stays at parity in a single
    pass because the whole working set is loaded once either way. Original gap: one cold query: −20% EFS; 100 distinct cold queries (4a): −3.7%. Enable
    PageTouch on multi-query runs and report its per-query distribution beside EFS totals to explain the gap
    (likely shared hub/dense-region pages across queries).
11. **Replica-routing noise** — pin `?preference=<fixed>` (or 1 replica) so small-n latency comparisons are deterministic.
12. **Cost framing** — OCU sweep (16/24/32/48) at fixed paced load: lowest OCU meeting a p99 target per fleet.
13. **Sibling-greedy ordering on a cluster** — sim −16% pages/query vs shipped greedy; implemented behind
    `-Dknn.locality.ordering=sibling`, not yet cluster-tested. §0.5 suggests this is exactly the missing headroom
    (most neighbors still land in different blocks).
14. **Steady-state / in-memory latency (Q3)** — **resolved as "no slowdown" (§0.8):** the −17/−18% warm-latency gaps
    did not reproduce on new hosts (E6r −3%). To claim an in-memory speed-up would need several repeats on the
    same hosts (or a CPU profile); not required for the design doc.

**Headline (current build, prefetch ON, §0.6–§0.7):** at 100M, across all 10,000 queries the median query touches
**37% fewer distinct 32 KB pages** (every query improved; read-amp 141× → 82×) at identical recall; under heavy
eviction that becomes **+50% successful queries and −37% p99**, with −22% bytes read per successful query.

**Cleanest single "the feature works" statement (unconfounded, §0):** at 100M with prefetch off on both fleets,
reordering alone cuts EFS reads **~20% per cold query** (PageTouch distinct pages −28%), cold latency **−21.5% p50 /
−30% mean**, and under cache pressure (16 OCU) delivers **+31–43% successful queries and −23–39% p90/p99** at recall
parity. It gives ~0 when the cache holds the full union of touched pages (10k sweep, 0 evictions).

**Earlier (prefetch-confounded) statement:** at 113M, warm, iso-recall
(0.95/0.97) — **p99 −36%, p100 −87%, throughput +14%**, backed by graph read-amp −28% and
per-query bytes −55% (rescore off). At 100M — **p90 −20.6%, p100 −78.9%**, peak read-pressure −17–29%.
Now corroborated by the **steady-state rescore-OFF sweep (§1a)**: 100M p50 −10% / QPS +12% with
bytes −10.5% / loads −13.4% / WSS −13.5% *simultaneously*, and 113M bytes −22.6% / loads −16.8% —
the benefit growing monotonically with scale (none at 1M → −14% p50 at 10M → read-amp win at 100M/113M).

### Gap-closing run — DONE (§1a)
The recommended multi-scale, `rescore=false`, steady-state, nonzero-OCU run was executed (§1a,
`benchmarks/steady_state_results.md`): 10 unique queries × 600 s at 1M/10M/100M/113M, baseline and
candidate simultaneously, capturing clean p50/p90 *and* the peak node metrics on the graph path.
Still open: a ≥ few-hundred-distinct-query variant (to firm up p99/tail) and the 100M rescore-OFF *Sum*-view metrics.
