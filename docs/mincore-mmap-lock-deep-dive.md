# Design: Tracking Resident Memory for Memory Optimized Search

## 1. Problem Statement

The k-NN plugin's stats API (`/_plugins/_knn/stats`) reports `graph_memory_usage` for graphs
loaded via the traditional path (JNI + `NativeMemoryCacheManager`). This stat tells customers
exactly how much RAM their graphs consume, enabling capacity planning and troubleshooting.

The **Memory Optimized Search** path uses `mmap` (via Lucene's `MMapDirectory` /
`MemorySegment`) to load FAISS graphs. This path is completely invisible to the current stats
API. Customers using Memory Optimized Search have no way to know:
- How much graph data is loaded
- Whether their graphs are fully resident in physical RAM
- Whether they are running low on memory (graphs being evicted by the OS)

**Goal**: Provide customers with memory usage stats for the Memory Optimized Search path,
analogous to what `graph_memory_usage` provides for the traditional path.

---

## 2. Background

### 2.1 How the Traditional Path Tracks Memory

The traditional path explicitly manages memory:

1. `NativeMemoryCacheManager.get()` → JNI `loadIndex()` → `malloc()` in native code
2. The entire graph is copied into off-heap native memory
3. `IndexAllocation.getSizeInKB()` records the exact allocation size
4. `NativeMemoryCacheManager.getIndicesSizeInKilobytes()` sums all allocations

Every byte allocated is a byte of physical RAM consumed. `graph_memory_usage` = allocated
size = actual RAM usage. The relationship is 1:1.

### 2.2 How the Memory Optimized Path Loads Graphs

The Memory Optimized path delegates memory management to the OS:

1. `FaissMemoryOptimizedSearcher` opens an `IndexInput` backed by `MMapDirectory`
2. `FaissIndex.load(IndexInput)` parses the FAISS file structure, recording section offsets
   via `FaissSection` objects — but does NOT read all data into memory
3. During search, `FaissHnswGraph` reads neighbor lists on-demand via `indexInput.seek()` +
   `indexInput.readInt()` — the OS pages in data as needed
4. Warmup (`WarmupUtil.readAll()`) sequentially reads the entire file, causing the OS to page
   all data into RAM

Key differences from the traditional path:
- **No explicit allocation** — the OS manages physical memory via the page cache
- **No cache** — `VectorSearcherHolder` is per-segment, not a shared cache like
  `NativeMemoryCacheManager`
- **No size tracking** — `FaissSection.sectionSize` exists but is never aggregated
- **No eviction control** — the OS can silently evict pages under memory pressure

### 2.3 The Three Layers of "Memory" in mmap

| Layer | What it means | Can we track it? |
|-------|--------------|-----------------|
| **Virtual (mapped) size** | Total file size of all mmap'd graph files. The address space reserved. | Yes — known at load time from `IndexInput.length()` |
| **Resident size** | How many pages are actually in physical RAM right now. | Only via kernel queries that take `mmap_lock` |
| **OS page cache** | System-wide cache shared across all processes. | Not trackable per-file from userspace |

After warmup, mapped size ≈ resident size. Under memory pressure, resident size drops below
mapped size as the OS evicts pages. The gap between mapped and resident is the signal that the
customer needs more RAM.

---

## 3. Solutions Considered

### Quick Comparison

| Option | Approach | Tracks Resident? | `mmap_lock` Free? | Overhead | Kernel Req | Assessment |
|--------|----------|:-:|:-:|----------|------------|:--:|
| **A** | `mincore()` via JNI | ✅ Yes | ❌ No | High — page table walk per graph | Any | ❌ |
| **B** | `/proc/self/smaps` | ✅ Yes | ❌ No | Very high — walks ALL mappings | Any | ❌ |
| **C** | `mincore()` via Panama FFM | ✅ Yes | ❌ No | High — same as A | Any | ❌ |
| **D** | `mlock()` + tracked size | ✅ Guaranteed | N/A | Low | Any | ❌ |
| **E** | Major page fault counter | ⚠️ Indirect | ✅ Yes | Zero | Any | ⚠️ |
| **F** | Search-path latency | ⚠️ Indirect | ✅ Yes | Zero | Any | ⚠️ |
| **G** | Mapped size tracking | ❌ No | ✅ Yes | Zero | Any | ✅ Primary |
| **H** | `cachestat()` syscall | ✅ Yes | ✅ Yes | Low — RCU xarray walk | Linux 6.5+ | ⚠️ Future |

**Recommended path**: Implement **G** now (mapped size — zero overhead, same contract as traditional path). Pursue **H** as a future enhancement when Linux 6.5+ adoption is sufficient (the only lock-free approach that provides actual resident size).

---

### 3.1 Option A: `mincore()` via JNI — Exact Resident Size

**Approach**: Use the `mincore()` Linux syscall to query which pages of each mmap'd graph file
are resident in RAM. The k-NN plugin already extracts raw `MemorySegment` addresses via
`AbstractMemorySegmentAddressExtractor`. Pass those addresses to a native `mincore()` call,
count resident pages × page size = resident bytes.

**Implementation**:
- Add a new JNI method (small — `mincore` is a single syscall, ~30 lines of C)
- For each active `FaissMemoryOptimizedSearcher`, extract addresses via
  `MemorySegmentAddressExtractorUtil`, call `mincore()`, count resident pages
- Register new stats: `memory_optimized_graph_resident_size_kb`,
  `memory_optimized_graph_resident_percentage`

**Pros**:
- Exact, per-graph resident size — the only approach that truly answers "how much is in RAM"
- Existing infrastructure: address extraction (`AbstractMemorySegmentAddressExtractor`) and
  JNI build system (`jni/`, CMake) are already in place
- Gives the exact answer customers want: "62% of your graph data is resident"

**Cons — `mmap_lock` contention (CRITICAL)**:

`mincore()` acquires the process-wide `mmap_lock` in read mode. This lock contends with ALL
virtual address space modifications across the entire process. See
[Appendix A](#appendix-a-deep-dive-mincore-mmap_lock-and-performance-impact) for the full
kernel-level analysis. The key problems:

1. **Cross-file contention**: Calling `mincore()` on graph file A blocks `mmap()`/`munmap()`
   of completely unrelated file B. Lucene continuously opens/closes segment files during
   merges and refreshes — all of these would contend with `mincore()`.
2. **Convoy effect**: A pending `mmap_write_lock` (from segment open/close) blocks ALL
   subsequent `mmap_read_lock` attempts, including page faults from search threads. One
   `mincore()` call can cascade into stalls across all threads.
3. **Scales with graph size**: A 10GB graph requires ~640 lock/unlock cycles, each walking
   page tables. Estimated ~10ms per graph, but the real cost is the cascading impact.
4. **Even periodic calls cause stalls**: Reducing frequency doesn't eliminate the problem —
   when a periodic `mincore()` call coincides with a segment merge, the stall still occurs.
   On a busy node with continuous merges, collisions are inevitable.

**Assessment**: ❌ Rejected. The `mmap_lock` contention makes this unsuitable for production use
in a system that continuously opens and closes mmap'd segments.

---

### 3.2 Option B: `/proc/self/smaps` Parsing — Resident Size Without JNI

**Approach**: Read `/proc/self/smaps`, which lists every memory mapping with its `Rss`
(resident set size). Match address ranges to known graph file addresses.

**Pros**:
- No JNI needed — pure Java file I/O
- Kernel provides pre-aggregated Rss per mapping

**Cons**:
- **Same `mmap_lock` problem as `mincore()`** — reading `smaps` acquires `mmap_read_lock`
- **Actually worse than `mincore()`** — `smaps` walks page tables for EVERY mapping in the
  process, not just the graphs. A process with thousands of Lucene segments means walking
  thousands of VMAs.
- Linux-only (no `/proc` on macOS)
- Parsing is fragile — format could change across kernel versions
- Lucene splits files into multiple mmap chunks (up to 1GB each), requiring correlation of
  multiple `smaps` entries back to a single graph file
- The kernel per-VMA lock patches (as of mid-2025) explicitly do NOT apply to `smaps`:
  *"similar approach would not work for /proc/pid/smaps reading as it also walks the page
  table and that's not RCU-safe."*

**Assessment**: ❌ Rejected. Same fundamental `mmap_lock` problem as `mincore()`, with additional
downsides.

---

### 3.3 Option C: `mincore()` via Panama FFM — Resident Size Without JNI/C Code

**Approach**: Same as Option A, but use Java 21's Foreign Function & Memory API
(`java.lang.foreign.Linker`) to call `mincore()` directly instead of writing JNI code.

**Pros**:
- No C code to write or maintain
- Cleaner than JNI — the `MemorySegment` addresses are already in Java

**Cons**:
- **Same `mmap_lock` problem** — the syscall is identical regardless of how it's invoked
- FFM is preview in Java 21, standard in Java 22 — depends on minimum JDK target
- Requires `--enable-native-access` JVM flag
- Linux/macOS only for `mincore`

**Assessment**: ❌ Rejected. Cleaner invocation mechanism, but the fundamental `mmap_lock`
contention is unchanged.

---

### 3.4 Option D: `mlock()` + Tracked Size — Guaranteed Residency

**Approach**: Use `mlock()` to pin graph pages in RAM. Once mlock'd, pages are guaranteed
resident, so mapped size == resident size with certainty. Track mapped size at load time.

**Pros**:
- Mapped size becomes a reliable indicator of resident size (100% guaranteed)
- No periodic queries needed

**Cons**:
- Requires `CAP_IPC_LOCK` privilege or sufficient `RLIMIT_MEMLOCK`
- Defeats the OS's ability to manage memory under pressure — the whole point of mmap is to
  let the OS make intelligent eviction decisions
- If the node doesn't have enough RAM for all mlock'd graphs, `mlock()` fails or the OOM
  killer is invoked — worse than graceful degradation via page eviction
- Fundamentally changes the memory management model of the Memory Optimized path

**Assessment**: ❌ Rejected. Contradicts the design philosophy of the Memory Optimized path.

---

### 3.5 Option E: Major Page Fault Counter — Detect Eviction by Symptom

**Approach**: Instead of measuring resident size, detect when graphs are NOT resident by
monitoring major page faults. Read `/proc/self/stat` field 12 (`majflt`), which is a simple
counter read with no `mmap_lock` involvement.

**Pros**:
- Zero lock contention — `/proc/self/stat` reads task struct counters, no page table walk
- Detects the actual problem (page faults = latency impact) rather than a proxy
- A rising major page fault count during searches is a direct signal: "your graphs are being
  evicted, you need more RAM"

**Cons**:
- Process-wide counter — cannot distinguish graph page faults from other page faults (though
  in a k-NN-heavy workload, graph faults would dominate)
- Indirect signal — tells you "something is wrong" but not "how much is missing"
- Customers expect a memory usage number (like `graph_memory_usage`), not a fault counter
- Does not provide the familiar mental model of "X KB in memory"

**Assessment**: ⚠️ Useful as a supplementary health signal, but insufficient as the primary stat.
Customers are accustomed to `graph_memory_usage` reporting a concrete memory number.

---

### 3.6 Option F: Search-Path Latency Tracking — Detect Eviction by Performance

**Approach**: Instrument `FaissMemoryOptimizedSearcher.search()` to measure I/O latency. Fully
resident mmap reads complete in nanoseconds; page faults take milliseconds (10,000x
difference). Detect eviction by observing search latency anomalies.

**Pros**:
- Zero syscalls, zero lock contention — pure application-level measurement
- Detects the actual user-visible impact (slow searches)
- Already partially exists via `ProfileMemoryOptKNNWeight`

**Cons**:
- Indirect — measures the symptom, not the cause
- Cannot distinguish "slow because of page faults" from "slow because of other I/O"
- Does not provide a memory usage number
- Same customer expectation problem as Option E

**Assessment**: ⚠️ Useful for profiling, but not a replacement for memory usage stats.

---

### 3.7 Option G: Mapped Size as Memory Usage — Application-Level Tracking

**Approach**: Track the total mapped size of all mmap'd graph files at load time. Report this
as `memory_optimized_graph_memory_usage`. This is the virtual footprint — the total size of
graph data that has been mmap'd.

**Implementation**:
- Create a lightweight `MemoryOptimizedSearcherTracker` singleton (a `ConcurrentHashMap`)
- When `FaissMemoryOptimizedSearcher` is created, register with file size
- When closed, deregister
- New suppliers read from the tracker, following the `NativeMemoryCacheManagerSupplier` pattern
- Register stats in `KNNStats.buildStatsMap()`

**Stats exposed**:

| Stat name | Value | Purpose |
|-----------|-------|---------|
| `memory_optimized_graph_memory_usage` | Sum of all mmap'd graph file sizes (KB) | Total memory footprint |
| `memory_optimized_graph_count` | Number of active mmap'd graphs | Scale indicator |
| `memory_optimized_indices_graph_stats` | Per-index breakdown (count + size) | Per-index visibility |

**Pros**:
- Zero overhead — size is known at load time, no syscalls, no locks
- Follows the existing stats API pattern exactly
- Provides the familiar mental model: "X KB of graphs loaded"
- Semantically equivalent to what `graph_memory_usage` reports for the traditional path —
  both report "how much graph data has been loaded," not "how much is guaranteed to be in
  physical RAM at this instant" (the traditional path also doesn't check if the OS has
  swapped malloc'd pages to disk)

**Cons**:
- Reports mapped size, not resident size — on a memory-constrained node, the actual RAM
  usage could be lower than reported
- Cannot tell the customer "your graphs are being evicted" — the number stays the same
  whether 100% or 50% of pages are resident
- Customers may assume this means "in RAM" when it really means "mapped"

**Assessment**: ✅ **Recommended as the primary stat.** It provides the same contract as the
traditional path's `graph_memory_usage` (allocated/mapped size), is zero-overhead, and fits
the existing API shape. Documentation should clearly state this represents the mapped
footprint.

---

### 3.8 Option H: `cachestat()` Syscall — Lock-Free Resident Size (Linux 6.5+)

**Approach**: Use the `cachestat()` syscall (introduced in Linux 6.5) to query page cache
statistics for each mmap'd graph file. Unlike `mincore()`, `cachestat()` operates on the
file's `address_space` (via file descriptor) rather than the process's virtual address space,
and uses only an RCU read lock — **completely bypassing `mmap_lock`**.

**How it works**:

```c
// Syscall signature
int cachestat(unsigned int fd,
              struct cachestat_range *range,  // { off, len }
              struct cachestat *stat,          // output
              unsigned int flags);

// Output structure
struct cachestat {
    __u64 nr_cache;             // Pages currently in page cache (= resident)
    __u64 nr_dirty;             // Dirty pages
    __u64 nr_writeback;         // Pages being written back
    __u64 nr_evicted;           // Pages previously in cache, now evicted
    __u64 nr_recently_evicted;  // Evicted pages whose reentry would indicate memory pressure
};
```

**Kernel implementation** (from `mm/filemap.c`, Linux 6.8):

```c
static void filemap_cachestat(struct address_space *mapping,
        pgoff_t first_index, pgoff_t last_index, struct cachestat *cs)
{
    XA_STATE(xas, &mapping->i_pages, first_index);
    struct folio *folio;

    rcu_read_lock();                              // <-- RCU only, NO mmap_lock
    xas_for_each(&xas, folio, last_index) {
        // Walk the xarray (radix tree) of the file's page cache
        // Check folio flags — no page table walk needed
        if (xa_is_value(folio)) {
            cs->nr_evicted += nr_pages;           // Shadow entry = evicted page
            if (workingset_test_recent(...))
                cs->nr_recently_evicted += nr_pages;
        } else {
            cs->nr_cache += nr_pages;             // Folio in cache = resident
            if (xas_get_mark(&xas, PAGECACHE_TAG_DIRTY))
                cs->nr_dirty += nr_pages;
            if (xas_get_mark(&xas, PAGECACHE_TAG_WRITEBACK))
                cs->nr_writeback += nr_pages;
        }
    }
    rcu_read_unlock();
}

SYSCALL_DEFINE4(cachestat, unsigned int, fd, ...)
{
    mapping = f.file->f_mapping;                  // File's address_space
    filemap_cachestat(mapping, first_index, last_index, &cs);
    // No mmap_lock anywhere in the call chain
}
```

**Why this avoids the `mmap_lock` problem**:

`mincore()` answers "which pages in my process's virtual address space are resident?" — this
requires walking the process's page tables, which requires `mmap_lock` to stabilize the VMA
tree.

`cachestat()` answers "which pages of this file are in the page cache?" — this walks the
file's `address_space` xarray (radix tree), which is a per-file data structure completely
independent of any process's VMA tree. It uses only `rcu_read_lock()`, which is essentially
lock-free for readers and has zero contention with `mmap()`/`munmap()`.

| | `mincore()` | `cachestat()` |
|---|---|---|
| **Lock** | `mmap_read_lock` (process-wide r/w semaphore) | `rcu_read_lock()` (lock-free for readers) |
| **Data structure walked** | Process page tables (per-process VMA tree) | File page cache xarray (per-file `address_space`) |
| **Cross-file contention** | Yes — blocks `mmap()`/`munmap()` of ANY file | **No** — per-file, independent of process address space |
| **Blocks segment open/close** | Yes | **No** |
| **Input** | Memory address + length | File descriptor + byte range |
| **Output** | Per-page resident/not-resident bitmap | Aggregate: `nr_cache`, `nr_evicted`, `nr_recently_evicted`, `nr_dirty`, `nr_writeback` |

**Resident size calculation**:

For mmap'd files, pages in the page cache ARE the pages mapped into the process. The page
cache is the backing store for mmap. Therefore:

```
resident_size_bytes = cachestat.nr_cache × PAGE_SIZE (4096)
resident_percentage = nr_cache / total_pages × 100
```

**The `nr_evicted` and `nr_recently_evicted` fields** are a bonus — they directly answer
"are your graphs being evicted?" and "is the eviction due to active memory pressure?". This
is exactly the signal needed to tell customers they're running low on memory.

**Stats that could be exposed**:

| Stat name | Value | Purpose |
|-----------|-------|---------|
| `memory_optimized_graph_resident_size_kb` | `nr_cache × 4` summed across all graphs | Actual RAM used |
| `memory_optimized_graph_mapped_size_kb` | Total mapped size (from Option G) | Total demand |
| `memory_optimized_graph_resident_percentage` | resident / mapped × 100 | **The health signal** |
| `memory_optimized_graph_evicted_pages` | Sum of `nr_evicted` across all graphs | Eviction indicator |
| `memory_optimized_graph_recently_evicted_pages` | Sum of `nr_recently_evicted` | Active memory pressure indicator |
| `memory_optimized_graph_count` | Number of active mmap'd graphs | Scale indicator |

**Cons**:
- **Requires Linux 6.5+** — this is a new syscall added in August 2023. OpenSearch nodes
  running older kernels (e.g., Amazon Linux 2 with kernel 5.x) will not have it. This is
  the primary blocker — the stat would only be available on newer kernels, requiring
  graceful degradation (fall back to mapped-size-only on older kernels).
- **Requires JNI or Panama FFM to invoke from Java** — `cachestat()` is a syscall, not
  accessible via standard Java APIs. Needs either a small JNI wrapper or Java 22+ FFM.
  The k-NN plugin already has JNI infrastructure (`jni/`, CMake) so this is feasible.
- **Requires file descriptor access** — `cachestat()` takes an fd, but
  `FaissMemoryOptimizedSearcher` holds a Lucene `IndexInput`, not a raw fd. Options:
  - Extract the fd from Lucene's `MemorySegmentIndexInput` via reflection (similar to how
    `AbstractMemorySegmentAddressExtractor` already uses reflection to extract
    `MemorySegment[]`)
  - Open the graph file separately with a read-only fd just for stats queries
  - Obtain the fd at searcher creation time and store it
- **Point-in-time snapshot** — like `mincore()`, page status can change after the call
  returns. The man page explicitly notes: *"the returned values may contain stale
  information."* However, this is inherent to any residency query.
- **`nr_cache` counts pages in the page cache, not pages mapped into the process** — for
  mmap'd files these are the same pages (the page cache IS the backing store for mmap),
  but this assumption should be validated for edge cases (e.g., files also being read via
  regular I/O, or files shared across multiple OpenSearch processes).
- **hugetlbfs not supported** — `cachestat()` returns `-EOPNOTSUPP` for hugetlbfs files.
  This is unlikely to affect k-NN graph files but should be noted.

**Open questions**:
1. **Kernel adoption**: What Linux kernel versions do OpenSearch customers typically run?
   If 6.5+ is common (e.g., Amazon Linux 2023 uses kernel 6.1, which does NOT have it;
   Amazon Linux 2023 with kernel 6.6+ would), this becomes viable. If most customers are
   on 5.x or early 6.x, this would be a best-effort stat available only on newer kernels.
2. **File descriptor extraction**: Can we reliably extract an fd from Lucene's
   `MemorySegmentIndexInput`? The `AbstractMemorySegmentAddressExtractor` already uses
   reflection to access private fields of `MemorySegmentIndexInput` — a similar approach
   could extract the underlying `FileChannel` and its fd. Alternatively, since we know the
   file path, we could open a separate read-only fd purely for `cachestat()` queries.
3. **Accuracy validation**: Need to verify empirically that for mmap'd files,
   `nr_cache × PAGE_SIZE` matches the actual resident size as reported by `mincore()` or
   `/proc/self/smaps`. Conceptually they should agree, but edge cases around folio sizes,
   readahead, and filesystem-specific behavior should be tested.
4. **Performance at scale**: While `cachestat()` is lock-free (RCU only), walking the xarray
   for a very large file (100GB+ = 25M+ entries) still takes CPU time. Need to benchmark
   the wall-clock cost and determine if periodic invocation (e.g., every 30-60 seconds) is
   acceptable.
5. **Graceful degradation**: If `cachestat()` is unavailable (older kernel), the syscall
   returns `-ENOSYS`. The implementation should detect this at startup and fall back to
   mapped-size-only stats (Option G) transparently.

**Assessment**: ⚠️ **Promising alternative that solves the `mmap_lock` problem.** If the kernel
version requirement can be met (or graceful degradation is acceptable), this is the only
known approach that provides actual resident memory tracking without impacting search or
indexing performance. Worth pursuing as an enhancement on top of Option G.

---

## 4. Recommendation

**Primary stats (implement now)**: Option G (Mapped Size as Memory Usage)

This provides the same level of information as the traditional path's `graph_memory_usage`.
Both paths report "how much graph data has been loaded" — the traditional path reports
malloc'd bytes, the Memory Optimized path reports mmap'd bytes. Neither path verifies
physical residency at query time.

**Resident size tracking (future enhancement)**: Option H (`cachestat()` Syscall)

If the kernel version requirement is acceptable, `cachestat()` is the only known mechanism
that provides actual resident memory tracking without `mmap_lock` contention. It should be
implemented as an enhancement on top of Option G, with graceful fallback to mapped-size-only
stats on kernels older than 6.5. The `nr_evicted` and `nr_recently_evicted` fields provide
the exact "you're running low on memory" signal that motivated this design.

**Why not `mincore()` or `/proc/self/smaps`**: Every mechanism that walks the process's page
tables acquires the process-wide `mmap_lock`, which contends with Lucene's continuous segment
open/close operations. This contention causes stalls in indexing and search, even when called
periodically. See Appendix A for the full analysis.

**Buffer Pool Based Memory Tracking**
As we already know the Index Level Encryption Plugin is already implementing a Buffer Pool where all the memory segments 
will loaded for a file after decryption. Now this gives us an advantage that if buffer pool is enabled then we track the 
memory for graph files since in that case we will know what pages are present in BufferPool. This check should be simple 
L1 cache check of buffer pool. But since buffer pool is not production-ize right now and might be applicable to all 
domains we still see a gap.


### The Fundamental Tradeoff

The Memory Optimized Search path chose `mmap` to avoid the overhead of explicit memory
management (no malloc, no cache, no circuit breaker). The tradeoff is that memory management
is delegated to the OS kernel. Querying residency via traditional mechanisms (`mincore`,
`smaps`) requires `mmap_lock`, which contends with the very mmap operations the path depends
on. The `cachestat()` syscall (Linux 6.5+) sidesteps this by querying the file's page cache
directly via RCU, but requires a modern kernel.

---

## 5. Future Considerations

- **`cachestat()` kernel adoption**: Track adoption of Linux 6.5+ across the OpenSearch
  customer base. As Amazon Linux 2023 and other distributions move to 6.5+ kernels,
  `cachestat()`-based resident tracking (Option H) becomes viable for a wider audience.
- **Linux kernel evolution**: The per-VMA lock work (Linux 6.x+) is actively reducing
  `mmap_lock` contention for various operations. If future kernels extend per-VMA locks to
  page table walks (currently not planned), `mincore()` could become viable.
- **`madvise(MADV_COLD)` / `MADV_PAGEOUT`**: These hints tell the OS to deprioritize pages
  but don't help with observability.
- **eBPF-based tracking**: A custom eBPF program could potentially track page-in/page-out
  events for specific file mappings without `mmap_lock`. This is a research direction, not
  a production-ready solution today.

---

## Appendix A: Deep Dive — `mincore()`, `mmap_lock`, and Performance Impact

### A.1 What is `mmap_lock`?

Every Linux process has a single `struct mm_struct` that describes its entire virtual address
space. Within this structure lives `mmap_lock` — a **process-wide read/write semaphore** that
guards the process's VMA (Virtual Memory Area) tree.

```
struct mm_struct {
    ...
    struct rw_semaphore mmap_lock;   // Guards the entire address space
    ...
};
```

Key properties:
- **One per process** — not per-file, not per-mapping, not per-thread. Every thread in the
  process shares the same `mmap_lock`.
- **Read/write semaphore** — multiple readers can hold it concurrently, but a writer requires
  exclusive access. A pending writer blocks new readers (to prevent writer starvation).
- **Guards all VMAs** — every `mmap()`, `munmap()`, `mprotect()`, `mremap()`, and `brk()` call
  must acquire this lock in **write** mode. Page faults and `mincore()` acquire it in **read**
  mode.

### Operations that acquire `mmap_lock`

| Operation | Lock mode | What it does |
|-----------|-----------|-------------|
| `mmap()` / opening a new file mapping | **Write** | Creates a new VMA in the address space |
| `munmap()` / closing a mapping | **Write** | Removes a VMA from the address space |
| `mprotect()` | **Write** | Modifies VMA protection flags |
| `mremap()` | **Write** | Moves/resizes a VMA |
| `brk()` | **Write** | Expands/shrinks the heap |
| Page fault handling | **Read** | Traverses VMAs to resolve a fault |
| `mincore()` | **Read** | Walks page tables to check residency |
| `/proc/self/smaps` | **Read** | Walks page tables for all mappings |

### A.2 How `mincore()` Uses `mmap_lock`

From the Linux kernel source (`mm/mincore.c`, kernel 6.8):

```c
SYSCALL_DEFINE3(mincore, unsigned long, start, size_t, len,
                unsigned char __user *, vec)
{
    ...
    while (pages) {
        mmap_read_lock(current->mm);          // <-- ACQUIRE read lock
        retval = do_mincore(start, min(pages, PAGE_SIZE), tmp);
        mmap_read_unlock(current->mm);        // <-- RELEASE read lock

        copy_to_user(vec, tmp, retval);
        pages -= retval;
        start += retval << PAGE_SHIFT;
    }
    ...
}
```

#### The chunked processing model

`mincore()` does NOT hold `mmap_lock` for the entire duration. It processes pages in chunks of
`PAGE_SIZE` result bytes (4096 bytes), where each byte represents one page. Since each byte
covers one 4KB page, each chunk covers:

```
4096 bytes × 4KB per page = 16 MB of virtual address space per chunk
```

For each chunk:
1. Acquire `mmap_read_lock`
2. Call `do_mincore()` → `walk_page_range()` → walks the page table
3. Release `mmap_read_lock`
4. Copy results to userspace

#### What happens inside `do_mincore()`

```c
static long do_mincore(unsigned long addr, unsigned long pages, unsigned char *vec)
{
    vma = vma_lookup(current->mm, addr);     // Find the VMA for this address
    ...
    err = walk_page_range(vma->vm_mm, addr, end, &mincore_walk_ops, vec);
    ...
}
```

The page walk callback `mincore_pte_range()` acquires an additional **PTE-level spinlock** for
each PMD range it processes:

```c
static int mincore_pte_range(pmd_t *pmd, unsigned long addr, unsigned long end,
                             struct mm_walk *walk)
{
    spinlock_t *ptl;
    ...
    ptep = pte_offset_map_lock(walk->mm, pmd, addr, &ptl);  // PTE spinlock
    for (; addr != end; ptep++, addr += PAGE_SIZE) {
        pte_t pte = ptep_get(ptep);
        if (pte_none_mostly(pte))
            __mincore_unmapped_range(addr, addr + PAGE_SIZE, vma, vec);
        else if (pte_present(pte))
            *vec = 1;                                         // Page is resident
        else { /* swap entry handling */ }
        vec++;
    }
    pte_unmap_unlock(ptep - 1, ptl);                          // Release PTE spinlock
    cond_resched();                                           // Yield CPU if needed
    return 0;
}
```

#### Summary of locks taken per chunk

```
For each 16MB chunk of the queried range:
  1. mmap_read_lock(mm)                    [process-wide read/write semaphore]
  2.   For each 2MB PMD range within the chunk:
  3.     pte_offset_map_lock(mm, pmd, ...)  [per-PTE-page spinlock]
  4.       Walk 512 PTEs, check present bit
  5.     pte_unmap_unlock(...)
  6. mmap_read_unlock(mm)
```

### A.3 The Cross-File Contention Problem

**`mmap_lock` is per-process, not per-file.** This is the critical insight.

#### Scenario: `mincore()` on File A blocks `mmap()` of File B

Consider an OpenSearch node with the MemoryOptimized search path:
- 50 mmap'd FAISS graph files (Lucene segments)
- Lucene is merging segments, which requires opening new segment files (`mmap`) and closing
  old ones (`munmap`)
- A stats API call invokes `mincore()` on one graph file to check residency

```
Timeline:
─────────────────────────────────────────────────────────────────────────

Thread 1 (Stats):        Thread 2 (Lucene merge):

mincore(graph_A, ...)
  │
  ├─ mmap_read_lock()
  │  [ACQUIRED - read]
  │                        wants to open new segment file:
  │                        mmap(segment_new, ...)
  │                          │
  │  walking page tables     ├─ mmap_write_lock()
  │  of graph_A ...          │  [BLOCKED - waiting for all readers]
  │                          │
  │  ... still walking ...   │  ... still blocked ...
  │                          │
  ├─ mmap_read_unlock()      │
  │                          ├─ [ACQUIRED - write]
  ├─ mmap_read_lock()        │  creates new VMA
  │  [BLOCKED - writer       │
  │   is pending]            ├─ mmap_write_unlock()
  │
  ├─ [ACQUIRED - read]
  │  walking next chunk...
  ...
```

**What happened:**
1. Thread 1 (stats) holds `mmap_read_lock` while walking page tables of `graph_A`.
2. Thread 2 (Lucene merge) tries to `mmap()` a completely unrelated new segment file. This
   requires `mmap_write_lock`, which must wait for ALL readers to release.
3. Thread 2 is **blocked** until Thread 1 finishes its current chunk.
4. Once Thread 2's write lock request is pending, Thread 1's **next** `mmap_read_lock` attempt
   is also blocked (the semaphore prevents new readers when a writer is waiting, to avoid
   writer starvation).

**The result**: calling `mincore()` on graph file A causes the opening of a completely unrelated
segment file B to stall. The two files have nothing to do with each other, but they share the
same `mmap_lock`.

#### This gets worse with multiple threads

In a real OpenSearch node:
- Multiple search threads are handling page faults (each takes `mmap_read_lock`)
- Lucene merge threads are opening/closing segments (`mmap_write_lock`)
- Background refresh creates new segments (`mmap_write_lock`)
- Warmup reads trigger page faults (`mmap_read_lock`)

Adding `mincore()` calls into this mix increases the **hold time** of `mmap_read_lock`, which
directly increases the wait time for any pending write operations.

### A.4 Quantifying the Impact

#### Cost per `mincore()` call

For a graph file of size `S`:

```
Number of pages       = S / 4KB
Number of chunks      = ceil(pages / 4096)
Lock acquisitions     = Number of chunks (for mmap_read_lock)
PTE spinlock acquires = pages / 512 (one per PMD, 512 PTEs per PMD)
```

| Graph size | Pages | Chunks (lock cycles) | PTE locks | Estimated wall time |
|-----------|-------|---------------------|-----------|-------------------|
| 100 MB | 25,600 | 7 | 50 | ~0.1 ms |
| 1 GB | 262,144 | 64 | 512 | ~1 ms |
| 10 GB | 2,621,440 | 640 | 5,120 | ~10 ms |
| 100 GB | 26,214,400 | 6,400 | 51,200 | ~100 ms |

These are optimistic estimates for the `mincore()` call itself. The **real cost** is the
cascading impact on other threads waiting for `mmap_write_lock`.

#### Cascading stall duration

When `mincore()` holds `mmap_read_lock` for one chunk (~16MB, ~microseconds to low
milliseconds), any pending `mmap()` or `munmap()` is stalled for that duration. But:

- If multiple search threads are also holding `mmap_read_lock` (for page faults), the write
  lock must wait for ALL of them.
- Once a write lock is pending, ALL new read lock attempts queue behind it.
- This creates a **convoy effect**: one `mincore()` call can trigger a chain of stalls across
  all threads.

#### Impact on Lucene segment lifecycle

Lucene's segment lifecycle operations that require `mmap_write_lock`:

| Operation | Frequency | Lock needed |
|-----------|-----------|-------------|
| Opening a new segment after merge | Per merge (minutes) | `mmap()` → write lock |
| Closing an old segment after merge | Per merge (minutes) | `munmap()` → write lock |
| Opening a new segment on refresh | Per refresh (seconds) | `mmap()` → write lock |
| Closing a deleted segment | On GC | `munmap()` → write lock |

On a busy node with frequent refreshes (default: every 1 second), there are constant
`mmap()`/`munmap()` calls. Any additional `mmap_read_lock` hold time from `mincore()` directly
increases the latency of these operations.

### A.5 Why `/proc/self/smaps` Has the Same Problem

Reading `smaps` also acquires `mmap_read_lock`:

```c
// fs/proc/task_mmu.c
static ssize_t smaps_read(...)
{
    mmap_read_lock(mm);
    // Walk ALL VMAs, walk ALL page tables
    // Report Rss, Pss, Shared_Clean, etc. for each
    mmap_read_unlock(mm);
}
```

This is actually **worse** than `mincore()` because:
- `smaps` walks page tables for **every** mapping in the process, not just the graphs.
- A process with thousands of mmap'd segments means walking thousands of VMAs.
- The kernel per-VMA lock patches (as of mid-2025) only apply to `/proc/pid/maps`, NOT to
  `/proc/pid/smaps`. The patch author explicitly states: *"similar approach would not work for
  /proc/pid/smaps reading as it also walks the page table and that's not RCU-safe."*

### A.6 Why There Is No Lock-Free Alternative

Every mechanism to query page residency requires walking page tables, which requires
stabilizing the VMA tree:

| Approach | Lock required | Cross-file contention? |
|----------|--------------|----------------------|
| `mincore()` | `mmap_read_lock` per chunk | Yes |
| `/proc/self/smaps` | `mmap_read_lock` for entire walk | Yes |
| `/proc/self/smaps_rollup` | `mmap_read_lock` | Yes |
| `move_pages()` | `mmap_read_lock` | Yes |
| `/proc/self/pagemap` | `mmap_read_lock` | Yes |

The Linux kernel does not provide any lock-free mechanism to query page residency. The
per-VMA lock work (Linux 6.x) reduces contention for `/proc/pid/maps` reads and page fault
handling, but does NOT help with page table walks needed for residency queries.

The fundamental reason: the kernel must ensure that the VMA tree and page tables are not being
modified while they are being read. Since `mmap()` and `munmap()` modify the VMA tree, and
residency queries read the page tables within VMAs, mutual exclusion is required. The
`mmap_lock` read/write semaphore is the mechanism that provides this.

---

## Appendix B: `cachestat()` Accuracy Analysis — Does `nr_cache` Equal Resident Pages?

The central claim of Option H is that for mmap'd files, `cachestat().nr_cache × PAGE_SIZE`
equals the amount of physical RAM consumed by the file. This appendix traces through the
kernel's page lifecycle to verify this claim and examines edge cases.

### B.1 How mmap'd File Pages Flow Through the Page Cache

When Lucene opens a file via `MMapDirectory`, the kernel calls `mmap()` which creates a VMA
pointing to the file's `address_space`. No pages are loaded yet — the xarray (`i_pages`) has
no entries for this range.

**Page fault (loading a page into RAM)**:

```
1. Process accesses address in mmap'd range
2. CPU raises page fault → kernel enters fault handler
3. Kernel looks up the file's address_space → checks xarray (i_pages)
4. Page NOT in xarray:
   a. Kernel allocates a folio (physical memory)
   b. Reads data from disk into the folio
   c. Inserts folio into the xarray                    ← nr_cache increases
   d. Maps folio into process page tables              ← process can now access it
5. Page IS in xarray (already cached):
   a. Maps existing folio into process page tables     ← no disk I/O needed
```

**Page eviction (OS reclaiming memory)**:

```
1. Kernel's reclaim path selects a folio for eviction
2. For clean, file-backed folios (read-only mmap'd graph files):
   a. Removes page table mappings via reverse mapping   ← process loses access
   b. Removes folio from xarray                         ← nr_cache decreases
   c. Stores a shadow entry in xarray                   ← nr_evicted increases
   d. Frees the physical memory
```

The key insight: **the xarray IS the source of truth for the page cache.** A folio in the
xarray = a folio in physical RAM. A shadow entry in the xarray = a folio that was evicted.
`cachestat()` walks this xarray, so its counts directly reflect physical memory state.

### B.2 Edge Case Analysis

#### Case 1: Readahead — Pages in Cache but Not Yet Mapped into the Process

When a page fault occurs, the kernel performs readahead — loading nearby pages into the page
cache proactively. These readahead pages are in the xarray (`nr_cache` counts them) but may
not yet have page table entries in the process.

**Impact on accuracy**: `cachestat()` counts these pages. This is the **correct behavior** for
our use case. These pages are consuming physical RAM on behalf of our graph file. Whether the
process has a page table entry for them is irrelevant — the memory is being used. In fact,
`mincore()` would NOT count these pages (since they're not in the process's page tables),
making `cachestat()` arguably more accurate for measuring actual RAM consumption.

#### Case 2: Pages Mapped into the Process but Not in the Page Cache

**Can this happen for file-backed mmap?** No. The page cache IS the backing store for
file-backed mmap. A page table entry for a file-backed mmap always points to a folio in the
page cache. When the kernel evicts a folio from the page cache, it also removes the page
table mapping (via the reverse mapping mechanism). The two are always in sync.

#### Case 3: Folios (Large Pages / Compound Pages)

Modern Linux (5.16+) uses "folios" — which can be larger than a single 4KB page. A folio
might be 16KB (order-2), 64KB (order-4), or even 2MB (order-9 for THP). The `cachestat()`
implementation correctly accounts for this:

```c
order = xa_get_order(xas.xa, xas.xa_index);
nr_pages = 1 << order;
// ... boundary adjustments for folios straddling the query range ...
cs->nr_cache += nr_pages;
```

It counts the number of base (4KB) pages within each folio. A 64KB folio in the cache adds
16 to `nr_cache`. This is correct — 64KB of physical RAM is being consumed.

#### Case 4: Transparent Huge Pages (THP) for File-Backed Mappings

If THP is enabled for file-backed mappings (possible on newer kernels with
`CONFIG_READ_ONLY_THP_FOR_FS`), a single folio could be 2MB. `cachestat()` handles this
identically to Case 3 — it counts base pages within the folio. The count remains accurate.

#### Case 5: Race Condition — Eviction During the `cachestat()` Call

The `cachestat()` man page states: *"the returned values may contain stale information."*
A folio could be evicted after `cachestat()` checks it but before the result is returned to
userspace. This is inherent to any point-in-time residency query and applies equally to
`mincore()`. For a stats API that reports periodically, this level of staleness is acceptable.

#### Case 6: Can `nr_cache` Overcount? (Pages in Cache but Swapped to Disk)

For **file-backed** pages, there is no "swap." Clean file-backed pages are simply dropped
from the page cache when reclaimed — the data can be re-read from the original file. They
are never written to swap space. So if a folio is in the xarray as a real folio (not a
shadow entry), it is in physical RAM. Period.

Dirty file-backed pages would be written back to disk first, then potentially reclaimed. But
k-NN graph files are mmap'd read-only (`PROT_READ`), so they are always clean. There is no
scenario where a graph file page is "in the page cache but not in RAM."

#### Case 7: Another Process or `read()` Loaded the Pages

If another process mmap'd the same graph file, or if a `read()` syscall was issued on the
file, those pages would be in the page cache. `cachestat()` would count them.

**Impact on accuracy**: This is the **correct behavior**. The pages are in physical RAM
backing our graph file. It doesn't matter which process or syscall caused them to be loaded.
The physical memory is consumed regardless.

#### Case 8: DAX (Direct Access) Filesystems

If the filesystem uses DAX (e.g., ext4 on persistent memory with `dax` mount option), pages
are directly mapped to persistent memory and bypass the page cache entirely. `cachestat()`
would not see these pages.

**Impact**: Not relevant for OpenSearch. OpenSearch uses ext4/xfs on regular block devices
(EBS, local NVMe), not DAX-enabled persistent memory.

### B.3 Summary

| Scenario | `nr_cache` = resident? | Notes |
|----------|----------------------|-------|
| Normal operation (after warmup) | ✅ Yes | All pages faulted in, all in xarray |
| Partial eviction under memory pressure | ✅ Yes | Evicted pages become shadow entries (`nr_evicted`), `nr_cache` decreases |
| Readahead loaded extra pages | ✅ Yes (correct) | Pages are in RAM, should be counted |
| Folios / THP (large pages) | ✅ Yes | `cachestat()` counts base pages within folios |
| Race condition during call | ⚠️ Slightly stale | Inherent to any point-in-time query, same as `mincore()` |
| Another process loaded pages | ✅ Yes (correct) | Pages are in RAM backing our file |
| Read-only mmap (our case) | ✅ Yes | Always clean, never swapped, eviction = removal from xarray |
| DAX filesystem | ❌ No | Not relevant — OpenSearch doesn't use DAX |

### B.4 Conclusion

For read-only mmap'd files on standard filesystems (ext4, xfs), `nr_cache` from `cachestat()`
is an accurate count of pages resident in physical RAM. The xarray that `cachestat()` walks is
the definitive source of truth for the page cache, and for file-backed mmap, being in the page
cache is equivalent to being in physical RAM.

The only recommended empirical validation is to compare `cachestat().nr_cache` against
`mincore()` results on the same file under controlled conditions (fully resident, partially
evicted, fully evicted) to confirm that the base-page counts match and there are no off-by-one
issues at folio boundaries.
