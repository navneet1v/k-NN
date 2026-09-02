"""
Locality Simulation for HNSW Graph Reordering Strategies.

Builds a Faiss HNSW index on a real dataset (Cohere 768D 1M vectors),
runs queries, records which nodes are visited per query, and measures
pages touched under different vector ordinal layouts:

1. Insertion order (baseline)
2. BFS ordering from entry point
3. Reverse Cuthill-McKee (bandwidth minimization)
4. Greedy page-growing (proposed algorithm) with three expansion strategies

Usage:
    pip install faiss-cpu numpy scipy h5py
    python locality_simulation.py --dataset scripts/documents-1m.hdf5
"""

import argparse
import heapq
import time
from collections import deque

import faiss
import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def load_dataset(path: str, max_train: int = None):
    """Load HDF5 dataset with train, test, and neighbors."""
    print(f"Loading dataset from {path}...")
    with h5py.File(path, "r") as f:
        print(f"  Keys: {list(f.keys())}")
        train = np.array(f["train"])
        test = np.array(f["test"])
        gt_neighbors = np.array(f["neighbors"]) if "neighbors" in f else None

        if max_train and max_train < train.shape[0]:
            train = train[:max_train]

        print(f"  Train: {train.shape}, Test: {test.shape}")
        if gt_neighbors is not None:
            print(f"  Ground truth neighbors: {gt_neighbors.shape}")
    return train, test, gt_neighbors


def build_hnsw_index(vectors: np.ndarray, M: int, ef_construction: int, shuffle: bool = True):
    """
    Build single-layer HNSW index with shuffled insertion order wrapped in IndexIDMap.
    Single-layer: all nodes at level 0, no upper layers (CAGRA-style flat graph).
    Returns (index, insertion_order) where insertion_order[i] = original vector ID
    that was inserted at position i.

    When shuffle=False the insertion order is the dataset's natural order (0..n-1), i.e.
    internal ordinal == original id. Use this to model an ingest whose doc order already
    follows vector-space locality (as OpenSearch does when fed a pre-clustered dataset),
    vs the default shuffle which models the realistic worst case of random doc arrival.
    """
    dim = vectors.shape[1]
    n = vectors.shape[0]

    if shuffle:
        # Shuffle insertion order so insertion order != semantic order
        rng = np.random.default_rng(seed=42)
        insertion_order = rng.permutation(n).astype(np.int64)
    else:
        # Natural order: insertion order == semantic order (no decorrelation)
        insertion_order = np.arange(n, dtype=np.int64)

    base_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    base_index.hnsw.efConstruction = ef_construction

    # Force single-layer: all nodes at level 0, no upper layers
    base_index.hnsw.assign_probas.clear()
    base_index.hnsw.assign_probas.push_back(1.0)

    index = faiss.IndexIDMap(base_index)

    # Insert vectors in shuffled order with their original IDs
    shuffled_vectors = vectors[insertion_order]
    index.add_with_ids(shuffled_vectors, insertion_order)

    return index, insertion_order


def get_base_hnsw_index(index):
    """Unwrap IndexIDMap to get the underlying IndexHNSWFlat."""
    if hasattr(index, 'index'):
        return faiss.downcast_index(index.index)
    return index


def extract_layer0_neighbors(index, n: int, M: int) -> list[list[int]]:
    """Extract layer 0 adjacency lists from a Faiss HNSW index.
    Neighbor IDs are internal ordinals (0..N-1), NOT original vector IDs."""
    base = get_base_hnsw_index(index)
    hnsw = base.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets)
    neighbors_table = faiss.vector_to_array(hnsw.neighbors)
    cum_nb = faiss.vector_to_array(hnsw.cum_nneighbor_per_level)

    neighbors = []
    for i in range(n):
        b = int(offsets[i]) + int(cum_nb[0])
        e = int(offsets[i]) + int(cum_nb[1])
        nbrs = neighbors_table[b:e]
        neighbor_list = [int(x) for x in nbrs if x >= 0]
        neighbors.append(neighbor_list)
    return neighbors


def extract_levels(index, n: int) -> np.ndarray:
    """Extract the HNSW level of each node. Returns levels for internal ordinals 0..N-1."""
    base = get_base_hnsw_index(index)
    hnsw = base.hnsw
    levels = faiss.vector_to_array(hnsw.levels).astype(np.int32) - 1
    return levels[:n]


def simulate_flat_search(
    neighbors: list[list[int]],
    vectors: np.ndarray,
    query: np.ndarray,
    entry_points: list[int],
    ef_search: int,
) -> list[int]:
    """
    Simulate greedy beam search on layer 0 only (CAGRA-style flat search).
    Returns the list of visited node ordinals in visit order.
    """
    visited = set()
    visit_order = []

    candidates = []  # min-heap by distance (negative IP = smaller is better)
    results = []  # max-heap by -distance

    def dist(node_id):
        return -float(np.dot(vectors[node_id], query))

    for ep in entry_points:
        if ep not in visited:
            visited.add(ep)
            visit_order.append(ep)
            d = dist(ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

    while candidates:
        d_candidate, candidate = heapq.heappop(candidates)

        if results:
            d_furthest = -results[0][0]
        else:
            d_furthest = float("inf")

        if d_candidate > d_furthest and len(results) >= ef_search:
            break

        for neighbor in neighbors[candidate]:
            if neighbor not in visited:
                visited.add(neighbor)
                visit_order.append(neighbor)
                d = dist(neighbor)

                if len(results) < ef_search or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))

                    while len(results) > ef_search:
                        heapq.heappop(results)

    return visit_order


def simulate_multilayer_search(
    index,
    vectors: np.ndarray,
    query: np.ndarray,
    ef_search: int,
) -> list[int]:
    """
    Simulate multi-layer HNSW search (standard algorithm).
    Descends from entry point through upper layers, then beam search on layer 0.
    vectors should be ordered by internal ordinal (insertion order).
    Returns the list of ALL visited internal ordinals across all layers.
    """
    base = get_base_hnsw_index(index)
    hnsw = base.hnsw
    offsets = faiss.vector_to_array(hnsw.offsets)
    neighbors_table = faiss.vector_to_array(hnsw.neighbors)
    cum_nb = faiss.vector_to_array(hnsw.cum_nneighbor_per_level)
    levels = faiss.vector_to_array(hnsw.levels) - 1  # Faiss stores level+1

    entry_point = hnsw.entry_point
    max_level = int(levels[entry_point])

    visited = set()
    visit_order = []

    def dist(node_id):
        return -float(np.dot(vectors[node_id], query))

    def get_neighbors_at_level(node, level):
        b = int(offsets[node]) + int(cum_nb[level])
        e = int(offsets[node]) + int(cum_nb[level + 1])
        nbrs = neighbors_table[b:e]
        return [int(x) for x in nbrs if x >= 0]

    # Phase 1: Greedy descent through upper layers (efSearch=1 per layer)
    current = entry_point
    visited.add(current)
    visit_order.append(current)

    for level in range(max_level, 0, -1):
        improved = True
        while improved:
            improved = False
            nbrs = get_neighbors_at_level(current, level)
            for nbr in nbrs:
                if nbr not in visited:
                    visited.add(nbr)
                    visit_order.append(nbr)
                if dist(nbr) < dist(current):
                    current = nbr
                    improved = True

    # Phase 2: Beam search on layer 0 with efSearch
    candidates = []  # min-heap
    results = []  # max-heap

    d_current = dist(current)
    if current not in visited:
        visited.add(current)
        visit_order.append(current)
    heapq.heappush(candidates, (d_current, current))
    heapq.heappush(results, (-d_current, current))

    while candidates:
        d_candidate, candidate = heapq.heappop(candidates)

        if results:
            d_furthest = -results[0][0]
        else:
            d_furthest = float("inf")

        if d_candidate > d_furthest and len(results) >= ef_search:
            break

        nbrs = get_neighbors_at_level(candidate, 0)
        for neighbor in nbrs:
            if neighbor not in visited:
                visited.add(neighbor)
                visit_order.append(neighbor)
                d = dist(neighbor)

                if len(results) < ef_search or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))

                    while len(results) > ef_search:
                        heapq.heappop(results)

    return visit_order


def compute_pages_touched(visit_order: list[int], permutation: np.ndarray, page_capacity: int) -> int:
    """Given a visit order and a permutation, count distinct pages touched."""
    pages_seen = set()
    for node in visit_order:
        physical_ord = permutation[node]
        page_id = physical_ord // page_capacity
        pages_seen.add(page_id)
    return len(pages_seen)


def compute_recall(visit_order: list[int], gt_neighbors: np.ndarray, k: int, max_id: int = None) -> float:
    """
    Compute recall@k given visited nodes and ground truth.
    If max_id is set, only ground truth neighbors with id < max_id are considered
    (needed when using a subset of the full dataset).
    """
    if max_id is not None:
        gt_valid = [int(x) for x in gt_neighbors if x < max_id][:k]
    else:
        gt_valid = [int(x) for x in gt_neighbors[:k]]

    if len(gt_valid) == 0:
        return 0.0

    gt_set = set(gt_valid)
    found = sum(1 for node in visit_order if node in gt_set)
    return found / len(gt_set)


# --- Reordering Strategies ---


def identity_permutation(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int32)


def bfs_permutation(neighbors: list[list[int]], entry_point: int, n: int) -> np.ndarray:
    """BFS ordering from the HNSW entry point."""
    visited = np.full(n, False)
    permutation = np.full(n, -1, dtype=np.int32)
    order = 0

    queue = deque([entry_point])
    visited[entry_point] = True

    while queue:
        node = queue.popleft()
        permutation[node] = order
        order += 1

        for neighbor in neighbors[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    for i in range(n):
        if not visited[i]:
            permutation[i] = order
            order += 1

    return permutation


def rcm_permutation(neighbors: list[list[int]], n: int) -> np.ndarray:
    """Reverse Cuthill-McKee ordering."""
    rows, cols = [], []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.int8)
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    rcm_order = reverse_cuthill_mckee(adj_matrix)

    permutation = np.empty(n, dtype=np.int32)
    for new_ord, old_ord in enumerate(rcm_order):
        permutation[old_ord] = new_ord

    return permutation


def greedy_page_growing_permutation(
    neighbors: list[list[int]],
    levels: np.ndarray,
    n: int,
    page_capacity: int,
    expansion: str = "partial_page",
) -> np.ndarray:
    """
    Greedy page-growing with hub-based seeding and STRICT PAGE BOUNDARIES.

    Each page occupies exactly `page_capacity` ordinal slots. When a page closes
    early (no more affine candidates), the ordinal counter is padded to the next
    page boundary — leaving gaps that map to no vector (zero-filled on disk).
    This guarantees every logical page maps to exactly one physical 32KB page,
    which matters because pages also carry graph edges, correction factors, and
    metadata — not just vectors.

    Uses bucket queue for O(1) best-candidate selection and incremental affinity
    updates for O(degree) per slot fill.

    Returns a permutation array of length n where permutation[oldOrd] = newOrd.
    Note: newOrd values are NOT dense — they skip padded slots at page ends.
    Physical file size = num_pages * page_capacity slots.
    """
    assigned = np.full(n, False)
    permutation = np.full(n, -1, dtype=np.int32)
    affinity = np.zeros(n, dtype=np.int32)
    is_candidate = np.full(n, False)
    next_ord = 0  # physical ordinal (includes padding gaps)

    degree = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int32)
    # Seed pages from highest-degree nodes (most connected = best hub candidates)
    hub_indices = np.argsort(-degree)
    max_bucket = int(degree.max()) + 1

    def grow_page(seed: int) -> int:
        nonlocal next_ord
        if assigned[seed]:
            return 0

        page = []
        candidates_to_clear = []
        buckets = [[] for _ in range(max_bucket)]
        top_bucket = 0

        def add_candidate(node):
            is_candidate[node] = True
            candidates_to_clear.append(node)
            score = int(affinity[node])
            buckets[score].append(node)
            nonlocal top_bucket
            if score > top_bucket:
                top_bucket = score

        def add_to_page(node):
            nonlocal next_ord
            assigned[node] = True
            page.append(node)
            permutation[node] = next_ord
            next_ord += 1
            if is_candidate[node]:
                is_candidate[node] = False

            for nbr in neighbors[node]:
                if assigned[nbr]:
                    continue
                old_score = int(affinity[nbr])
                affinity[nbr] = old_score + 1
                new_score = old_score + 1
                if is_candidate[nbr]:
                    buckets[new_score].append(nbr)
                    nonlocal top_bucket
                    if new_score > top_bucket:
                        top_bucket = new_score
                else:
                    add_candidate(nbr)

        def pop_best():
            nonlocal top_bucket
            while top_bucket >= 0:
                bucket = buckets[top_bucket]
                while bucket:
                    candidate = bucket.pop()
                    if is_candidate[candidate] and int(affinity[candidate]) == top_bucket:
                        return candidate
                top_bucket -= 1
            return -1

        # Start of this page must be page-aligned
        page_start = next_ord
        assert page_start % page_capacity == 0, "page must start on a boundary"

        add_to_page(seed)

        while len(page) < page_capacity:
            best = pop_best()
            if best < 0:
                # No affine candidates left — close the page early (partial_page)
                break
            is_candidate[best] = False
            add_to_page(best)

        # STRICT BOUNDARY: pad next_ord to the next page boundary.
        # Padded slots [page_start + len(page), page_start + page_capacity) are
        # unused (zero-filled on disk).
        next_ord = page_start + page_capacity

        # Cleanup for next page
        for node in page:
            for nbr in neighbors[node]:
                affinity[nbr] = 0
        for c in candidates_to_clear:
            is_candidate[c] = False
            affinity[c] = 0

        return len(page)

    for hub in hub_indices:
        if not assigned[hub]:
            grow_page(hub)

    for i in range(n):
        if not assigned[i]:
            grow_page(i)

    return permutation


def run_simulation(args):
    print("=" * 70)
    print("  HNSW Locality Reordering Simulation")
    print("=" * 70)
    print()

    # Load dataset
    train, test, gt_neighbors = load_dataset(args.dataset, max_train=args.max_train)
    n = train.shape[0]
    dim = train.shape[1]
    # nq <= 0 means "use all queries in the dataset"
    nq = test.shape[0] if args.nq <= 0 else min(args.nq, test.shape[0])
    queries = test[:nq]

    # Compute record size (SQ 1-bit: dim/8 bytes binary code + 16 bytes correction)
    record_size = (dim + 7) // 8 + 16
    page_capacity = args.page_size // record_size
    print(f"\nConfig:")
    print(f"  Vectors: {n}, Dim: {dim}, Queries: {nq}")
    print(f"  HNSW M: {args.M}, efConstruction: {args.ef_construction}, efSearch: {args.ef_search}")
    print(f"  Page size: {args.page_size} bytes")
    print(f"  Record size (SQ 1-bit): {record_size} bytes")
    print(f"  Page capacity: {page_capacity} vectors/page")
    print(f"  Total pages: {(n + page_capacity - 1) // page_capacity}")
    print(f"  Entry points per query: {args.num_entry_points}")
    print()

    # Build single-layer HNSW index with shuffled insertion order
    print("Building single-layer HNSW index (shuffled insertion order)...")
    t0 = time.time()
    index, insertion_order = build_hnsw_index(train, args.M, args.ef_construction, shuffle=not args.no_shuffle)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")
    print(f"  Insertion order shuffled (first 5 internal ordinals map to original IDs: {insertion_order[:5]})")

    # Reorder vectors to match internal ordinal order.
    # Internal ordinal i corresponds to original vector ID insertion_order[i].
    # For distance computation during simulated search, we need vectors indexed by internal ordinal.
    vectors_by_internal_ord = train[insertion_order]

    # Extract graph (uses internal ordinals 0..N-1)
    print("Extracting graph structure...")
    t0 = time.time()
    neighbors_list = extract_layer0_neighbors(index, n, args.M)
    levels = extract_levels(index, n)
    base_index = get_base_hnsw_index(index)
    entry_point = base_index.hnsw.entry_point
    extract_time = time.time() - t0
    print(f"  Extracted in {extract_time:.1f}s")
    print(f"  Entry point: {entry_point} (level {levels[entry_point]})")
    print(f"  Max level: {levels.max()}")
    print(f"  Level distribution: {dict(zip(*np.unique(levels, return_counts=True)))}")
    degrees = np.array([len(nbrs) for nbrs in neighbors_list])
    print(f"  Layer 0 degree — avg: {degrees.mean():.1f}, p10: {np.percentile(degrees, 10):.0f}, "
          f"p50: {np.percentile(degrees, 50):.0f}, p90: {np.percentile(degrees, 90):.0f}, "
          f"p99: {np.percentile(degrees, 99):.0f}, max: {degrees.max()}")
    print()

    # Compute permutations
    print("Computing reordering strategies...")
    permutations = {}

    t0 = time.time()
    permutations["Insertion Order"] = identity_permutation(n)
    print(f"  Identity: {time.time() - t0:.3f}s")

    t0 = time.time()
    permutations["BFS Order"] = bfs_permutation(neighbors_list, entry_point, n)
    print(f"  BFS: {time.time() - t0:.3f}s")

    t0 = time.time()
    permutations["Reverse Cuthill-McKee"] = rcm_permutation(neighbors_list, n)
    print(f"  RCM: {time.time() - t0:.3f}s")

    t0 = time.time()
    permutations["Greedy (strict pages)"] = greedy_page_growing_permutation(
        neighbors_list, levels, n, page_capacity, "partial_page"
    )
    print(f"  Greedy (strict pages): {time.time() - t0:.3f}s")

    # Report padding overhead from strict page boundaries
    greedy_perm = permutations["Greedy (strict pages)"]
    num_pages_used = int(greedy_perm.max()) // page_capacity + 1
    padding_slots = num_pages_used * page_capacity - n
    print(f"  Strict pages: {num_pages_used} pages, {padding_slots} padded slots "
          f"({100*padding_slots/(num_pages_used*page_capacity):.1f}% overhead)")

    print()

    # --- Faiss entry-point search (uses Faiss's own entry point selection) ---
    print(f"Simulating {nq} searches with Faiss entry point (efSearch={args.ef_search})...")
    multilayer_visits = []
    t0 = time.time()
    for qi in range(nq):
        visit_order = simulate_multilayer_search(index, vectors_by_internal_ord, queries[qi], args.ef_search)
        multilayer_visits.append(visit_order)
        if (qi + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {qi+1}/{nq} queries ({elapsed:.1f}s, {(qi+1)/elapsed:.1f} qps)")

    search_time = time.time() - t0
    avg_visited_ml = np.mean([len(v) for v in multilayer_visits])
    print(f"  Completed in {search_time:.1f}s, avg nodes visited: {avg_visited_ml:.1f}")

    if gt_neighbors is not None:
        recalls_ml = []
        for qi in range(nq):
            # Map internal ordinals back to original IDs for recall comparison
            visited_original_ids = [int(insertion_order[x]) for x in multilayer_visits[qi]]
            r = compute_recall(visited_original_ids, gt_neighbors[qi], k=10, max_id=n)
            recalls_ml.append(r)
        print(f"  Recall@10 (Faiss entry point): {np.mean(recalls_ml):.4f}")
    print()

    # Also validate recall using Faiss's own search (sanity check)
    print("Faiss native search (sanity check)...")
    base_index.hnsw.efSearch = args.ef_search
    _, I = index.search(queries, 10)
    if gt_neighbors is not None:
        faiss_recalls = []
        for qi in range(nq):
            gt_valid = [int(x) for x in gt_neighbors[qi] if x < n][:10]
            if gt_valid:
                faiss_recalls.append(len(set(I[qi].tolist()) & set(gt_valid)) / len(gt_valid))
        faiss_recall = np.mean(faiss_recalls) if faiss_recalls else 0.0
        print(f"  Recall@10 (Faiss native): {faiss_recall:.4f}")
    print()

    # --- Flat search with hub-based entry points ---
    # With single-layer graph, all nodes are at level 0.
    # Select hubs by degree (highest-degree nodes are most connected, best entry points).
    hub_candidates = np.argsort(-degrees)
    num_hubs = min(256, n)
    hub_pool = hub_candidates[:num_hubs].tolist()
    hub_vectors = vectors_by_internal_ord[hub_pool]
    print(f"Hub pool: {num_hubs} nodes (degree range: {degrees[hub_pool[0]]} down to {degrees[hub_pool[-1]]})")

    entry_point_counts = [args.num_entry_points]
    flat_visits_by_ep = {}

    for num_ep in entry_point_counts:
        print(f"Simulating {nq} FLAT searches (efSearch={args.ef_search}, {num_ep} hub entry points)...")
        flat_visits = []
        t0 = time.time()
        for qi in range(nq):
            # Pick closest hubs to query by inner product
            scores = hub_vectors @ queries[qi]
            top_hub_indices = np.argsort(-scores)[:num_ep]
            entry_points = [hub_pool[i] for i in top_hub_indices]
            visit_order = simulate_flat_search(neighbors_list, vectors_by_internal_ord, queries[qi], entry_points, args.ef_search)
            flat_visits.append(visit_order)
            if (qi + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {qi+1}/{nq} queries ({elapsed:.1f}s, {(qi+1)/elapsed:.1f} qps)")

        search_time = time.time() - t0
        avg_visited = np.mean([len(v) for v in flat_visits])

        recall_str = ""
        if gt_neighbors is not None:
            recalls = []
            for qi in range(nq):
                visited_original_ids = [int(insertion_order[x]) for x in flat_visits[qi]]
                recalls.append(compute_recall(visited_original_ids, gt_neighbors[qi], k=10, max_id=n))
            recall_str = f", Recall@10: {np.mean(recalls):.4f}"

        print(f"  {search_time:.1f}s, avg visited: {avg_visited:.1f}{recall_str}")
        flat_visits_by_ep[num_ep] = flat_visits

    print()

    # --- Measure pages touched ---
    # Use multi-layer as the primary baseline, then show flat with best recall match
    print("=" * 80)
    print("PAGES TOUCHED: Faiss entry point search (baseline)")
    print("=" * 80)
    print(f"{'Layout':<30} {'Avg Pages':>10} {'Median':>8} {'P95':>8} {'P99':>8}")
    print("-" * 80)

    ml_baseline_avg = None
    for name, perm in permutations.items():
        pages_per_query = [compute_pages_touched(v, perm, page_capacity) for v in multilayer_visits]
        pages_arr = np.array(pages_per_query)
        avg = pages_arr.mean()
        if ml_baseline_avg is None:
            ml_baseline_avg = avg
        improvement = ((ml_baseline_avg - avg) / ml_baseline_avg * 100) if ml_baseline_avg > 0 else 0
        suffix = f" ({improvement:+.1f}%)" if name != "Insertion Order" else ""
        print(
            f"{name:<30} {avg:>8.1f}{suffix:<10} {np.median(pages_arr):>8.1f} "
            f"{np.percentile(pages_arr, 95):>8.1f} {np.percentile(pages_arr, 99):>8.1f}"
        )
    print()

    for num_ep in entry_point_counts:
        flat_visits = flat_visits_by_ep[num_ep]
        print("=" * 80)
        print(f"PAGES TOUCHED: Flat search ({num_ep} entry points)")
        print("=" * 80)
        print(f"{'Layout':<30} {'Avg Pages':>10} {'Median':>8} {'P95':>8} {'P99':>8}")
        print("-" * 80)

        flat_baseline_avg = None
        for name, perm in permutations.items():
            pages_per_query = [compute_pages_touched(v, perm, page_capacity) for v in flat_visits]
            pages_arr = np.array(pages_per_query)
            avg = pages_arr.mean()
            if flat_baseline_avg is None:
                flat_baseline_avg = avg
            improvement = ((flat_baseline_avg - avg) / flat_baseline_avg * 100) if flat_baseline_avg > 0 else 0
            suffix = f" ({improvement:+.1f}%)" if name != "Insertion Order" else ""
            print(
                f"{name:<30} {avg:>8.1f}{suffix:<10} {np.median(pages_arr):>8.1f} "
                f"{np.percentile(pages_arr, 95):>8.1f} {np.percentile(pages_arr, 99):>8.1f}"
            )
        print()

    # Intra-page edge ratio
    print("=" * 80)
    print("INTRA-PAGE EDGE RATIO (edges staying within same page)")
    print("=" * 80)
    for name, perm in permutations.items():
        intra_edges = 0
        total_edges = 0
        for node in range(n):
            node_page = perm[node] // page_capacity
            for nbr in neighbors_list[node]:
                total_edges += 1
                if perm[nbr] // page_capacity == node_page:
                    intra_edges += 1
        ratio = intra_edges / total_edges if total_edges > 0 else 0
        print(f"  {name:<30} {ratio:.4f} ({ratio*100:.1f}%)")

    print()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HNSW Locality Reordering Simulation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset file")
    parser.add_argument("--max-train", type=int, default=None, help="Limit training vectors (for quick tests)")
    parser.add_argument("--nq", type=int, default=1000, help="Number of queries (<=0 means all queries in dataset)")
    parser.add_argument("--M", type=int, default=32, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction")
    parser.add_argument("--ef-search", type=int, default=64, help="HNSW efSearch")
    parser.add_argument("--page-size", type=int, default=32768, help="Page size in bytes")
    parser.add_argument("--num-entry-points", type=int, default=1, help="Hub entry points for flat search")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="insert vectors in natural (dataset) order instead of shuffling; models an ingest whose "
        "doc order already follows vector-space locality (removes the reordering's headroom)",
    )
    args = parser.parse_args()
    run_simulation(args)
